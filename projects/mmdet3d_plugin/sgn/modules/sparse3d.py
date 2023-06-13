import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
import numpy as np

class BasicBlock(spconv.SparseModule):
    def __init__(self, C_in, C_out, indice_key):
        super(BasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(C_out, C_out, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out)
        )
        self.relu2 = spconv.SparseSequential(
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        identity = self.layers_in(x)
        out = self.layers(x)
        output = spconv.SparseConvTensor(sum([i.features for i in [identity, out]]),
                                         out.indices, out.spatial_shape, out.batch_size)
        output.indice_dict = out.indice_dict
        output.grid = out.grid
        return self.relu2(output)


def make_layers_sp(C_in, C_out, blocks, indice_key):
    layers = []
    layers.append(BasicBlock(C_in, C_out, indice_key))
    for _ in range(1, blocks):
        layers.append(BasicBlock(C_out, C_out, indice_key))
    return spconv.SparseSequential(*layers)


def scatter(x, idx, method, dim=0):
    if method == "max":
        return torch_scatter.scatter_max(x, idx, dim=dim)[0]
    elif method == "mean":
        return torch_scatter.scatter_mean(x, idx, dim=dim)
    elif method == "sum":
        return torch_scatter.scatter_add(x, idx, dim=dim)
    else:
        print("unknown method")
        exit(-1)


class SFE(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, layer_name, layer_num=2):
        super().__init__()
        self.spconv_layers = make_layers_sp(in_channels, out_channels, layer_num, layer_name)

    def forward(self, inputs):
        conv_features = self.spconv_layers(inputs)
        return conv_features


class SGFE(nn.Module):
    def __init__(self, input_channels, output_channels, reduce_channels, name, p_scale=[2, 4, 6, 8]):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name

        self.feature_reduce = nn.Linear(input_channels, reduce_channels)
        self.pooling_scale = p_scale
        self.fc_list = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _, _ in enumerate(self.pooling_scale):
            self.fc_list.append(nn.Sequential(
            nn.Linear(reduce_channels, reduce_channels//2),
            nn.ReLU(),
            ))
            self.fcs.append(nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2)))
        self.scale_selection = nn.Sequential(
            nn.Linear(len(self.pooling_scale) * reduce_channels//2,
                                       reduce_channels),nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2, bias=False),
                                nn.ReLU(inplace=False))
        self.out_fc = nn.Linear(reduce_channels//2, reduce_channels, bias=False)
        self.linear_output = nn.Sequential(
            nn.Linear(2 * reduce_channels, reduce_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduce_channels, output_channels),
        )

    def forward(self, input_data, input_coords):
        reduced_feature = F.relu(self.feature_reduce(input_data))
        output_list = [reduced_feature]
        for j, ps in enumerate(self.pooling_scale):
            index = torch.cat([input_coords[:, 0].unsqueeze(-1),
                              (input_coords[:, 1:] // ps).int()], dim=1)
            _, unq_inv = torch.unique(index, return_inverse=True, dim=0)
            fkm = scatter(reduced_feature, unq_inv, method="mean", dim=0)
            att = self.fc_list[j](fkm)[unq_inv]
            output_list.append(att)
        scale_features = torch.stack(output_list[1:], dim=1)
        feat_S = scale_features.sum(1)
        feat_Z = self.fc(feat_S)
        attention_vectors = [fc(feat_Z) for fc in self.fcs]
        attention_vectors = torch.sigmoid(torch.stack(attention_vectors, dim=1))
        scale_features = self.out_fc(torch.sum(scale_features * attention_vectors, dim=1))

        output_f = torch.cat([reduced_feature, scale_features], dim=1)
        proj = self.linear_output(output_f)
        
        return proj


class Sparse3d(nn.Module):
    def __init__(self, channels=128, p_scale_1=[2, 4, 6, 8], p_scale_2=[2, 4, 6, 8]):
        super().__init__()
        c = channels
        self.conv_block = SFE(c, c, "svpfe")
        self.proj_block = SGFE(input_channels=c, output_channels=c, reduce_channels=c, name="proj", p_scale=p_scale_1)

        self.conv_block_down = SFE(c, c, "svpfe_down")
        self.proj_block_down = SGFE(input_channels=c, output_channels=c, reduce_channels=c, name="proj_down", p_scale=p_scale_2)

        self.fc_out = nn.Linear(2*c, c)

    def forward(self, input_tensor, sizes):
        coord_ind = input_tensor.indices
        batch_size = torch.unique(coord_ind[:,0]).max() + 1

        conv_output = self.conv_block(input_tensor)
        proj_vw1 = self.proj_block(conv_output.features, input_coords=coord_ind.int())

        # downsample
        index = torch.cat([coord_ind[:, 0].unsqueeze(-1),
                          (coord_ind[:, 1:] // 2).int()], dim=1)
        unq, unq_inv = torch.unique(index, return_inverse=True, dim=0)
        proj_vw1_down = scatter(proj_vw1, unq_inv, method="max", dim=0)

        input_tensor_down = spconv.SparseConvTensor(
            proj_vw1_down, unq.int(), np.array(sizes, np.int32)//2, batch_size
        )
        conv_output_down = self.conv_block_down(input_tensor_down)
        proj_vw2 = self.proj_block_down(conv_output_down.features, input_coords=unq.int())

        proj_vw = self.fc_out(torch.cat([proj_vw1, proj_vw2[unq_inv]], dim=-1))

        output_tensor = spconv.SparseConvTensor(
                proj_vw, coord_ind.int(), np.array(sizes, np.int32), batch_size)

        return output_tensor
