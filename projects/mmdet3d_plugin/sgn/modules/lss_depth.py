# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
import torch.nn as nn
from mmdet3d.ops.bev_pool import bev_pool
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
from mmdet.models.backbones.resnet import BasicBlock
import torch.nn.functional as F
import numpy as np


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])  # voxel size
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])  # voxel min
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]) # voxel num
    return dx, bx, nx


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        depth_channels,
        infer_mode=False,
    ):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp = Mlp(1, mid_channels, mid_channels)
        self.se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )

        self.depth_pred = nn.Conv2d(
            mid_channels, depth_channels, kernel_size=1, stride=1, padding=0
        )
        self.infer_mode = infer_mode

    def forward(
        self,
        x,
        intrins,
        scale_depth_factor=1000.0,
    ):
        inv_intrinsics = torch.inverse(intrins)
        pixel_size = torch.norm(
            torch.stack(
                [inv_intrinsics[..., 0, 0], inv_intrinsics[..., 1, 1]], dim=-1
            ),
            dim=-1,
        ).reshape(-1, 1)
        scaled_pixel_size = pixel_size * scale_depth_factor

        x = self.reduce_conv(x)
        x_se = self.mlp(scaled_pixel_size)[..., None, None]

        x = self.se(x, x_se)
        x = self.depth_conv(x)
        depth = self.depth_pred(x)

        return depth


class LSSDepth(nn.Module):
    def __init__(self, grid_config, data_config, depth_config, downsample=16):
        super().__init__()
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.img_size = data_config['input_size']
        self.downsample = downsample

        self.points = self.create_points()
        self.d_min, self.d_max, self.d_size = self.grid_config['dbound']
        self.D = (self.d_max-self.d_min)/self.d_size
        self.depth_net = DepthNet(**depth_config, depth_channels=self.D)
    
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        fH, fW = int(math.ceil(H / self.downsample)), int(math.ceil(W / self.downsample))
        gt_depths = F.interpolate(gt_depths, (fH*self.downsample, fW*self.downsample), mode='nearset')

        gt_depths = gt_depths.view(B * N,
                                   fH, self.downsample,
                                   fW, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, fH, fW)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths_vals, gt_depths.float()
    
    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return depth_loss

    def create_points(self):
        xv, yv, zv = np.meshgrid(range(int(self.nx[0])), range(int(self.nx[1])), range(int(self.nx[2])), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1)], axis=0).astype(int).T
        vox_coords = torch.from_numpy(vox_coords)
        # N x 3
        voxels = vox_coords * self.dx[None] + self.bx[None]
        
        return nn.Parameter(voxels, requires_grad=False)

    def get_grids(self, projmats):
        # B, N, 4, 4
        ones = torch.ones(
            (len(self.points), 1), device=self.points.device, dtype=torch.float32
        )
        points_h = torch.cat((self.points, ones), dim=-1) # [nq, 4]

        im_p = projmats.float() @ points_h.t().float()  # [bs, nv, 4, 4] @ [4, nq] -> [bs, nv, 4, nq]
        im_x, im_y, im_z = im_p[:, :, 0], im_p[:, :, 1], im_p[:, :, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z
        im_grid = torch.stack(
            [2 * im_x / (self.img_size[1] - 1) - 1, 2 * im_y / (self.img_size[0] - 1) - 1],
            dim=-1
        )  # [bs, nv, nq, 2], (-1, 1)
        im_grid[torch.isinf(im_grid)] = -2

        depth = (im_z-self.d_min)/self.D
        depth = (2*depth / (self.D-1) - 1).unsqueeze(-1)
        depth[torch.isinf(depth)] = -2

        grids = torch.cat([im_grid, depth], dim=-1) # [bs, nv, nq, 3]
        
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)  #[bs, nv, nq]
        
        return grids, mask

    def grid_sampling(self, volum, grid, mask):
        B, N = volum.shape[:2]
        vox_prob = F.grid_sample(
            volum.flatten(0, 1)[:, None],  # bn, 1, d, h, w
            grid.flatten(0, 1)[:, None, None],  # bn, 1, 1, nq, 3
            padding_mode="zeros",
            mode="nearest",
            align_corners=False
        )[:, 0, 0, 0]  # bn, nq

        vox_prob = vox_prob.view(B, N, -1)  # b, n, nq
        weights = torch.sum(mask, dim=1) # b, nq
        weights[weights == 0] = 1
        vox_prob = torch.sum(vox_prob * mask, dim=1) / weights
        vox_prob = vox_prob.view(B, int(self.nx[0])), int(self.nx[1]), int(self.nx[2])

        return vox_prob

    def forward(self, input):
        (x, intrins, projmats) = input[:3]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        depth_feature = self.depth_net(x, intrins)
        depth_probs = depth_feature.softmax(1)

        volum = depth_probs.view(B, N, self.D, H, W)  # B, N, D, H, W

        grids, mask = self.get_grids(projmats)
        vox_prob = self.grid_sampling(volum, grids, mask)  # B, x, y, z
        
        return vox_prob, depth_probs