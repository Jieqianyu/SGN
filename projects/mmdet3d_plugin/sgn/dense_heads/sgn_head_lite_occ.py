import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from mmdet.models import HEADS
from projects.mmdet3d_plugin.sgn.modules.sparse3d import Sparse3d

@HEADS.register_module()
class SGNHeadOccLite(nn.Module):
    def __init__(
        self,
        *args,
        scene_size,
        voxel_origin,
        voxel_size,
        embed_dims,
        img_size,
        num_level,
        **kwargs
    ):
        super().__init__()
        self.scene_size = np.array(scene_size)
        self.voxel_origin = np.array(voxel_origin)
        self.voxel_size = voxel_size
        self.num_level = num_level
        self.imheight, self.imwidth = img_size[0], img_size[1]

        self.output_layers = nn.ModuleDict()
        self.net3ds = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict()
        self.reductions = nn.ModuleDict()
        p_scale_dict = {
            0: [[2, 4], [2]],
            1: [[2, 4, 4], [2, 4]],
            2: [[2, 4, 8, 8], [2, 4, 8]]
        }
        for _iter in range(self.num_level):
            self.layer_norms[f'norm_{_iter}'] = nn.LayerNorm(embed_dims)
            in_ch = 2*embed_dims + 1 if _iter > 0 else embed_dims
            self.reductions[f'reduction_{_iter}'] = nn.Sequential(
                nn.Linear(in_ch, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.LeakyReLU(0.1),
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.LeakyReLU(0.1),
            )
            self.net3ds[f'net3d_{_iter}'] = Sparse3d(embed_dims, p_scale_1=p_scale_dict[_iter][0], p_scale_2=p_scale_dict[_iter][1])
            self.output_layers[f'output_{_iter}'] = nn.Sequential(
                nn.Linear(embed_dims, embed_dims//2),
                nn.LayerNorm(embed_dims//2),
                nn.LeakyReLU(0.1),
                nn.Linear(embed_dims//2, 1)
            )

        self.upsampler = Upsampler()

    def forward(self, mlvl_feats, mlvl_depths, img_metas, target):
        batch_size = mlvl_feats[0].shape[0]
        mlvl_feats = mlvl_feats[::-1]
        assert batch_size == 1
        proj_mats = []
        for img_meta in img_metas:
            proj_mats.append(img_meta['lidar2img'])
        proj_mats = mlvl_feats[0].new_tensor(np.asarray(proj_mats))  # (B, N, 4, 4)

        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        voxel_outputs = {}

        voxel_size = self.voxel_size*(2**self.num_level)
        voxel_inds = self.get_voxel_indices(voxel_size, device=device)
        voxel_dim = voxel_inds[-1][:3] + 1
        voxel_features = torch.empty(
            (len(voxel_inds), 0), dtype=dtype, device=device
        )
        voxel_logits = torch.empty(
            (len(voxel_inds), 0), dtype=dtype, device=device
        )

        proj_mats = proj_mats.squeeze(0)
        for _iter in range(self.num_level):
            bp_uv, bp_depth, bp_mask = self.project_voxels(voxel_inds, proj_mats, voxel_size)
            depth = F.grid_sample(
                mlvl_depths[_iter].squeeze(0),  # nv, 1, h, w
                bp_uv[:, None],  # nv, 1, nq, 2
                padding_mode="zeros",
                mode="nearest",
                align_corners=False
            )[:, 0, 0]  # nv, nq
            bp_prob = self.depth_gaussion(bp_depth, depth, voxel_size)
            bp_feats = self.back_project_features(bp_uv, bp_mask, bp_prob, mlvl_feats[_iter].squeeze(0))
            bp_feats = self.layer_norms[f'norm_{_iter}'](bp_feats)

            voxel_features = self.reductions[f'reduction_{_iter}'](torch.cat((voxel_features, bp_feats, voxel_logits), dim=-1))
            voxel_features = spconv.SparseConvTensor(
                voxel_features, torch.cat([torch.zeros_like(voxel_inds[:, :1]), voxel_inds], dim=1).int(), voxel_dim, batch_size)
            voxel_features = self.net3ds[f'net3d_{_iter}'](voxel_features, voxel_dim.int().tolist())

            voxel_logits = spconv.SparseConvTensor(voxel_features.features,
                voxel_features.indices, voxel_features.spatial_shape, voxel_features.batch_size)
            voxel_logits = voxel_logits.replace_feature(self.output_layers[f'output_{_iter}'](voxel_logits.features))
            voxel_outputs[f'output_{_iter}'] = voxel_logits

            if _iter < self.num_level-1:
                # sparsify & upsample
                occupancy = voxel_logits.features.squeeze(1).sigmoid() > 0.4
                if not torch.sum(occupancy) > 1:
                    return voxel_outputs
                voxel_features = self.upsampler.upsample_feats(
                    voxel_features.features[occupancy]
                )
                voxel_inds = self.upsampler.upsample_inds(voxel_logits.indices[:, 1:][occupancy])
                voxel_logits = self.upsampler.upsample_feats(voxel_logits.features[occupancy])
                voxel_dim = voxel_dim * 2
                voxel_size = voxel_size / 2

        return voxel_outputs

    def depth_gaussion(self, bp_depth, depth, sigma=1):
        return torch.exp(-(bp_depth-depth)**2/(2*sigma)**2)

    def project_voxels(self, voxel_coords, projmat, voxel_size):
        device = projmat.device
        voxel_origin = torch.from_numpy(self.voxel_origin).to(device).unsqueeze(0)
        points = (voxel_coords + 0.5) * voxel_size + voxel_origin
        ones = torch.ones(
            (len(points), 1), device=voxel_coords.device, dtype=torch.float32
        )
        points_h = torch.cat((points, ones), dim=-1) # [nq, 4]

        im_p = projmat.float() @ points_h.t().float()  # [nv, 4, 4] @ [4, nq] -> [nv, 4, nq]
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z
        im_grid = torch.stack(
            [2 * im_x / (self.imwidth - 1) - 1, 2 * im_y / (self.imheight - 1) - 1],
            dim=-1
        )  # [nv, nq, 2], (-1, 1)
        im_grid[torch.isinf(im_grid)] = -2
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        bp_uv = im_grid  # [nv, nq, 2], (-1, 1)
        bp_depth = im_z  # nv, nq
        bp_mask = mask  # nv, nq

        return bp_uv, bp_depth, bp_mask

    def back_project_features(self, bp_uv, bp_mask, bp_prob, feats):
        n_imgs, in_channels, = feats.shape[:2]

        bp_uv = bp_uv.view(n_imgs, 1, -1, 2)
        features = F.grid_sample(
            feats,
            bp_uv.to(feats.dtype),
            padding_mode="reflection",
            align_corners=True,
        )
        features = features.view(n_imgs, in_channels, -1)  # nv, c, nq
        features = features * bp_prob.unsqueeze(1) * 100
        pooled_features = self.mv_fusion_mean(features.transpose(1, 2), bp_mask)

        return pooled_features

    def mv_fusion_mean(self, features, valid_mask):
        weights = torch.sum(valid_mask, dim=0)  # nq
        weights[weights == 0] = 1
        pooled_features = (
            torch.sum(features * valid_mask[..., None], dim=0) / weights[:, None]
        )
        if torch.any(torch.isnan(pooled_features)):
            import IPython; IPython.embed()
        return pooled_features

    def get_voxel_indices(self, voxel_size, device):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
        """
        scene_size = self.scene_size
        vox_origin = self.voxel_origin

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1)], axis=0).astype(int).T

        vox_coords = torch.from_numpy(vox_coords).to(device)

        return vox_coords

    def step(self, out_dict, target, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """
        target = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
        ones = torch.ones_like(target).to(target.device)
        target = torch.where(torch.logical_or(target==255, target==0), target, ones)

        if step_type== "train":     
            voxel_losses = {}
            scale_dict = {0:2**self.num_level, 1:2**(self.num_level-1), 2:2**(self.num_level-2)}
            for _iter in range(self.num_level):
                gt = torch.from_numpy(img_metas[0][f'target_1_{scale_dict[_iter]}']).unsqueeze(0).to(target.device)
                ones = torch.ones_like(gt).to(gt.device)
                gt = torch.where(torch.logical_or(gt==255, gt==0), gt, ones)
                cur_loss = torch.zeros(1, device=gt.device, dtype=torch.float32)
                if f'output_{_iter}' in out_dict.keys():
                    logits = out_dict[f'output_{_iter}']
                    coords = logits.indices.long()
                    gt = gt[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]
                    logits = logits.features.squeeze(1)

                    valid = gt!=255
                    gt = gt[valid]
                    logits = logits[valid]
                
                    if len(logits) > 0:
                        cur_loss = F.binary_cross_entropy_with_logits(logits, gt)
                voxel_losses[f"voxel_loss_{_iter}"] = cur_loss

            return voxel_losses

        elif step_type== "val" or "test":
            y_true = target.cpu().numpy()
            y_pred = out_dict[f'output_{self.num_level-1}'].dense().detach().cpu().numpy()
            y_pred = (y_pred.squeeze(1) > 0).astype(np.int8)

            result = dict()
            result['y_pred'] = y_pred
            result['y_true'] = y_true

            return result

    def training_step(self, out_dict, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, target, img_metas, "val")


class Upsampler(torch.nn.Module):
    # nearest neighbor 2x upsampling for sparse 3D array

    def __init__(self):
        super().__init__()
        self.register_buffer('upsample_offsets',
            torch.Tensor(
                [
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ]
            ).to(torch.int32)
        )
        self.register_buffer('upsample_mul', 
            torch.Tensor([[[2, 2, 2]]]).to(torch.int32)
        )

    def upsample_inds(self, voxel_inds):
        return (
            voxel_inds[:, None] * self.upsample_mul + self.upsample_offsets
        ).reshape(-1, 3)

    def upsample_feats(self, feats):
        return (
            feats[:, None]
            .repeat(1, 8, 1)
            .reshape(-1, feats.shape[-1])
            .to(torch.float32)
        )