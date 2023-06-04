# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.sgn.modules.former3d.sdfformer import SDFFormer

@DETECTORS.register_module()
class Former3D(MVXTwoStageDetector):
    def __init__(self,
                 attn_heads=None,
                 attn_layers=None,
                 use_proj_occ=None,
                 voxel_size=0.4,
                 voxel_orin=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):

        super(Former3D,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        '''
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        '''

        super().__init__()
        self.sdfformer = SDFFormer(attn_heads=attn_heads, attn_layers=attn_layers, use_proj_occ=use_proj_occ, voxel_size=voxel_size)
        self.voxel_orin = torch.from_numpy(np.array(voxel_orin))

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.foward_training(**kwargs)
        else:
            return self.foward_test(**kwargs)


    # @auto_fp16(apply_to=('img', 'points'))
    def foward_training(self,
                        img_metas=None,
                        img=None,
                        target=None):

        len_queue = img.size(1)
        img_metas = [each[len_queue-1] for each in img_metas] # [dict(), dict(), ...] 
        img = img[:, -1, ...]

        voxel_coords_16 = []
        voxel_gt_coarse = []
        voxel_gt_medium = []
        voxel_gt_fine = []
        lidar2img = []
        gt_depths = []
        for i, img_meta in enumerate(img_metas):
            x = torch.from_numpy(img_meta["voxels_1_8"]).to(img.device)
            voxel_coords_16.append(torch.cat([x, torch.ones([x.shape[0], 1], device=x.device) * i], dim=1).int())
            voxel_gt_coarse.append(torch.from_numpy(img_meta['target_1_8']).to(img.device))
            voxel_gt_medium.append(torch.from_numpy(img_meta['target_1_4']).to(img.device))
            voxel_gt_fine.append(torch.from_numpy(img_meta['target_1_2']).to(img.device))
            lidar2img.append(img_meta['lidar2img'])
            gt_depths.append(img_meta['depth'])
        gt_depths = img.new_tensor(np.asarray(gt_depths)) # B, N, H, W
        voxel_coords_16 = torch.cat(voxel_coords_16, dim=0) # B, voxel, 3
        voxel_gt_coarse = torch.stack(voxel_gt_coarse) # B, H, W, Z
        voxel_gt_medium = torch.stack(voxel_gt_medium)
        voxel_gt_fine = torch.stack(voxel_gt_fine)
        mats = img.new_tensor(np.asarray(lidar2img))  # (B, N, 4, 4)

        origin = self.voxel_orin.to(img.device)
        voxel_outputs, proj_occ_logits, bp_data = self.sdfformer(img, mats, mats[:, :, :3, 3], origin, voxel_coords_16)
        voxel_gt = {
            "coarse": voxel_gt_coarse,
            "medium": voxel_gt_medium,
            "fine": voxel_gt_fine,
        }
        losses = self.sdfformer.losses(
            voxel_outputs, voxel_gt, proj_occ_logits, bp_data, gt_depths
        )

        return losses

    def foward_test(self,
                        img_metas=None,
                        sequence_id=None,
                        img=None,
                        target=None,
                        T_velo_2_cam=None,
                        cam_k=None, **kwargs):

        len_queue = img.size(1)
        img_metas = [each[len_queue-1] for each in img_metas] # [dict(), dict(), ...] 
        img = img[:, -1, ...]

        device = target.device
        target_2 = torch.stack([torch.from_numpy(img_meta['target_1_2']).to(device) for img_meta in img_metas])
        ones = torch.ones_like(target_2).to(device)
        target = torch.where(torch.logical_or(target_2==255, target_2==0), target_2, ones) # [1, 128, 128, 16]

        voxel_coords_16 = []
        lidar2img = []
        for i, img_meta in enumerate(img_metas):
            x = torch.from_numpy(img_meta["voxels_1_8"]).to(device)
            voxel_coords_16.append(torch.cat([x, torch.ones([x.shape[0], 1], device=device) * i], dim=1).int())
            lidar2img.append(img_meta['lidar2img'])
        voxel_coords_16 = torch.cat(voxel_coords_16, dim=0)
        lidar2img = np.asarray(lidar2img)
        mats = img.new_tensor(lidar2img)  # (B, N, 4, 4)

        origin = self.voxel_orin.to(img.device)
        voxel_outputs, _, _ = self.sdfformer(img, mats, mats[:, :, :3, 3], origin, voxel_coords_16)

        logit = voxel_outputs["fine"].dense()
        y_pred = (logit > 0).squeeze(1).long().cpu().numpy()
        
        result = dict()
        y_true = target.cpu().numpy()
        result['y_pred'] = y_pred
        result['y_true'] = y_true
        return result


