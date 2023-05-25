# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from projects.mmdet3d_plugin.sgn.modules.sdb import SDB
from projects.mmdet3d_plugin.sgn.modules.flosp import FLoSP
from projects.mmdet3d_plugin.sgn.modules.lss_depth import LSSDepth
from projects.mmdet3d_plugin.sgn.utils.ssc_loss import BCE_ssc_loss

@HEADS.register_module()
class SGNHeadOcc(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        embed_dims,
        lss_depth,
        scale_2d_list,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag = False,
        **kwargs
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.n_classes = 20
        self.embed_dims = embed_dims

        self.lss_depth = LSSDepth(**lss_depth)
        self.flosp = FLoSP(scale_2d_list)
        self.bottleneck = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1)

        self.occ_header = nn.Sequential(
            SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=2),
            nn.Conv3d(self.embed_dims//2, 2, kernel_size=3, padding=1)
        )

        self.class_frequencies =  np.array([5.41773033e09, 4.03113667e08])

        self.class_weights = torch.from_numpy(
            1 / np.log(self.class_frequencies + 0.001)
        )

        self.class_names =  [ "empty", "occupied"]

        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag
    
    def gen_depth_prob(self, x, img_metas):
        lidar2cam, intrins = [], []
        for img_meta in img_metas:
            lidar2cam.append(img_meta['lidar2cam'])
            intrins.append(img_meta['cam_intrinsic'])
        lidar2cam = np.asarray(lidar2cam)
        intrins = np.asarray(intrins)
        lidar2cam = x.new_tensor(lidar2cam)  # (B, N, 4, 4)
        intrins = x.new_tensor(intrins)  # (B, N, 3, 3)
        rots, trans = lidar2cam[:, :, :3, :3], lidar2cam[:, :, :3, 3]
        geo_inputs = [rots, trans, intrins]
        
        x, depth = self.lss_depth([x] + geo_inputs)
        
        return x, depth
        
    def forward(self, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """
        x3d = self.flosp(mlvl_feats, img_metas) # bs, c, nq
        bs, c, _ = x3d.shape
        x3d = self.bottleneck(x3d.reshape(bs, c, self.bev_h, self.bev_w, self.bev_z))

        prob_3d, depth = self.gen_depth_prob(mlvl_feats[0], img_metas)
        occ = self.occ_header(x3d*prob_3d)

        out = {}
        out["occ"] = occ
        out["depth"] = depth

        return out 

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
            loss_dict = dict()

            gt_depths = []
            for img_meta in img_metas:
                gt_depths.append(img_meta['depth'])
            gt_depths = np.asarray(gt_depths)
            gt_depths = out_dict["depth"].new_tensor(gt_depths)
            loss_depth = self.lss_depth.get_bce_depth_loss(gt_depths, out_dict["depth"])
            loss_dict['loss_depth'] = loss_depth

            class_weights = self.class_weights.type_as(target)
            loss_occ = BCE_ssc_loss(out_dict['occ'], target, class_weights, 0.5)
            loss_dict['loss_occ'] = loss_occ

            return loss_dict

        elif step_type== "val" or "test":
            y_true = target.cpu().numpy()
            y_pred = out_dict['occ'].detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            result = dict()
            result['y_pred'] = y_pred
            result['y_true'] = y_true

            if self.save_flag:
                self.save_pred(img_metas, y_pred)

            return result

    def training_step(self, out_dict, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, target, img_metas, "val")

    def get_voxel_indices(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        return vox_coords

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        """

        y_pred[y_pred==10] = 44
        y_pred[y_pred==11] = 48
        y_pred[y_pred==12] = 49
        y_pred[y_pred==13] = 50
        y_pred[y_pred==14] = 51
        y_pred[y_pred==15] = 70
        y_pred[y_pred==16] = 71
        y_pred[y_pred==17] = 72
        y_pred[y_pred==18] = 80
        y_pred[y_pred==19] = 81
        y_pred[y_pred==1] = 10
        y_pred[y_pred==2] = 11
        y_pred[y_pred==3] = 15
        y_pred[y_pred==4] = 18
        y_pred[y_pred==5] = 20
        y_pred[y_pred==6] = 30
        y_pred[y_pred==7] = 31
        y_pred[y_pred==8] = 32
        y_pred[y_pred==9] = 40

        # save predictions
        pred_folder = os.path.join("./sgn", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))
