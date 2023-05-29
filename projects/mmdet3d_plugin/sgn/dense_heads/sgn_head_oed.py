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
import spconv.pytorch as spconv
from mmdet.models import HEADS
from projects.mmdet3d_plugin.sgn.utils.header import Header, SparseHeader
from projects.mmdet3d_plugin.sgn.modules.sgb import SGB
from projects.mmdet3d_plugin.sgn.modules.sdb import SDB
from projects.mmdet3d_plugin.sgn.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.sgn.utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss

@HEADS.register_module()
class SGNHeadOED(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        embed_dims,
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
        self.n_classes = 20
        self.embed_dims = embed_dims

        self.bp_mlp = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.LeakyReLU()
        )

        self.guidance = nn.Sequential(
            SDB(channel=self.embed_dims, out_channel=self.embed_dims//4, depth=1),
            nn.Conv3d(self.embed_dims//4, 24, kernel_size=1)
        )
        self.cspn = Affinity_Propagate(prop_time=1)

        self.sgb = SGB(sizes=[self.bev_h, self.bev_w, self.bev_z], channels=self.embed_dims)
        self.sdb = SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=2)

        self.seed_header = SparseHeader(self.n_classes, feature=self.embed_dims)
        self.ssc_header = Header(self.n_classes, feature=self.embed_dims//2)

        self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 
                                                        0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag
        
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
        out = {}
        # camera parameters
        device = mlvl_feats[0].device
        lidar2img = img_metas[0]['lidar2img']
        lidar2img = torch.from_numpy(np.asarray(lidar2img)).float().to(device)  # (N, 4, 4)

        # Load proposals and compute seed features
        proposal =  img_metas[0]['proposal'].reshape(self.bev_h, self.bev_w, self.bev_z)
        seed_coords = torch.from_numpy(np.stack(np.where(proposal>0), -1).astype(np.int32)).to(device)
        seed_feats = self.img_feats_sampling(seed_coords, lidar2img, mlvl_feats[0].squeeze(0), voxel_size=0.4)
        seed_feats_desc = self.sgb(seed_feats, torch.cat([torch.zeros_like(seed_coords[:, :1]), seed_coords], dim=1))
        seed_logit = self.seed_header(seed_feats_desc)

        # Complete voxel features
        voxel_feats = spconv.SparseConvTensor(
            seed_feats_desc, torch.cat([torch.zeros_like(seed_coords[:, :1]), seed_coords.int()], dim=1), 
            np.array([self.bev_h, self.bev_w, self.bev_z], np.int32), 1
        ).dense()
        voxel_feats_diff = self.sdb(voxel_feats) # 1, C, H, W, Z
        coarse_logit = self.ssc_header(voxel_feats_diff)["ssc_logit"]
        
        voxel_logit = spconv.SparseConvTensor(
            seed_logit, torch.cat([torch.zeros_like(seed_coords[:, :1]), seed_coords.int()], dim=1), 
            np.array([self.bev_h, self.bev_w, self.bev_z], np.int32), 1
        ).dense().squeeze(0)  # 20, h, w, z
        guidance = self.guidance(voxel_feats).squeeze(0)
        refine_logit = F.interpolate(self.cspn(guidance, voxel_logit).unsqueeze(0), scale_factor=2, mode='trilinear', align_corners=True)

        out['coarse_logit'] = coarse_logit
        out["ssc_logit"] = coarse_logit + refine_logit
        out["sem_logit"] = seed_logit
        out["coords"] = seed_coords

        return out 

    def img_feats_sampling(self, voxel_coords, projmat, feats, voxel_size):
        imheight, imwidth = feats.shape[-2:]
        bp_uv, _, bp_mask = self.project_voxels(voxel_coords, projmat, imheight, imwidth, voxel_size=voxel_size)
        sampled_feats = self.back_project_features(bp_uv, bp_mask, feats)

        return sampled_feats

    def project_voxels(
        self, voxel_coords, projmat, imheight, imwidth, voxel_origin=[0, -25.6, -2], voxel_size=0.2
    ):
        voxel_origin = torch.tensor(voxel_origin).to(voxel_coords.device)
        points = (voxel_coords + 0.5) * voxel_size + voxel_origin.unsqueeze(0)
        ones = torch.ones(
            (len(points), 1), device=voxel_coords.device, dtype=torch.float32
        )
        points_h = torch.cat((points, ones), dim=-1) # [nq, 4]

        im_p = projmat @ points_h.t()  # [nv, 4, 4] @ [4, nq] -> [nv, 4, nq]
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z
        im_grid = torch.stack(
            [2 * im_x / (imwidth - 1) - 1, 2 * im_y / (imheight - 1) - 1],
            dim=-1
        )  # [nv, nq, 2], (-1, 1)
        im_grid[torch.isinf(im_grid)] = -2
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        bp_uv = im_grid  # [nv, nq, 2], (-1, 1)
        bp_depth = im_z  # nv, nq
        bp_mask = mask  # nv, nq

        return bp_uv, bp_depth, bp_mask

    def back_project_features(self, bp_uv, bp_mask, feats):
        n_imgs, in_channels, = feats.shape[:2]

        bp_uv = bp_uv.view(n_imgs, 1, -1, 2)
        features = F.grid_sample(
            feats,
            bp_uv.to(feats.dtype),
            padding_mode="reflection",
            align_corners=True,
        )
        features = features.view(n_imgs, in_channels, -1)  # nv, c, nq
        var_imgs = ((features-features.mean(dim=0))**2)
        var = var_imgs.mean(0)
        pooled_features = self.mv_fusion_mean(features.transpose(1, 2), bp_mask)
        feature_volume = torch.cat([pooled_features, var.transpose(0, 1)], dim=1)

        return self.bp_mlp(feature_volume)

    def mv_fusion_mean(self, features, valid_mask):
        weights = torch.sum(valid_mask, dim=0)  # nq
        weights[weights == 0] = 1
        pooled_features = (
            torch.sum(features * valid_mask[..., None], dim=0) / weights[:, None]
        )
        if torch.any(torch.isnan(pooled_features)):
            import IPython; IPython.embed()
        return pooled_features

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

        ssc_pred = out_dict["ssc_logit"]

        if step_type== "train":
            sem_pred_2 = out_dict["sem_logit"]

            target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
            coords = out_dict['coords'].long()
            sp_target_2 = target_2.clone()[0, coords[:, 0], coords[:, 1], coords[:, 2]]
            loss_dict = dict()

            class_weight = self.class_weights.type_as(target)
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                loss_dict['loss_ssc'] = loss_ssc
                loss_dict['loss_ssc_cr'] = CE_ssc_loss(out_dict['coarse_logit'], target, class_weight)

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal
                loss_dict['loss_sem_scal_cr'] = sem_scal_loss(out_dict['coarse_logit'], target)

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal
                loss_dict['loss_geo_scal_cr'] = geo_scal_loss(out_dict['coarse_logit'], target)

            loss_sem = lovasz_softmax(F.softmax(sem_pred_2, dim=1), sp_target_2, ignore=255)
            loss_sem += F.cross_entropy(sem_pred_2, sp_target_2.long(), ignore_index=255)
            loss_dict['loss_sem'] = loss_sem

            return loss_dict

        elif step_type== "val" or "test":
            y_true = target.cpu().numpy()
            y_pred = ssc_pred.detach().cpu().numpy()
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


class Affinity_Propagate(nn.Module):

    def __init__(self,
                 prop_time,
                 norm_type='8sum'):
        """
        Inputs:
            prop_time: how many steps for CSPN to perform
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super(Affinity_Propagate, self).__init__()
        self.prop_time = prop_time
        self.norm_type = norm_type
        assert norm_type in ['8sum', '8sum_abs']

        self.in_feature = 1
        self.out_feature = 1

    def gen_operator(self):
        self.sum_conv = nn.Conv3d(in_channels=8,
                                  out_channels=1,
                                  kernel_size=(1, 1, 1),
                                  stride=1,
                                  padding=0,
                                  bias=False)
        weight = torch.ones(1, 8, 1, 1, 1).cuda()
        self.sum_conv.weight = nn.Parameter(weight)
        for param in self.sum_conv.parameters():
            param.requires_grad = False

    def propagate_once(self, gate_wb, gate_sum, blur):
        result = self.pad_blur(blur)
        neigbor_weighted_sum = self.sum_conv(gate_wb * result)
        neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
        neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
        result = neigbor_weighted_sum

        if '8sum' in self.norm_type:
                result = (1.0 - gate_sum) * blur + result
        else:
            raise ValueError('unknown norm %s' % self.norm_type)

        return result

    def forward(self, guidance, blur):
        self.gen_operator()
        
        guidance_xyz = torch.split(guidance, guidance.shape[0]//3, dim=0)

        gate_wb_x, gate_sum_x = self.affinity_normalization(guidance_xyz[0].permute(3, 0, 1, 2))
        gate_wb_y, gate_sum_y = self.affinity_normalization(guidance_xyz[1].permute(2, 0, 1, 3))
        gate_wb_z, gate_sum_z = self.affinity_normalization(guidance_xyz[2].permute(1, 0, 2, 3))

        result = blur  # 20, h, w, z
        for _ in range(self.prop_time):
            # one propagation
            result = self.propagate_once(gate_wb_x, gate_sum_x, result.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
            result = self.propagate_once(gate_wb_y, gate_sum_y, result.permute(2, 0, 1, 3)).permute(1, 2, 0, 3)
            result = self.propagate_once(gate_wb_z, gate_sum_z, result.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        return result

    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)

        gate1_wb_cmb = guidance.narrow(1, 0                   , self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                             gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)

        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = self.sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum


    def pad_blur(self, blur):
        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        blur_1 = left_top_pad(blur).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        blur_2 = center_top_pad(blur).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        blur_3 = right_top_pad(blur).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        blur_4 = left_center_pad(blur).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        blur_5 = right_center_pad(blur).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        blur_6 = left_bottom_pad(blur).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        blur_7 = center_bottom_pad(blur).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        blur_8 = right_bottm_pad(blur).unsqueeze(1)

        result_depth = torch.cat((blur_1, blur_2, blur_3, blur_4,
                                  blur_5, blur_6, blur_7, blur_8), 1)
        return result_depth
