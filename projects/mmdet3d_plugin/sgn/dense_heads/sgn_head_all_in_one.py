import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, builder
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from projects.mmdet3d_plugin.sgn.utils.header import Header, SparseHeader
from projects.mmdet3d_plugin.sgn.modules.sgb import SGB
from projects.mmdet3d_plugin.sgn.modules.sdb import SDB
from projects.mmdet3d_plugin.sgn.modules.flosp import FLoSP
from projects.mmdet3d_plugin.sgn.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.sgn.utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss


@HEADS.register_module()
class SGNHeadAll(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        embed_dims,
        view_transform_dict,
        pts_header_dict,
        sdb_depth=3,
        geo_prior=True,
        pad_zero=False,
        geo_guidance=True,
        sem_guidance=True,
        one_stage = True,
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
        self.real_h = 51.2
        self.real_w = 51.2
        self.n_classes = 20
        self.embed_dims = embed_dims

        self.geo_prior = geo_prior
        self.pad_zero = pad_zero
        self.geo_guidance = geo_guidance
        self.sem_guidance = sem_guidance
        self.one_stage = one_stage
        self.sdb_depth = sdb_depth

        self.view_transform_dict = view_transform_dict
        if self.view_transform_dict['type'] == 'flosp':
            self.flosp = FLoSP(view_transform_dict['scale_2d_list'])
        elif self.view_transform_dict['type'] == 'lss':
            self.lss = builder.build_neck(view_transform_dict['lss_neck'])
        elif self.view_transform_dict['type'] == 'cross-attn':
            self.bev_embed = nn.Embedding((self.bev_h) * (self.bev_w) * (self.bev_z), self.embed_dims)
            self.positional_encoding = build_positional_encoding(view_transform_dict['positional_encoding'])
            self.cross_transformer = build_transformer(view_transform_dict['cross_transformer'])
            if not self.geo_prior and not self.pad_zero:
                self.mask_embed = nn.Embedding(1, self.embed_dims)
        else:
            raise NotImplementedError

        self.bottleneck = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1)

        if self.sem_guidance:
            self.sgb = SGB(sizes=[self.bev_h, self.bev_w, self.bev_z], channels=self.embed_dims)
            self.sem_header = SparseHeader(self.n_classes, feature=self.embed_dims)

        if self.geo_prior:
            self.mlp_prior = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dims//2, self.embed_dims)
            )

        occ_channel = 8 if self.one_stage and pts_header_dict.get('guidance', False) else 0
        if sdb_depth > 0:
            self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=self.embed_dims//2, depth=sdb_depth)
        
        if self.geo_guidance:
            self.occ_header = nn.Sequential(
                SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=1),
                nn.Conv3d(self.embed_dims//2, 1, kernel_size=3, padding=1)
            )
        
        self.ssc_header = Header(self.n_classes, feature=self.embed_dims//2)

        if self.one_stage:
            self.pts_header = builder.build_head(pts_header_dict)

        self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 
                                                        0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag

    def view_transform(self, mlvl_feats, img_metas):
        if self.view_transform_dict['type'] == 'flosp':
            x3d = self.flosp(mlvl_feats, img_metas) # bs, c, nq
            return x3d, None
        elif self.view_transform_dict['type'] == 'lss':
            x = mlvl_feats[-1]
            lidar2cam, intrins = [], []
            for img_meta in img_metas:
                lidar2cam.append(img_meta['lidar2cam'])
                intrins.append(img_meta['cam_intrinsic'])
            lidar2cam = np.asarray(lidar2cam)
            intrins = np.asarray(intrins)
            lidar2cam = x.new_tensor(lidar2cam)  # (B, N, 4, 4)
            intrins = x.new_tensor(intrins)  # (B, N, 3, 3)
            cam2lidar = torch.inverse(lidar2cam)
            rots, trans = cam2lidar[:, :, :3, :3], cam2lidar[:, :, :3, 3]
            B, N = rots.shape[:2]
            post_rots, post_trans, bda = x.new_tensor(np.eye(3)).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1), \
                torch.zeros_like(trans), x.new_tensor(np.eye(4)).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            mlp_input = self.lss.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
            geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
            
            x, depth = self.lss([x] + geo_inputs)
            x3d = x.flatten(-3)
            return x3d, depth
        elif self.view_transform_dict['type'] == 'cross-attn':
            bs = mlvl_feats[0].shape[0]
            dtype = mlvl_feats[0].dtype

            bev_queries = self.bev_embed.weight.to(dtype) #[128*128*16, dim]
            ref_3d = self.get_ref_3d()
            bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
            x3d = self.cross_transformer(
                mlvl_feats, 
                bev_queries,
                self.bev_h,
                self.bev_w,
                ref_3d=ref_3d,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos_cross_attn,
                img_metas=img_metas,
                prev_bev=None,
            )
            x3d = x3d.permute(0, 2, 1)
            return x3d, None
        else:
            raise NotImplementedError
        
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
        x3d, depth = self.view_transform(mlvl_feats, img_metas) # bs, c, nq
        out["depth"] = depth
        bs, c, _ = x3d.shape
        x3d = self.bottleneck(x3d.reshape(bs, c, self.bev_h, self.bev_w, self.bev_z))
        if self.geo_guidance:
            occ = self.occ_header(x3d).squeeze(1)
            out["occ"] = occ

        x3d = x3d.reshape(bs, c, -1)
        # Load proposals
        if self.one_stage:
            pts_out = self.pts_header(mlvl_feats, img_metas, target)
            pts_occ = pts_out['occ_logit'].squeeze(1)
            proposal =  (pts_occ > 0).float().detach().cpu().numpy()
            out['pts_occ'] = pts_occ
        else:
            proposal =  img_metas[0]['proposal'].reshape(self.bev_h, self.bev_w, self.bev_z)

        unmasked_idx = np.asarray(np.where(proposal.reshape(-1)>0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1)==0)).astype(np.int32)
        vox_coords = self.get_voxel_indices()

        # Compute seed features
        seed_feats = x3d[0, :, vox_coords[unmasked_idx[0], 3]].permute(1, 0)
        if self.sem_guidance:
            seed_coords = vox_coords[unmasked_idx[0], :3]
            coords_torch = torch.from_numpy(np.concatenate(
                [np.zeros_like(seed_coords[:, :1]), seed_coords], axis=1)).to(seed_feats.device)
            seed_feats = self.sgb(seed_feats, coords_torch)
            sem = self.sem_header(seed_feats)
            out["sem_logit"] = sem
            out["coords"] = seed_coords

        # Complete voxel features
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=x3d.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats

        if self.geo_prior:
            vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mlp_prior(x3d[0, :, vox_coords[masked_idx[0], 3]].permute(1, 0))
        else:
            if self.pad_zero:
                vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = torch.zeros(masked_idx.shape[1], self.embed_dims).to(mlvl_feats[0].dtype)
            else:
                vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(
                    masked_idx.shape[1], self.embed_dims).to(mlvl_feats[0].dtype)

        vox_feats_diff = vox_feats_flatten.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0)
        if self.pts_header.guidance:
            vox_feats_diff = torch.cat([vox_feats_diff, pts_out['occ_x']], dim=1)
        if self.sdb_depth > 0:
            vox_feats_diff = self.sdb(vox_feats_diff) # 1, C,H,W,Z
        ssc_dict = self.ssc_header(vox_feats_diff)

        out.update(ssc_dict)
        
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

        ssc_pred = out_dict["ssc_logit"]

        if step_type== "train":
            target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)

            if self.sem_guidance:
                sem_pred_2 = out_dict["sem_logit"]
                coords = out_dict['coords']
                sp_target_2 = target_2.clone()[0, coords[:, 0], coords[:, 1], coords[:, 2]]

            loss_dict = dict()
            class_weight = self.class_weights.type_as(target)
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                loss_dict['loss_ssc'] = loss_ssc

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal

            if self.sem_guidance:
                loss_sem = lovasz_softmax(F.softmax(sem_pred_2, dim=1), sp_target_2, ignore=255)
                loss_sem += F.cross_entropy(sem_pred_2, sp_target_2.long(), ignore_index=255)
                loss_dict['loss_sem'] = loss_sem

            if self.geo_guidance:
                ones = torch.ones_like(target_2).to(target_2.device)
                target_2_binary = torch.where(torch.logical_or(target_2==255, target_2==0), target_2, ones)
                loss_occ = F.binary_cross_entropy(out_dict['occ'].sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())
                loss_dict['loss_occ'] = loss_occ

            if self.one_stage:
                loss_dict['loss_pts'] = F.binary_cross_entropy(out_dict['pts_occ'].sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())

            if self.view_transform_dict['type'] == 'lss':
                depths = []
                for img_meta in img_metas:
                    depths.append(img_meta['depth'])
                depths = np.asarray(depths)
                depths = out_dict["depth"].new_tensor(depths)
                loss_depth = self.lss.get_depth_loss(depths, out_dict["depth"])
                loss_dict['loss_depth'] = loss_depth

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

    def get_voxel_indices(self):
        """Get reference points in 3D.
        Returns:
            vox_coords (Array): Voxel indices
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = 51.2 / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        return vox_coords

    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = 51.2 / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1,-1)+0.5)/self.bev_h, (yv.reshape(1,-1)+0.5)/self.bev_w, (zv.reshape(1,-1)+0.5)/self.bev_z,], axis=0).astype(np.float64).T 

        return ref_3d

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
