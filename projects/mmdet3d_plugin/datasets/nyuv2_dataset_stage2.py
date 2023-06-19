# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.sgn.utils.ssc_metric import SSCMetrics

import pickle

@DATASETS.register_module()
class NYUDatasetStage2(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        img_size=[480, 640],
        labels_tag = 'base',
        color_jitter=None,
        scale=2
    ):
        super().__init__()
        
        self.data_root = os.path.join(data_root, 'depthbin', "NYU" + split)
        self.label_root = os.path.join(preprocess_root, labels_tag, "NYU" + split)
        self.lidar_root = os.path.join(preprocess_root, 'adabins_lidar', "NYU" + split)

        self.n_classes = 12
        self.class_names =  [ "empty", "ceiling", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furniture", "objecs"]
        self.metrics = SSCMetrics(self.n_classes)
        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 2.88, 4.8)  # (4.8m, 2.88m, 4.8m)
        self.scale = scale

        self.img_W = img_size[1]
        self.img_H = img_size[0]

        self.cam_k = np.array([[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]])

        self.scans = glob.glob(os.path.join(self.data_root, "*.bin"))

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.test_mode = test_mode
        self.set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)

    def set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def vox2pix(self, cam_E, cam_k, img_W=None, img_H=None, voxel_size=None, vox_origin=None, scene_size=None):
        """
        compute the 2D projection of voxels centroids
        
        Parameters:
        ----------
        cam_E: 4x4
        =camera pose in case of NYUv2 dataset
        =Transformation from camera to lidar coordinate in case of SemKITTI
        cam_k: 3x3
            camera intrinsics
        vox_origin: (3,)
            lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
        img_W: int
            image width
        img_H: int
            image height
        scene_size: (3,)
            scene size in meter: (51.2, 51.2, 6.4) for SemKITTI
        
        Returns
        -------
        projected_pix: (N, 2)
            Projected 2D positions of voxels
        fov_mask: (N,)
            Voxels mask indice voxels inside image's FOV 
        pix_z: (N,)
            Voxels'distance to the sensor in meter
        """
        img_W, img_H, voxel_size, scene_size = self.img_W, self.img_H, \
            self.scale*self.voxel_size, self.scene_size
        # Compute the x, y, z bounding of the scene in meter
        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels centroids in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        xv, yv, zv = np.meshgrid(
                range(vol_dim[0]),
                range(vol_dim[1]),
                range(vol_dim[2]),
                indexing='ij'
            )
        vox_coords = np.concatenate([
                xv.reshape(1,-1),
                yv.reshape(1,-1),
                zv.reshape(1,-1)
            ], axis=0).astype(int).T

        # Project voxels'centroid from lidar coordinates to camera coordinates
        cam_pts = vox2world(vox_origin, vox_coords, voxel_size)
        cam_pts = rigid_transform(cam_pts, cam_E)

        # Project camera coordinates to pixel positions
        projected_pix = cam2pix(cam_pts, cam_k)
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

        # Eliminate pixels outside view frustum
        pix_z = cam_pts[:, 2]
        fov_mask = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < img_W,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < img_H,
                    pix_z > 0))))

        return projected_pix, fov_mask

    def prepare_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []
        example = self.get_data_info(index)

        data_queue.insert(0, example)

        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'] for each in queue]
        metas_map = {}
        
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        img, meta_dict, target = self.get_input_info(index)

        data_info = dict(
            img_metas = meta_dict,
            img = img,
            target = target
        )
        return data_info

    def get_input_info(self, index):
        file_path = self.scans[index]
        filename = os.path.basename(file_path)
        name = filename[:-4]

        filepath = os.path.join(self.label_root, name + ".pkl")

        with open(filepath, "rb") as handle:
            data = pickle.load(handle)

        pts_filename = os.path.join(
                self.lidar_root, name + ".bin"
            )
        pts = np.fromfile(pts_filename, dtype=np.float32)
        pts = pts.reshape((-1, 4))
        pts = pts[:, :3]

        proposal_path = None
        proposal_bin = self.read_occupancy_NYU(proposal_path) if proposal_path is not None else None

        cam_pose = data["cam_pose"]
        world2cam = inv(cam_pose)
        vox_origin = data["voxel_origin"]

        viewpad = np.eye(4)
        viewpad[:self.cam_k.shape[0], :self.cam_k.shape[1]] = self.cam_k
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        world2img = (viewpad @ world2cam)

        # compute the 3D-2D mapping
        projected_pix, fov_mask = self.vox2pix(world2cam, self.cam_k, vox_origin)

        target = data["target_1_4"]  # Following SSC literature, the output resolution on NYUv2 is set to 1:4
        target_1_2 = data["target_1_8"]

        rgb_path = os.path.join(self.data_root, name + "_color.jpg")
        img = Image.open(rgb_path).convert("RGB")

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        image_tensor = self.normalize_rgb(img)  # (3, img_H, img_W)

        meta_dict = dict(
            cam_intrinsic=[self.cam_k],
            projected_pix=[projected_pix],
            fov_mask=[fov_mask],
            target_1_2=target_1_2,
            img_filename=[file_path],
            lidar2img=[world2img],
            lidar2cam=[world2cam],
            lidar=pts,
            proposal=proposal_bin,
            img_shape=(self.img_H, self.img_W)
        )

        return image_tensor, target, meta_dict

    def read_occupancy_NYU(self, path):
        bin = np.fromfile(path, dtype=np.uint8)
        occupancy = self.unpack(bin)
        
        return occupancy

    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        detail = dict()

        for result in results:
            self.metrics.add_batch(result['y_pred'], result['y_true'])
        metric_prefix = f'{result_name}_SemanticKITTI'

        stats = self.metrics.get_stats()
        for i, class_name in enumerate(self.class_names):
            detail["{}/SemIoU_{}".format(metric_prefix, class_name)] = stats["iou_ssc"][i]

        detail["{}/mIoU".format(metric_prefix)] = stats["iou_ssc_mean"]
        detail["{}/IoU".format(metric_prefix)] = stats["iou"]
        detail["{}/Precision".format(metric_prefix)] = stats["precision"]
        detail["{}/Recall".format(metric_prefix)] = stats["recall"]
        self.metrics.reset()

        return detail


def vox2world(vol_origin, vox_coords, vox_size, offsets=(0.5, 0.5, 0.5)):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        offsets = np.array(offsets).astype(np.float32)
        cam_pts = vol_origin[None] + vox_size*(vox_coords + offsets[None])

        return cam_pts


def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates."""
    intr = intr.astype(np.float32)
    pix = np.dot(intr, cam_pts.T).T
    pix = pix[:, :2] / np.maximum(pix[:, 2:3], np.ones_like(pix[:, 2:3])*1e-6)

    return np.round(pix).astype(np.int64)


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]

