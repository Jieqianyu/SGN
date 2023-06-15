# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
from PIL import Image
import glob
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.sgn.utils.ssc_metric import SSCMetrics
import skimage
import skimage.io
import cv2

@DATASETS.register_module()
class SemanticKittiDataset(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        img_size=[370, 1220],
        temporal = [],
        eval_range = 51.2,
        labels_tag = 'labels',
        color_jitter=None,
        use_strong_img_aug=False,
        do_flip=False,
        scale=2
    ):
        super().__init__()
        
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, labels_tag)
        self.eval_range = eval_range
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split 
        self.sequences = splits[split]
        self.n_classes = 20
        self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", 
                            "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", 
                            "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        self.metrics = SSCMetrics(self.n_classes)
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m
        self.scale = scale

        self.img_W = img_size[1]
        self.img_H = img_size[0]

        self.poses=self.load_poses()
        self.target_frames = temporal
        self.load_scans()
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.do_flip = do_flip
        # strong_img_aug
        self.do_strong_img_aug = use_strong_img_aug
        print("INFO: Use strong_img_aug: {}".format(self.do_strong_img_aug))
        if self.do_strong_img_aug:
            self.strong_img_aug = transforms.Compose(
                [
                    transforms.RandomGrayscale(p=0.1),
                ]
            )
            self.blur_img_aug = transforms.Compose(
                [
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                ]
            )

        self.test_mode = test_mode
        self.set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )

            for path in glob.glob(glob_path):

                self.scans.append(
                    {
                        "sequence": sequence,
                        "pose": self.poses[sequence],
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": path
                    }
                )

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
        img_W, img_H, voxel_size, vox_origin, scene_size = self.img_W, self.img_H, \
            self.scale*self.voxel_size, self.vox_origin, self.scene_size
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
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        scan = self.scans[index]

        path = scan["voxel_path"]

        sequence = scan["sequence"]
        filename = os.path.basename(path)
        frame_id = os.path.splitext(filename)[0]

        img, post_mat_list = self.get_input_info(sequence, frame_id)
        meta_dict = self.get_meta_info(scan, sequence, post_mat_list, frame_id)
        target = self.get_gt_info(sequence, frame_id)

        data_info = dict(
            img_metas = meta_dict,
            img = img,
            target = target
        )
        return data_info

    def get_meta_info(self, scan, sequence, post_mat_list, frame_id):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id,
            proposal_path (str): proposal path.

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )

        # for multiple images
        lidar2img_rts = []
        image_paths = []
        image_depths = []

        projected_pixs = []
        fov_masks = []

        # transform points from lidar to camera coordinate
        lidar2cam_rt = scan["T_velo_2_cam"]
        # camera intrisic
        P = scan["P"]
        cam_k = P[0:3, 0:3]
        post_mat = post_mat_list.pop(0)
        intrinsic = post_mat @ cam_k
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        lidar2img_rt = (viewpad @ lidar2cam_rt)

        pix, mask = self.vox2pix(lidar2cam_rt, intrinsic)
        projected_pixs.append(pix)
        fov_masks.append(mask)

        if self.split == "train":
            depth_path = os.path.join(self.data_root, "dataset", "sequences_msnet3d_depth", 'sequences', sequence, frame_id + ".npy")
            depth = np.load(depth_path).astype("float32")
            depth = cv2.resize(depth, (self.img_W, self.img_H), interpolation=cv2.INTER_NEAREST)
            if post_mat[0, 0] < 0:
                depth = np.ascontiguousarray(np.fliplr(depth))
        else:
            depth = None

        lidar2img_rts.append(lidar2img_rt)
        image_paths.append(rgb_path)
        image_depths.append(depth)

        # for reference img
        seq_len = len(self.poses[sequence])
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )

            pose_list = self.poses[sequence]

            ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
            target = pose_list[int(target_id)]
            ref2target = np.matmul(inv(target), ref) # both for lidar

            target2cam = scan["T_velo_2_cam"] # lidar to camera
            ref2cam = target2cam @ ref2target

            post_mat = post_mat_list.pop(0)
            intrinsic = post_mat @ cam_k
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2cam_rt  = ref2cam
            lidar2img_rt = (viewpad @ lidar2cam_rt)

            pix, mask = self.vox2pix(lidar2cam_rt, intrinsic)
            projected_pixs.append(pix)
            fov_masks.append(mask)

            if self.split == "train":
                depth_path = os.path.join(self.data_root, "dataset", "sequences_msnet3d_depth", 'sequences', sequence, target_id + ".npy")
                depth = np.load(depth_path).astype("float32")
                depth = cv2.resize(depth, (self.img_W, self.img_H), interpolation=cv2.INTER_NEAREST)
                if post_mat[0, 0] < 0:
                    depth = np.ascontiguousarray(np.fliplr(depth))
            else:
                depth = None

            lidar2img_rts.append(lidar2img_rt)
            image_paths.append(rgb_path)
            image_depths.append(depth)

        if self.split == 'train' or self.split == 'val':
            target_1_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
            target_1_1 = np.load(target_1_1_path)
            target_1_1 = target_1_1.reshape(-1)
            target_1_1 = target_1_1.reshape(256, 256, 32)
            target_1_1 = target_1_1.astype(np.float32)

            target_1_2_path = os.path.join(self.label_root, sequence, frame_id + "_1_2.npy")
            target_1_2 = np.load(target_1_2_path)
            target_1_2 = target_1_2.reshape(-1)
            target_1_2 = target_1_2.reshape(128, 128, 16)
            target_1_2 = target_1_2.astype(np.float32)

            target_1_4_path = os.path.join(self.label_root, sequence, frame_id + "_1_4.npy")
            target_1_4 = np.load(target_1_4_path)
            target_1_4 = target_1_4.reshape(-1)
            target_1_4 = target_1_4.reshape(64, 64, 8)
            target_1_4 = target_1_4.astype(np.float32)

            target_1_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
            target_1_8 = np.load(target_1_8_path)
            target_1_8 = target_1_8.reshape(-1)
            target_1_8 = target_1_8.reshape(32, 32, 4)
            target_1_8 = target_1_8.astype(np.float32)
        else:
            target_1_1 = None
            target_1_2 = None
            target_1_4 = None
            target_1_8 = None

        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            target_1_1=target_1_1,
            target_1_2=target_1_2,
            target_1_4=target_1_4,
            target_1_8=target_1_8,
            depth=image_depths,
            projected_pix=projected_pixs,
            fov_mask=fov_masks, 
            img_filename=image_paths,
            lidar2img = lidar2img_rts,
            img_shape = (self.img_H, self.img_W)
        )

        return meta_dict

    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        seq_len = len(self.poses[sequence])
        image_list = []
        post_mat_list = []

        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )
        img = Image.open(rgb_path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        if self.do_strong_img_aug:
            if np.random.rand() < 0.3:
                img = self.blur_img_aug(img)
            if np.random.rand() < 0.3:
                img = self.strong_img_aug(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        H, W = img.shape[:2]
        rh, rw = float(self.img_H)/float(H), float(self.img_W)/float(W)
        img = cv2.resize(img, (self.img_W, self.img_H))

        post_mat = np.eye(3)
        post_mat[0] *= rw
        post_mat[1] *= rh

        if self.do_flip:
            if np.random.rand() < 0.3:
                img = np.ascontiguousarray(np.fliplr(img))
                post_mat[0, 0] *= -1
                post_mat[0, 2] = self.img_W

        image_list.append(torch.from_numpy(img).permute(2, 0, 1))
        post_mat_list.append(post_mat)

        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            if self.do_strong_img_aug:
                if np.random.rand() < 0.3:
                    img = self.blur_img_aug(img)
                if np.random.rand() < 0.3:
                    img = self.strong_img_aug(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            H, W = img.shape[:2]
            rh, rw = float(self.img_H)/float(H), float(self.img_W)/float(W)
            img = cv2.resize(img, (self.img_W, self.img_H))

            post_mat = np.eye(3)
            post_mat[0] *= rw
            post_mat[1] *= rh

            if self.do_flip:
                if np.random.rand() < 0.3:
                    img = np.ascontiguousarray(np.fliplr(img))
                    post_mat[0, 0] *= -1
                    post_mat[0, 2] = self.img_W

            image_list.append(torch.from_numpy(img).permute(2, 0, 1))
            post_mat_list.append(post_mat)

        image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]

        return image_tensor, post_mat_list

    def get_gt_info(self, sequence, frame_id):
        """Get the ground truth.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            array: target. 
        """
        if self.split == "train" or self.split == "val":
            # load full-range groundtruth
            target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
            target = np.load(target_1_path)
            # short-range groundtruth
            if self.eval_range == 25.6:
                target[128:, :, :] = 255
                target[:, :64, :] = 255
                target[:, 192:, :] = 255

            elif self.eval_range == 12.8:
                target[64:, :, :] = 255
                target[:, :96, :] = 255
                target[:, 160:, :] = 255
        else:
            target = np.ones((256,256,32))

        return target

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

