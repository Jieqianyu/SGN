import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import open3d as o3d


def project_disp_to_depth(depth, intrinc, cam2world, vox_origin, scene_size, viz=False):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T

    points[:, :2] = points[:, :2] * points[:, 2:3]
    points = np.matmul(np.linalg.inv(intrinc), points.T).T
    
    cloud = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
    cloud = np.matmul(cam2world, cloud.T).T

    mask_x = (cloud[:, 0] >= vox_origin[0]) & (cloud[:, 0] < vox_origin[0]+scene_size[0])
    mask_y = (cloud[:, 1] >= vox_origin[1]) & (cloud[:, 1] < vox_origin[1]+scene_size[1])
    mask_z = (cloud[:, 2] >= vox_origin[2]) & (cloud[:, 2] < vox_origin[2]+scene_size[2])

    valid = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    cloud = cloud[valid]

    if viz:
        pcobj = o3d.geometry.PointCloud()
        pcobj.points = o3d.utility.Vector3dVector(cloud[:, :3])
        pcobj.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([pcobj])

    return cloud


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--base_dir', type=str, default='/home/jm/Downloads/Compressed/nyu/NYU_dataset/base')
    parser.add_argument('--depth_dir', type=str, default='/home/jm/Downloads/Compressed/nyu/NYU_dataset/adabins_depth')
    parser.add_argument('--save_dir', type=str, default='/home/jm/Downloads/Compressed/nyu/NYU_dataset/adabins_lidar')
    parser.add_argument('--viz', action='store_true')
    
    args = parser.parse_args()

    assert os.path.isdir(args.depth_dir)
    assert os.path.isdir(args.base_dir)

    scene_size = (4.8, 2.88, 4.8)
    intrinc = np.array([[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]])

    for split in ["train", "test"]:
        base_dir = os.path.join(args.base_dir, "NYU" + split)
        depth_dir = os.path.join(args.depth_dir, "NYU" + split)
        save_dir = os.path.join(args.save_dir, "NYU" + split)
        os.makedirs(save_dir, exist_ok=True)

        depths = [x for x in os.listdir(depth_dir) if x[-3:] == 'npy' and 'std' not in x]
        depths = sorted(depths)

        for fn in depths:
            prefix = fn[:-4]
            filepath = os.path.join(base_dir, prefix + ".pkl")

            with open(filepath, "rb") as handle:
                data = pickle.load(handle)

            cam2world = data["cam_pose"]
            vox_origin = data["voxel_origin"]

            depth_map = np.load(depth_dir + '/' + fn)

            lidar = project_disp_to_depth(depth_map, intrinc, cam2world, vox_origin, scene_size, args.viz)

            lidar = lidar.astype(np.float32)
            lidar.tofile('{}/{}.bin'.format(save_dir, prefix))
            print('Finish Depth {}'.format(prefix))
