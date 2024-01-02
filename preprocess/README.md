# SemanticKITTI
## 1. Prepare data
Symlink the dataset root to ./kitti.
```
ln -s [SemanticKITTI root] ./kitti
```
The data is organized in the following format:

```
./kitti/dataset/
          └── sequences/
                  ├── 00/
                  │   ├── poses.txt
                  │   ├── calib.txt
                  │   ├── image_2/
                  │   ├── image_3/
                  |   ├── voxels/
                  |         ├ 000000.bin
                  |         ├ 000000.label
                  |         ├ 000000.occluded
                  |         ├ 000000.invalid
                  |         ├ 000005.bin
                  |         ├ 000005.label
                  |         ├ 000005.occluded
                  |         ├ 000005.invalid
                  ├── 01/
                  ├── 02/
                  .
                  └── 21/

```
## 2. Generating grounding truth
Setting up the environment
```shell
conda create -n preprocess python=3.7 -y
conda activate preprocess
conda install numpy tqdm pyyaml imageio
```
Preprocess the data to generate labels at a lower scale:
```
python label/label_preprocess.py --kitti_root=[SemanticKITTI root] --kitti_preprocess_root=[preprocess_root]
```

Then we have the following data:
```
./kitti/dataset/
          └── sequences/
          │       ├── 00/
          │       │   ├── poses.txt
          │       │   ├── calib.txt
          │       │   ├── image_2/
          │       │   ├── image_3/
          │       |   ├── voxels/
          │       ├── 01/
          │       ├── 02/
          │       .
          │       └── 21/
          └── labels/
                  ├── 00/
                  │   ├── 000000_1_1.npy
                  │   ├── 000000_1_2.npy
                  │   ├── 000005_1_1.npy
                  │   ├── 000005_1_2.npy
                  ├── 01/
                  .
                  └── 10/

```

## 3. Image to depth
### Disparity estimation
We use [MobileStereoNet3d](https://github.com/cogsys-tuebingen/mobilestereonet) to obtain the disparity. We add several lines to convert disparity into depth, and add filenames to support kitti odometry dataset. We upload our folder for your convenience. Please refer to the [original repository](https://github.com/cogsys-tuebingen/mobilestereonet) for detailed instructions.

### Requirements
The code is tested on:
- Ubuntu 18.04
- Python 3.6 
- PyTorch 1.4.0 
- Torchvision 0.5.0
- CUDA 10.0

### Setting up the environment

```shell
cd mobilestereonet
conda env create --file mobilestereonet.yaml # please modify prefix in .yaml
conda activate mobilestereonet
```

### Prediction

The following script could create depth maps for all sequences:
```shell
./image2depth.sh
```
## 4. Depth to pseudo point cloud
The following script could create pseudo point cloud for all sequences:

```shell
./depth2lidar.sh
```

Finally we have the following data:
```
./kitti/dataset/
          └── sequences/
          │       ├── 00/
          │       │   ├── poses.txt
          │       │   ├── calib.txt
          │       │   ├── image_2/
          │       │   ├── image_3/
          │       |   ├── voxels/
          │       ├── 01/
          │       ├── 02/
          │       .
          │       └── 21/
          └── labels/
          │       ├── 00/
          │       │   ├── 000000_1_1.npy
          │       │   ├── 000000_1_2.npy
          │       │   ├── 000005_1_1.npy
          │       │   ├── 000005_1_2.npy
          │       ├── 01/
          │       .
          │       └── 10/
          └── sequences_msnet3d_lidar/
                  └── sequences
                        ├── 00
                        │   ├ 000001.bin
                        │   ├ 000002.bin
                        ├── 01/
                        ├── 02/
                        .
                        └── 21/
```

# SSCBench-KITTI-360
## 1. Prepare data
Refer to [SSCBench](https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360) to download the dataset. And download the [poses](https://drive.google.com/file/d/1nsZLa-X3fz14ZZxZgPUOCm3dY5MDZ5vZ/view?usp=drive_link) that we have processed acoording the matching index between SSCBench-KITTI-360 and KITTI-360. Symlink the dataset root to ./kitti360.
```
ln -s [SSCBench-KITTI-360 root] ./kitti360
```
The data is organized in the following format:

```
./kitti360/
        └── data_2d_raw/
        │        ├── 2013_05_28_drive_0000_sync/ 
        │        │   ├── image_00/
        │        │   │   ├── data_rect/
        │        │   │   │	├── 000000.png
        │        │   │   │	├── 000001.png
        │        │   │   │	├── ...
        │        │   ├── image_01/
        │        │   │   ├── data_rect/
        │        │   │   │	├── 000000.png
        │        │   │   │	├── 000001.png
        │        │   │   │	├── ...
        │        │   ├── voxels/
        │        │   └── poses.txt
        │        ├── 2013_05_28_drive_0002_sync/
        │        ├── 2013_05_28_drive_0003_sync/
        │        .
        │        └── 2013_05_28_drive_0010_sync/
        └── preprocess/
                 ├── labels/ 
                 │   ├── 2013_05_28_drive_0000_sync/
                 │   │   ├── 000000_1_1.npy
		 │   │   ├── 000000_1_2.npy
		 │   │   ├── 000000_1_8.npy
		 │   │   ├── ...
		 │   ├── 2013_05_28_drive_0002_sync
		 │   ├── 2013_05_28_drive_0003_sync/
		 │   .
		 │   └── 2013_05_28_drive_0010_sync/
		 ├── labels_half/ 
		 └── unified/ 

```

## 2. Image to depth
### Disparity estimation
We use [MobileStereoNet3d](https://github.com/cogsys-tuebingen/mobilestereonet) to obtain the disparity. We add several lines to convert disparity into depth, and add filenames to support kitti odometry dataset. We upload our folder for your convenience. Please refer to the [original repository](https://github.com/cogsys-tuebingen/mobilestereonet) for detailed instructions.

### Prediction

The following script could create depth maps for all sequences:
```shell
./image2depth_kitti360.sh
```
## 3. Depth to pseudo point cloud
The following script could create pseudo point cloud for all sequences:

```shell
./depth2lidar_kitti360.sh
```

Finally we have the following data:
```
./kitti360/
        └── data_2d_raw/
        │        ├── 2013_05_28_drive_0000_sync/ # train:[0, 2, 3, 4, 5, 7, 10] + val:[6] + test:[9]
        │        │   ├── image_00/
        │        │   │   ├── data_rect/
        │        │   │   │	├── 000000.png
        │        │   │   │	├── 000001.png
        │        │   │   │	├── ...
        │        │   ├── image_01/
        │        │   │   ├── data_rect/
        │        │   │   │	├── 000000.png
        │        │   │   │	├── 000001.png
        │        │   │   │	├── ...
        │        │   └── voxels/
        │        ├── 2013_05_28_drive_0002_sync/
        │        ├── 2013_05_28_drive_0003_sync/
        │        .
        │        └── 2013_05_28_drive_0010_sync/
        └── preprocess/
	│        ├── labels/ 
	│        │   ├── 2013_05_28_drive_0000_sync/
	│        │   │   ├── 000000_1_1.npy
	│        │   │   ├── 000000_1_2.npy
	│        │   │   ├── 000000_1_8.npy
	│        │   │   ├── ...
	│        │   ├── 2013_05_28_drive_0002_sync/
	│        │   ├── 2013_05_28_drive_0003_sync/
	│        │   .
	│	 │   └── 2013_05_28_drive_0010_sync/ 
	│        ├── labels_half/ 
	│        └── unified/ 
	└── msnet3d_pseudo_lidar/
		 ├── 2013_05_28_drive_0000_sync/
		 │   ├── 000000.bin
		 │   ├── 000001.bin
		 │   ├── ...
		 ├── 2013_05_28_drive_0002_sync/
		 ├── 2013_05_28_drive_0003_sync/
		 .
		 └── 2013_05_28_drive_0010_sync/
```
