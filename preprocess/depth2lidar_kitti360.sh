#!/usr/bin/env bash

set -e
exeFunc(){
    num_seq=$1
    python utils/depth2lidar_kitti360.py \
    --depth_dir ../kitti360/msnet3d_depth/sequences/$num_seq \
    --save_dir ../kitti360/msnet3d_pseudo_lidar/$num_seq
}

seqs=("2013_05_28_drive_0000_sync" "2013_05_28_drive_0002_sync" "2013_05_28_drive_0003_sync" "2013_05_28_drive_0004_sync" "2013_05_28_drive_0005_sync" "2013_05_28_drive_0006_sync" "2013_05_28_drive_0007_sync" "2013_05_28_drive_0009_sync" "2013_05_28_drive_0010_sync")
for i in ${seqs[@]}
do
    exeFunc $i
done
