#!/usr/bin/env bash

set -e
exeFunc(){
    num_seq=$1
    python utils/depth2lidar.py --calib_dir  ./kitti/dataset/sequences/$num_seq \
    --depth_dir ./kitti/dataset/sequences_adabins_depth/sequences/$num_seq \
    --save_dir ./kitti/dataset/sequences_adabins_lidar/sequences/$num_seq
}

cd ..
for i in {00..21}
do
    exeFunc $i
done
