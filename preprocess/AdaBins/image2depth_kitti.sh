#!/usr/bin/env bash

set -e
exeFunc(){
    num_seq=$1
    CUDA_VISIBLE_DEVICES=0 python prediction.py --src_dir ../kitti/dataset/sequences/$num_seq/image_2 \
    --dst_dir ../kitti/dataset/sequences_adabins_depth/sequences/$num_seq \
    --dataset kitti
}

for i in {00..21}
do
    exeFunc $i
done
