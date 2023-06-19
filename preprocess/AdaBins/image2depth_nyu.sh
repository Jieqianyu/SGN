#!/usr/bin/env bash

# python prediction.py --src_dir /public/datasets/NYUv2/depthbin/NYUtrain --dst_dir /public/datasets/NYUv2/adabins_depth/NYUtrain
# python prediction.py --src_dir /public/datasets/NYUv2/depthbin/NYUtest --dst_dir /public/datasets/NYUv2/adabins_depth/NYUtest

python prediction.py --src_dir /home/jm/Downloads/Compressed/nyu/NYU_dataset/depthbin/NYUtrain --dst_dir /home/jm/Downloads/Compressed/nyu/NYU_dataset/adabins_depth/NYUtrain
python prediction.py --src_dir /home/jm/Downloads/Compressed/nyu/NYU_dataset/depthbin/NYUtest --dst_dir /home/jm/Downloads/Compressed/nyu/NYU_dataset/adabins_depth/NYUtest