#!/usr/bin/env bash

set -e
exeFunc(){
    cd mobilestereonet
    baseline=$1
    num_seq=$2
    CUDA_VISIBLE_DEVICES=0 python prediction.py --datapath ../kitti/dataset/sequences/$num_seq \
    --testlist ./filenames/$num_seq.txt --num_seq $num_seq --loadckpt ./MSNet3D_SF_DS_KITTI2015.ckpt --dataset kitti \
    --model MSNet3D --savepath "../kitti/dataset/sequences_msnet3d_depth" --baseline $baseline
    cd ..
}

for i in {00..02}
do
    exeFunc 388.1823 $i     
done

for i in {03..03}
do
    exeFunc 389.6304 $i     
done

for i in {04..12}
do
    exeFunc 381.8293 $i     
done

for i in {13..21}
do
    exeFunc 388.1823 $i     
done
