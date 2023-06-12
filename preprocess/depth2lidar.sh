set -e
exeFunc(){
    num_seq=$1
    python utils/depth2lidar.py --calib_dir  ./kitti/dataset/sequences/$num_seq \
    --depth_dir ./mobilestereonet/depth/sequences/$num_seq \
    --save_dir ./mobilestereonet/lidar/sequences/$num_seq

    cp data_odometry_calib/sequences/$num_seq/calib.txt ./mobilestereonet/lidar/sequences/$num_seq/
    cp data_odometry_calib/sequences/$num_seq/poses.txt ./mobilestereonet/lidar/sequences/$num_seq/
}
# Change data_path to your own specified path
# And make sure there is enough space under data_path to store the generated data
# data_path=/mnt/NAS/data/yiming/segformer3d_data
# data_path=/public/datasets/SemanticKITTI/dataset

# mkdir -p $data_path/sequences_msnet3d_lidar
# ln -s $data_path/sequences_msnet3d_lidar ./mobilestereonet/lidar
for i in {00..21}
do
    exeFunc $i
done
