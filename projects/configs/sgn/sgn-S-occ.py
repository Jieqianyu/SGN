work_dir = 'result/occ'
_base_ = [
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_dim_ = 128

_labels_tag_ = 'labels'
_temporal_ = []
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
img_size = (370, 1220)

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

# downsample ratio in [x, y, z] when generating 3D volumes in LSS
lss_downsample = [2, 2, 2]
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

_sem_scal_loss_ = True
_geo_scal_loss_ = True
_depthmodel_= 'msnet3d'
_nsweep_ = 10
_query_tag_ = 'query_iou5348_pre7043_rec6897'

model = dict(
   type='SGN',
   pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
   img_backbone=dict(
       type='ResNet',
       depth=50,
       num_stages=4,
       out_indices=(1, 2, 3),
       frozen_stages=0,
       norm_cfg=dict(type='BN', requires_grad=False),
       norm_eval=True,
       style='pytorch'),
   img_neck=dict(
       type='FPN',
       in_channels=[512, 1024, 2048],
       out_channels=_dim_,
       start_level=0,
       add_extra_convs='on_output',
       num_outs=3,
       relu_before_extra_convs=True),
   pts_bbox_head=dict(
       type='SGNHeadOcc',
       bev_h=128,
       bev_w=128,
       bev_z=16,
       embed_dims=_dim_,
       lss_depth=dict(
           depth_config=dict(
                in_channels=_dim_,
                mid_channels=_dim_//2),
           downsample=8,
           grid_config=grid_config,
           data_config={'input_size': img_size},
       ),
       CE_ssc_loss=True,
       geo_scal_loss=_geo_scal_loss_,
       sem_scal_loss=_sem_scal_loss_,
       scale_2d_list=[8, 16, 32]
       ),
   train_cfg=dict(pts=dict(
       grid_size=[512, 512, 1],
       voxel_size=voxel_size,
       point_cloud_range=point_cloud_range,
       out_size_factor=4)))


dataset_type = 'SemanticKittiDatasetStage2'
data_root = './kitti/'
file_client_args = dict(backend='disk')

data = dict(
   samples_per_gpu=1,
   workers_per_gpu=4,
   train=dict(
       type=dataset_type,
       split = "train",
       test_mode=False,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       img_size = img_size,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   val=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       img_size = img_size,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   test=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       img_size = img_size,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   shuffler_sampler=dict(type='DistributedGroupSampler'),
   nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
   type='AdamW',
   lr=2e-4,
   weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
   policy='CosineAnnealing',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=1.0 / 3,
   min_lr_ratio=1e-3)
total_epochs = 48
evaluation = dict(interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
   interval=50,
   hooks=[
       dict(type='TextLoggerHook'),
       dict(type='TensorboardLoggerHook')
   ])

checkpoint_config = None
# checkpoint_config = dict(interval=2)
