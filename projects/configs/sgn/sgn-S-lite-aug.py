work_dir = '/public/experiments/yzdad/sgn/result/oed_more'
_base_ = [
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_dim_ = 64

_labels_tag_ = 'labels'
_temporal_ = []
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]

_imgW = 1280
_imgH = 384

_sem_scal_loss_ = True
_geo_scal_loss_ = True

model = dict(
   type='FSN',
   img_backbone=dict(
       type='LiteMono',
       model='lite-mono', 
       height=_imgH, 
       width=_imgW,
       drop_path_rate=0.2,
       pretrained='ckpts/encoder.pth',
       frozen=False),
   img_neck=dict(
       type='FPN',
       in_channels=[48, 80, 128],
       out_channels=_dim_,
       start_level=0,
       add_extra_convs='on_output',
       num_outs=3,
       relu_before_extra_convs=True),
   pts_neck=dict(
       type='DepthNetOri',
       num_ch_enc=[48, 80, 128],
       scales=[0, 1, 2],
       pretrained='ckpts/depth.pth',
       frozen=False),
   pts_bbox_head=dict(
       type='SGNHeadLite',
       bev_h=128,
       bev_w=128,
       bev_z=16,
       embed_dims=_dim_,
       sparse_header_dict=dict(
           type='SGNHeadOccLite',
           scene_size=[51.2, 51.2, 6.4],
           voxel_origin=[0, -25.6, -2],
           voxel_size=0.2,
           embed_dims=_dim_,
           img_size=[_imgH, _imgW],
           num_level=3),
       CE_ssc_loss=True,
       geo_scal_loss=_geo_scal_loss_,
       sem_scal_loss=_sem_scal_loss_,
       scale_2d_list=[16]
       ),
   train_cfg=dict(pts=dict(
       grid_size=[512, 512, 1],
       voxel_size=voxel_size,
       point_cloud_range=point_cloud_range,
       out_size_factor=4)))


dataset_type = 'SemanticKittiDataset'
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
       img_size=[_imgH, _imgW],
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       use_strong_img_aug=True),
   val=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       img_size=[_imgH, _imgW],
       temporal = _temporal_,
       labels_tag = _labels_tag_,),
   test=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       img_size=[_imgH, _imgW],
       temporal = _temporal_,
       labels_tag = _labels_tag_,),
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
