<div align="center">   
  
# Camera-based Semantic Scene Completion with Sparse Guidance Network
</div>

> **Camera-based Semantic Scene Completion with Sparse Guidance Network**. 

> Jianbiao Mei, Yu Yang, Mengmeng Wang, Junyu Zhu, Xiangrui Zhao, Jongwon Ra, Laijian Li, Yong Liu*

>  [[Arxiv]](https://arxiv.org/abs/2312.05752)


## News
- [2023/12]: Our paper is on [arxiv](https://arxiv.org/abs/2312.05752).
- [2023/08]: SGN achieve the SOTA on Camera-based [SemanticKITTI 3D SSC (Semantic Scene Completion) Task](http://www.semantic-kitti.org/tasks.html#ssc) with **15.76% mIoU** and **45.52% IoU**.
</br>


## Abstract
Semantic scene completion (SSC) aims to predict the semantic occupancy of each voxel in the entire 3D scene from limited observations, which is an emerging and critical task for autonomous driving. Recently, many studies have turned to camera-based SSC solutions due to the richer visual cues and cost-effectiveness of cameras. However, existing methods usually rely on sophisticated and heavy 3D models to directly process the lifted 3D features that are not discriminative enough for clear segmentation boundaries. In this paper, we adopt the dense-sparse-dense design and propose an end to-end camera-based SSC framework, termed SGN, to diffuse semantics from the semantic- and occupancy-aware seed voxels to the whole scene based on geometry prior and occupancy information. By designing hybrid guidance (sparse semantic and geometry guidance) and effective voxel aggregation for spatial occupancy and geometry priors, we enhance the feature separation between different categories and expedite the convergence of semantic diffusion. Extensive experimental results on the SemanticKITTI dataset demonstrate the superiority of our SGN over existing state-of-the-art methods.


## Method

| ![SGN.jpg](./teaser/arch.png) | 
|:--:| 
| ***Figure 1. Overall framework of SGN**. The image encoder extracts 2D features to provide the foundation for 3D features lifted by the view transformation. Then auxiliary occupancy head is applied to provide geometry guidance. Before sparse semantic guidance, depth-based occupancy prediction is utilized for voxel proposals of indexing seed features. Afterward, the voxel aggregation layer forms the informative voxel features processed by the multi-scale semantic diffusion for the final semantic occupancy prediction. KT denotes the knowledge transfer layer for geometry prior.* |

## Getting Started
- [Installation and Dataset](https://github.com/NVlabs/VoxFormer). Please refer to Voxformer for the details.
- Run and Eval
  
Train SGN with 4 GPUs 
```
./tools/dist_train.sh ./projects/configs/sgn/sgn-T-one-stage-guidance.py 4
```

Eval SGN with 4 GPUs
```
./tools/dist_test.sh ./projects/configs/sgn/sgn-T-one-stage-guidance.py ./path/to/ckpts.pth 4
```

## Model Zoo
Coming soon...

| Backbone | Dataset| Method | IoU| mIoU | Config | Download |
| :---: | :---: | :---: | :---: | :---:| :---: | :---: |
| R50 | Sem.KITTI test| SGN-T |45.42| 15.76|[config](./projects/configs/sgn/sgn-T-one-stage-guidance.py) |[model]() 
| R18 | Sem.KITTI test| SGN-L | 43.71| 14.39|[config](./projects/configs/sgn/sgn-L-one-stage-guidance.py) |[model]()|
| R50 | Sem.KITTI test| SGN-S | 41.88| 14.01|[config](./projects/configs/sgn/sgn-S-one-stage-guidance.py) |[model]()|

Note that we used the ones that performed best on the validation set during training to test on the web server. You can acquire better results on test sets when incorporating validation images for training.
 
## TODO

- [x] SemanticKITTI
- [ ] Model Zoo
- [ ] SSCBench-KITTI-360

## Bibtex
If this work is helpful for your research, please cite the following BibTeX entry.

```
@article{mei2023camera,
      title={Camera-based 3D Semantic Scene Completion with Sparase Guidance Network},
      author={Mei, Jianbiao and Yang, Yu and Wang, Mengmeng and Zhu, Junyu and Zhao, Xiangrui and Ra Jongwon and Li, Laijian and Liu, Yong},
      journal={arXiv preprint arXiv:2312.05752},
      year={2023}
}
```

## Acknowledgement

Many thanks to these excellent open source projects:
- [VoxFormer](https://github.com/NVlabs/VoxFormer)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [AICNet](https://github.com/waterljwant/SSC)
