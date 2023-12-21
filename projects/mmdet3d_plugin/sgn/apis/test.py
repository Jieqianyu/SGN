# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os.path as osp
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from projects.mmdet3d_plugin.sgn.utils.ssc_metric import SSCMetrics

def custom_single_gpu_test(model, data_loader, show=False, out_dir=None):
    model.eval()
    
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    # evaluate ssc
    ssc_metric = SSCMetrics(len(dataset.class_names)).cuda()
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
    
        output_voxels = torch.argmax(result['output_voxels'], dim=1)
        target_voxels = result['target_voxels'].clone()
        ssc_metric.update(y_pred=output_voxels,  y_true=target_voxels)
        
        batch_size = output_voxels.shape[0]
        for _ in range(batch_size):
            prog_bar.update()
    
    res = {
        'ssc_scores': ssc_metric.compute(),
    }
    
    return res


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    ssc_results = []
    # evaluate ssc
    ssc_metric = SSCMetrics(len(dataset.class_names)).cuda()
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
        output_voxels = torch.argmax(result['output_voxels'], dim=1)
            
        if result['target_voxels'] is not None:
            target_voxels = result['target_voxels'].clone()
            ssc_results_i = ssc_metric.compute_single(
                y_pred=output_voxels, y_true=target_voxels)
            ssc_results.append(ssc_results_i)
            
        batch_size = output_voxels.shape[0]
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    # wait until all predictions are generated
    dist.barrier()
    
    res = {}
    res['ssc_results'] = collect_results_cpu(ssc_results, len(dataset), tmpdir)
    
    return res


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results