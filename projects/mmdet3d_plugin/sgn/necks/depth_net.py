from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from .layers import *
from timm.models.layers import trunc_normal_

from mmdet.models import NECKS
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast


@NECKS.register_module()
class DepthNet(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), use_skips=True, min_depth=2, max_depth=54, num_bins=128, **kwargs):
        super().__init__()

        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = np.array(num_ch_enc)
        self.num_ch_dec = self.num_ch_enc
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_bins = num_bins

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # print(i, num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_bins)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

        if kwargs.get('pretrained', None) is not None:
            depth_dict = torch.load(kwargs['pretrained'], map_location='cpu')
            print('load pretrained ckpt for depth net')
            self.load_state_dict({k: v for k, v in depth_dict.items() if k in self.state_dict()})

        if kwargs.get('frozen', False):
            print('depth frozen')
            for k, v in self.named_parameters():
                v.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        outputs = []
        x = input_features[-1]
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                f = self.convs[("dispconv", i)](x)
            depth = self.sigmoid(f)
            outputs.append(depth)

        return outputs

    def get_downsampled_gt_depth(self, gt_depths, scale):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        fH, fW = H // scale, W // scale

        gt_depths = gt_depths.view(B * N, fH, scale, fW, scale, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, scale * scale)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, fH, fW)
        
        gt_depths = bin_depths(gt_depths, self.min_depth, self.max_depth, self.num_bins, target=True)
        gt_depths = F.one_hot(gt_depths, num_classes=self.num_bins + 1).view(-1, self.num_bins + 1)[:, :self.num_bins]
        
        return gt_depths.float()
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels, scale=depth_labels.shape[-2]//depth_preds.shape[-2])
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.num_bins)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return depth_loss


    def losses(self, preds, img_metas):
        depths = []
        for img_meta in img_metas:
            depths.append(img_meta['depth'])
        depths = preds[0].new_tensor(np.asarray(depths))  # B, N, H, W

        losses = {}
        for i, pred in enumerate(preds):
            losses[f'loss_depth_{i}'] = self.get_depth_loss(depths, pred)
        return losses


def bin_depths(depth_map, depth_min, depth_max, num_bins, target=False, mode='UD'):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = (
            num_bins
            * (torch.log(1 + depth_map) - math.log(1 + depth_min))
            / (math.log(1 + depth_max) - math.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)

    return indices
