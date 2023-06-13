from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from .layers import *
from timm.models.layers import trunc_normal_

from mmdet.models import NECKS


@NECKS.register_module()
class DepthNetOri(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, **kwargs):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = np.array(num_ch_enc)
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

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
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

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
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
            _, depth = disp_to_depth(self.sigmoid(f), 0.1, 100)
            outputs.append(depth)

        return outputs

    def downsample_depth(self, depths, scale):
        """
        Input:
            depths: [B, N, H, W]
        Output:
            gt_depths: [B, N, h, w]
        """
        B, N, H, W = depths.shape
        fH, fW = H // scale, W // scale

        depths = depths.view(B * N, fH, scale, fW, scale, 1)
        depths = depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        depths = depths.view(-1, scale * scale)
        depths_tmp = torch.where(depths == 0.0, 1e5 * torch.ones_like(depths), depths)
        depths = torch.min(depths_tmp, dim=-1).values
        depths = depths.view(B, N, fH, fW)
        
        return depths.float()

    def losses(self, preds, img_metas, mask=None):
        depths = []
        for img_meta in img_metas:
            depths.append(img_meta['depth'])
        depths = preds[0].new_tensor(np.asarray(depths))  # B, N, H, W

        losses = {}
        for i, pred in enumerate(preds):
            target = self.downsample_depth(depths, scale=int(depths.shape[-2]/pred.shape[-2]))
            if mask is not None:
                pred = pred.suqeeze(2)[mask]
                target = target[mask]
            g = torch.log(pred) - torch.log(target)
            Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
            losses[f'loss_depth_{i}'] = torch.sqrt(Dg)
        return losses