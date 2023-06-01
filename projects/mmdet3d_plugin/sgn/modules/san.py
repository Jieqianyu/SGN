import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from projects.mmdet3d_plugin.sgn.utils.ssc_loss import CE_ssc_loss


class SimpleSAN(nn.Module):

    def __init__(self, inplanes, num_classes):
        super().__init__()
        self.margin = 0
        self.IN = nn.InstanceNorm3d(inplanes, affine=True)
        self.selected_classes = list(range(num_classes))
        self.CFR_branches = nn.ModuleList()
        for i in self.selected_classes:
            self.CFR_branches.append(
                nn.Conv3d(3, 1, kernel_size=7, stride=1, padding=3, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.mask_matrix = None
        self.classifier = nn.Conv3d(inplanes, num_classes, kernel_size=1)

    def Regional_Normalization(self, region_mask, x):
        masked = x*region_mask
        RN_feature_map = self.IN(masked)
        return RN_feature_map

    def gen_norm_target(self, mid_ori, targets):
        outs = []
        for i in self.selected_classes:
            mask = torch.unsqueeze((targets == i).float(), 1)
            mask = F.interpolate(mask, size=mid_ori.size()[-3:], mode='nearest')
            out = mid_ori * mask
            out = self.IN(out)
            outs.append(out)
        mid_label = sum(outs)
        mid_label = self.relu(mid_label)
        return mid_label

    def cal_loss(self, pred, mid_ined, targets, mid_ori, class_weight):
        mid_label = self.gen_norm_target(mid_ori, targets)
        loss_in = 0.1 * F.smooth_l1_loss(mid_ined, mid_label)

        loss_ssc = 0.1 * CE_ssc_loss(F.interpolate(pred, size=targets.size()[-3:], mode='trilinear', align_corners=False), targets, class_weight)

        loss = loss_in + loss_ssc

        return loss

    def set_class_mask_matrix(self, heatmap, thres=0.2):
        normalized_map = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
        mask_matrix = (normalized_map > thres).float()

        return mask_matrix

    def forward(self, x):
        outs=[]
        idx = 0
        logit = self.classifier(x.detach())
        masks = F.softmax(logit,dim=1)
        for i in self.selected_classes:
            mask = torch.unsqueeze(masks[:,i],1)
            mid = x * mask
            avg_out = torch.mean(mid, dim=1, keepdim=True)
            max_out,_ = torch.max(mid,dim=1, keepdim=True)
            atten = torch.cat([avg_out,max_out,mask],dim=1)
            atten = self.sigmoid(self.CFR_branches[idx](atten))
            out = mid*atten
            heatmap = torch.mean(out, dim=1, keepdim=True)

            class_region = self.set_class_mask_matrix(heatmap)
            out = self.Regional_Normalization(class_region,out)
            outs.append(out)
        out_ = sum(outs)
        out_ = self.relu(out_)

        return out_, logit