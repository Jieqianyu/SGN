import torch
import torch.nn as nn
import numpy as np


class FLoSP(nn.Module):
    def __init__(self, scale_2d_list):
        super().__init__()

        self.scale_2d_list = scale_2d_list

    def project(self, x2d, projected_pix, fov_mask):
        bs, c, h, w = x2d.shape

        src = x2d.view(bs, c, -1)
        zeros_vec = torch.zeros(bs, c, 1).type_as(src)
        src = torch.cat([src, zeros_vec], -1)

        pix_x, pix_y = projected_pix[:, :, 0], projected_pix[:, :, 1]
        img_indices = pix_y * w + pix_x
        img_indices[~fov_mask] = h * w
        img_indices = img_indices.unsqueeze(1).expand(-1, c, -1).long()  # b, c, N
        x = torch.gather(src, -1, img_indices)  # b, c, N

        return x

    def forward(self, mlvl_feats, img_metas):
        assert len(self.scale_2d_list) == len(mlvl_feats)

        # projected_pix: bs, num_cam, N, 2; fov_mask: bs, num_cam, N
        projected_pix, fov_mask = [], []
        for img_meta in img_metas:
            projected_pix.append(img_meta['projected_pix'])
            fov_mask.append(img_meta['fov_mask'])
        projected_pix = np.asarray(projected_pix)
        fov_mask = np.asarray(fov_mask)

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        device = mlvl_feats[0].device

        projected_pix = torch.from_numpy(projected_pix).flatten(0, 1).long().to(device)
        fov_mask = torch.from_numpy(fov_mask).flatten(0, 1).to(device)

        x3d = None
        for _i, scale_2d in enumerate(self.scale_2d_list):
            # project features at each 2D scale to target 3D scale
            scale_2d = int(scale_2d)
            x_scale_2d = mlvl_feats[_i].flatten(0, 1) # bs*num_cam, c, h, w

            # Sum all the 3D features
            if x3d is None:
                x3d = self.project(
                    x_scale_2d,
                    projected_pix // scale_2d,
                    fov_mask,
                )
            else:
                x3d += self.project(
                    x_scale_2d,
                    projected_pix // scale_2d,
                    fov_mask,
                )

        _, c, nq = x3d.shape
        x3d = x3d.view(bs, num_cam, c, nq)
        if num_cam > 1:
            fov_mask = fov_mask.view(bs, num_cam, nq)
            weights = torch.sum(fov_mask, dim=1)  # bs, nq
            weights[weights == 0] = 1
            x3d = torch.sum(x3d * fov_mask.unsqueeze(-2).float(), dim=1) / weights[:, None]
        else:
            x3d = x3d.squeeze(1)

        return x3d