import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(
        self,
        channel
    ):
        super(ASPP, self).__init__()
        self.feature = channel
        c = channel//2
        self.reduce = nn.Conv3d(channel, c, kernel_size=1)
        self.aspp_1 = nn.Conv3d(c, c, kernel_size=3, padding=1, dilation=1)
        self.aspp_2 = nn.Conv3d(c, c, kernel_size=3, padding=2, dilation=2)
        self.aspp_4 = nn.Conv3d(c, c, kernel_size=3, padding=4, dilation=4)
        self.fusion = nn.Conv3d(c*3, channel, kernel_size=1)

    def forward(self, x3d):
        # [1, C, 128, 128, 16]
        x = self.reduce(x3d)
        x_1 = self.aspp_1(x)
        x_2 = self.aspp_2(x)
        x_3 = self.aspp_4(x)
        x = F.relu(self.fusion(torch.cat([x_1, x_2, x_3], dim=1)), inplace=True)

        return x


class MPAC(nn.Module):
    def __init__(self, channel, kernel=(3, 5, 7), dilation=(1, 1, 1), residual=False):
        super().__init__()
        self.channel = channel
        self.residual = residual
        self.n = len(kernel)  # number of kernels
        self.conv_mx = nn.Conv3d(channel, 3 * self.n, (1, 1, 1), stride=1, padding=0, bias=False, dilation=1)
        self.softmax = nn.Softmax(dim=2)  # Applies the Softmax function in each axis

        # ---- Convs of each axis
        self.conv_1x1xk = nn.ModuleList()
        self.conv_1xkx1 = nn.ModuleList()
        self.conv_kx1x1 = nn.ModuleList()

        c = channel
        for _idx in range(self.n):
            k = kernel[_idx]
            d = dilation[_idx]
            p = k // 2 * d
            self.conv_1x1xk.append(nn.Conv3d(c, c, (1, 1, k), stride=1, padding=(0, 0, p), bias=True, dilation=(1, 1, d)))
            self.conv_1xkx1.append(nn.Conv3d(c, c, (1, k, 1), stride=1, padding=(0, p, 0), bias=True, dilation=(1, d, 1)))
            self.conv_kx1x1.append(nn.Conv3d(c, c, (k, 1, 1), stride=1, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1)))

    def forward(self, x):
        mx = self.conv_mx(x)  # (BS, 3n, D, H, W)
        _bs, _, _d, _h, _w = mx.size()
        mx = mx.view(_bs, 3, -1, _d, _h, _w)  # (BS, 3, n, D, H, W)

        mx = self.softmax(mx)  # dim=2

        mx_c = torch.unsqueeze(mx, dim=3)  # (BS, 3, n, 1, D, H, W)
        mx_c = mx_c.expand(-1, -1, -1, self.channel, -1, -1, -1)  # (BS, 3, n, c, D, H, W)
        mx_list = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)

        mx_z_list = []
        mx_y_list = []
        mx_x_list = []
        for i in range(self.n):
            mx_z, mx_y, mx_x = torch.split(torch.squeeze(mx_list[i], dim=2), 1, dim=1)  # 3 x (BS, 1, c, D, H, W)
            mx_z_list.append(torch.squeeze(mx_z, dim=1))  # (BS, c, D, H, W)
            mx_y_list.append(torch.squeeze(mx_y, dim=1))  # (BS, c, D, H, W)
            mx_x_list.append(torch.squeeze(mx_x, dim=1))  # (BS, c, D, H, W)

        # ------ x ------
        y_x = None
        for _idx in range(self.n):
            y1_x = self.conv_1x1xk[_idx](x)
            y1_x = F.relu(y1_x, inplace=True)
            y1_x = torch.mul(mx_x_list[_idx], y1_x)
            y_x = y1_x if y_x is None else y_x + y1_x

        # ------ y ------
        y_y = None
        for _idx in range(self.n):
            y1_y = self.conv_1xkx1[_idx](y_x)
            y1_y = F.relu(y1_y, inplace=True)
            y1_y = torch.mul(mx_y_list[_idx], y1_y)
            y_y = y1_y if y_y is None else y_y + y1_y

        # ------ z ------
        y_z = None
        for _idx in range(self.n):
            y1_z = self.conv_kx1x1[_idx](y_y)
            y1_z = F.relu(y1_z, inplace=True)
            y1_z = torch.mul(mx_z_list[_idx], y1_z)
            y_z = y1_z if y_z is None else y_z + y1_z

        y = F.relu(y_z + x, inplace=True) if self.residual else F.relu(y_z, inplace=True)
        return y


class MPACv2(nn.Module):
    def __init__(self, channel, kernel=(3, 5, 7), dilation=(1, 1, 1), residual=False):
        super().__init__()
        self.channel = channel
        self.residual = residual
        self.n = len(kernel)  # number of kernels
        self.conv_mx = nn.Conv3d(channel, 3 * self.n, (1, 1, 1), stride=1, padding=0, bias=False, dilation=1)
        self.sigmoid = nn.Sigmoid()

        # ---- Convs of each path
        self.conv_k = nn.ModuleList()

        c = channel
        for _idx in range(self.n):
            k = kernel[_idx]
            d = dilation[_idx]
            p = k // 2 * d
            self.conv_k.append(nn.ModuleList([
                nn.Conv3d(c, c, (1, 1, k), stride=1, padding=(0, 0, p), bias=True, dilation=(1, 1, d)),
                nn.Conv3d(c, c, (1, k, 1), stride=1, padding=(0, p, 0), bias=True, dilation=(1, d, 1)),
                nn.Conv3d(c, c, (k, 1, 1), stride=1, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1))
            ]))

    def forward(self, x):
        mx = self.conv_mx(x)  # (BS, 3n, D, H, W)
        _bs, _, _d, _h, _w = mx.size()
        mx = mx.view(_bs, 3, -1, _d, _h, _w)  # (BS, 3, n, D, H, W)

        mx = self.sigmoid(mx)  # dim=2

        mx_c = torch.unsqueeze(mx, dim=3)  # (BS, 3, n, 1, D, H, W)
        mx_c = mx_c.expand(-1, -1, -1, self.channel, -1, -1, -1)  # (BS, 3, n, c, D, H, W)
        mx_list = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)

        y = None
        for _idx in range(self.n):
            y_k = x
            for _j in range(3):
                y_k = torch.mul(torch.squeeze(mx_list[_idx][:, _j], dim=1), F.relu(self.conv_k[_idx][_j](y_k), inplace=True))
            y = y_k if y is None else y + y_k

        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class SDB(nn.Module):
    def __init__(self, channel, out_channel, depth=3, version='v1'):
        super().__init__()

        c = out_channel
        self.conv_in = nn.Conv3d(channel, c, kernel_size=3, padding=1)
        basic_block = MPAC if version=='v1' else MPACv2
        blocks = [basic_block(c, residual=True) for _ in range(depth)]
        self.diffusion = nn.Sequential(*blocks)
        self.aspp = ASPP(c)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.diffusion(x)
        x = self.aspp(x)

        return x


