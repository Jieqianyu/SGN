import math

import torch

from torch import nn
from torch.nn import functional as F
from torch import einsum

import numpy as np
from einops import rearrange


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# transformer layer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_CA(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Cross_attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, out_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, y):
        b, n, m, _, h = *y.shape, self.heads
        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> (b n) h 1 d", h=h)
        k, v = map(lambda t: rearrange(t, "b n m (h d) -> (b n) h m d", h=h), kv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "(b n) h 1 d -> b n (h d)", b=b)

        if self.out_attention:
            return self.to_out(out), rearrange(attn, "(b n) h i j -> b n h (i j)", b=b)
        else:
            return self.to_out(out)


class KnnTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        heads=4,
        dim_head=64,
        mlp_dim=256,
        dropout=0.0,
        out_attention=False,
    ):
        super().__init__()
        self.out_attention = out_attention
        self.layers = nn.ModuleList([])
        self.depth = depth

        self.pe = PositionEmbeddingCoordsFourier(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm_CA(
                            dim,
                            Cross_attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                out_attention=self.out_attention,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def get_neighbor_feature(self, src_points, src_feats, query_points, k=16):
        dists = torch.norm(query_points[:, None] - src_points[:, :3], p=2, dim=2)
        knn_idx = dists.argsort()[:, :k]  # nq, k
            
        knn_xyz = index_points(src_points.unsqueeze(0), knn_idx.unsqueeze(0))[0] # nq, k, 3
        knn_feats = index_points(src_feats.unsqueeze(0), knn_idx.unsqueeze(0))[0] # nq, k, c

        return knn_xyz, knn_feats

    def forward(self, src_feats, src_points, query_feats, query_points, src_mask=None, query_mask=None):
        if src_mask is not None:
            src_points = src_points[src_mask]
            src_feats = src_feats[src_mask]
        if query_mask is not None:
            query_points = query_points[query_mask]
            query_feats = query_feats[query_mask]

        if self.out_attention:
            out_cross_attention_list = []

        neighbor_points, neighbor_feats = self.get_neighbor_feature(src_points, src_feats, query_points)
        query_pos = self.pe(query_points.unsqueeze(0))
        neighbor_pos = self.pe(neighbor_points.unsqueeze(0))

        x, y = query_feats.unsqueeze(0), neighbor_feats.unsqueeze(0)
        for i, (cross_attn, ff) in enumerate(self.layers):
            if self.out_attention:
                x_att, cross_att = cross_attn(
                    x + query_pos, y + neighbor_pos
                )
                out_cross_attention_list.append(cross_att)
            else:
                x_att = cross_attn(
                    x + query_pos, y + neighbor_pos
                )
            x = x_att + x
            x = ff(x) + x

        if self.out_attention:
            return x[0], torch.stack(out_cross_attention_list, dim=2)[0]
        else:
            return x[0], None


class PositionEmbeddingCoordsFourier(nn.Module):
    def __init__(
        self,
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        assert d_pos is not None
        assert d_pos % 2 == 0
        # define a gaussian matrix input_ch -> output_ch
        B = torch.empty((d_in, d_pos // 2)).normal_()
        B *= gauss_scale
        self.register_buffer("gauss_B", B)
        self.d_pos = d_pos

    def get_fourier_embeddings(self, xyz, num_channels=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x npoints d_pos x  embedding
        final_embeds = torch.cat(final_embeds, dim=2)
        return final_embeds

    def forward(self, xyz, num_channels=None):
        assert isinstance(xyz, torch.Tensor)
        assert (xyz.ndim==2 or xyz.ndim == 3 or xyz.ndim==4)

        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels)[0]

        if xyz.ndim == 4:
            b, x, y, _ = xyz.shape
            xyz = xyz.flatten(0, 1)
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels).reshape(b, x, y, -1)

        with torch.no_grad():
            return self.get_fourier_embeddings(xyz, num_channels)