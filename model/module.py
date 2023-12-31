import torch
from torch import nn, einsum
import numpy as np
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.init import xavier_normal
import copy


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class CBT(nn.Module):
    # myTransformer_nonorm
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)))

    def forward(self, x):
        for attn in self.layers:
            x = attn(x) + x
        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def mask_create(self, dots):
        mask = dots.softmax(dim=-1)
        # mask = dots.clone()
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                tmp = mask[i][j].diagonal()
                NeedToCorrectTable = torch.abs((tmp - tmp.mean(0)) / tmp.std(0)) > 1.5
                for k, status in enumerate(NeedToCorrectTable):
                    if status is True:
                        mask[i][j][:, k] = 0
                        mask[i][j][k, :] = 0
        # mask = ((mask > 0).float() - 1) * 9999  # work well on some case, but not mine
        return mask

    def mask_softmax(self, matrix, mask, dim=-1):
        matrix_c = matrix.clone()
        matrix_c[mask == 0] = -float('inf')
        tmp = matrix_c.softmax(dim=-1)
        zero_matrix = torch.zeros_like(tmp)
        result = torch.where(torch.isnan(tmp), zero_matrix, tmp)
        # result = (matrix + mask).softmax(dim=dim)  # work well on some case, but not mine
        return result

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        mask = self.mask_create(dots)
        attn = self.mask_softmax(dots, mask)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class AMF(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(PreNorm(dim, AdaptiveFusion(dim, heads=heads, dim_head=dim_head, dropout=dropout)))

    def forward(self, x):
        for attn in self.layers:
            x = attn(x) + x
        return x
