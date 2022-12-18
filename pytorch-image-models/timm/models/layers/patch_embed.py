""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from .helpers import to_2tuple
from .trace_utils import _assert


class PatchEmbedConv(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class PatchEmbedLinear(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Linear(in_chans * patch_size[0] * patch_size[1], embed_dim, bias=bias)
        nn.init.normal_(self.proj.weight, mean=0, std=embed_dim ** -0.5)
        if self.proj.bias is not None:
            nn.init.normal_(self.proj.bias, mean=0, std=embed_dim ** -0.5)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        pH, pW = self.patch_size
        H, W = self.grid_size
        # B x C x H' x pH x W' x pW -> B x H' x W' x C x pH x pW
        x = x.view(B, C, H, pH, W, pW).permute(0, 2, 4, 1, 3, 5)
        # B x L x D
        x = x.reshape(B, H * W, C * pH * pW)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True, flatten=True, impl='conv'):
        super().__init__()
        if impl == 'conv':
            self.pemb = PatchEmbedConv(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                bias=bias,
                flatten=flatten
            )
        elif impl == 'linear':
            self.pemb = PatchEmbedLinear(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                bias=bias
            )
        else:
            raise ValueError('Unknown impl for patch embedding: {}'.format(impl))

    def forward(self, x):
        return self.pemb(x)

    @property
    def num_patches(self):
        return self.pemb.num_patches
