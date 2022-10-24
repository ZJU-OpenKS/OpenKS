# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

# from util.misc import NestedTensor


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats # TODO concat with input dim
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi * 32
        self.scale = scale

    def forward(self, xyz):  # xyz: to be normalized
        # input shape : B, N, C
        B, N, C = xyz.shape  # C=3
        xyz = xyz * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # print(dim_t, xyz, flush=True)

        # pos_dim = [xyz]
        pos_dim = []
        for i in range(C):
            # print(xyz[:, :, i, None].shape, dim_t.shape)
            pos_embd_dim = xyz[:, :, i, None].repeat(1, 1, self.num_pos_feats) / dim_t
            pos_embd_dim = torch.cat((pos_embd_dim[:, :, 0::2].sin(), pos_embd_dim[:, :, 1::2].cos()), dim=-1)
            # print(pos_embd_dim.shape, '<< pos embd shape', flush=True)
            # print(pos_embd_dim, '<< pos embd dim')
            # print(pos_embd_dim[0, 0, :], '<< pos embed', flush=True)
            pos_dim.append(pos_embd_dim.contiguous())
        # pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        val_xyz = torch.cat(pos_dim, dim=-1)
        return val_xyz  # final


def build_position_encoding(position_embedding, hidden_dim, input_dim, scale=None):
    N_steps = hidden_dim // input_dim
    assert hidden_dim % input_dim == 0, 'position encoding not divisable by input_dim'
    assert N_steps > 0, 'you should have position encoding'
    if position_embedding in ('sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine3D(num_pos_feats=N_steps, scale=scale)  # normalize ??
    elif position_embedding in ('learned'):
        # position_embedding = PositionEmbeddingLearned(N_steps)
        pass
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding
