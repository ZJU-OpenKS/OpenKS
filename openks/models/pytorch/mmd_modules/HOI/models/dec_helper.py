import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .tr_helper import get_activation_fn

__all__ = ['decoder_layer_builder']


def decoder_layer_builder(dec_type: str):
    decoder_entries = {
        'vanilla': DecoderLayerVanilla,
        'vanilla_bottleneck': DecoderLayerBottleneck,
    }
    return decoder_entries[dec_type]


class DecoderLayerVanilla(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        del tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        del tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt), inplace=True)))
        tgt = tgt + self.dropout3(tgt2)
        del tgt2
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        del tgt2

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        del tgt2

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2), inplace=True)))
        tgt = tgt + self.dropout3(tgt2)
        del tgt2
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, **kwargs):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class DecoderLayerBottleneck(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 split_query=False,
                 interact_query=False,
                 **kwargs):
        super().__init__()
        # Multihead Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Multihead Cross-Attention
        self.cross_attn_sub = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_obj = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_verb = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_sub = nn.LayerNorm(d_model)
        self.norm_obj = nn.LayerNorm(d_model)
        self.norm_verb = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_sub = nn.Dropout(dropout)
        self.dropout_obj = nn.Dropout(dropout)
        self.dropout_verb = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.split_query = split_query

        self.interactive = interact_query
        if self.interactive:
            self.so_fuse = nn.Linear(d_model*2, d_model)
            self.cross_attn_v2so = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.dropout_v2so = nn.Dropout(dropout)
            self.norm_v2so = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_one_branch_post(self, tgt, mem, branch,
                                tgt_mask: Optional[Tensor] = None,
                                memory_mask: Optional[Tensor] = None,
                                tgt_key_padding_mask: Optional[Tensor] = None,
                                memory_key_padding_mask: Optional[Tensor] = None,
                                pos: Optional[Tensor] = None,
                                query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = getattr(self, f'cross_attn_{branch}')(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(mem, pos),
            value=mem, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + getattr(self, f'dropout_{branch}')(tgt2)
        tgt = getattr(self, f'norm_{branch}')(tgt)

        return tgt

    def ffn_post(self, tgt, branch):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt), inplace=True)))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_one_branch_pre(self, tgt, mem, branch,
                               tgt_mask: Optional[Tensor] = None,
                               memory_mask: Optional[Tensor] = None,
                               tgt_key_padding_mask: Optional[Tensor] = None,
                               memory_key_padding_mask: Optional[Tensor] = None,
                               pos: Optional[Tensor] = None,
                               query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)

        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = getattr(self, f'norm_{branch}')(tgt)

        tgt2 = getattr(self, f'cross_attn_{branch}')(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(mem, pos),
            value=mem, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + getattr(self, f'dropout_{branch}')(tgt2)

        return tgt

    def ffn_pre(self, tgt, branch):
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2), inplace=True)))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt_sub, tgt_obj, tgt_verb,
                mem_sub, mem_obj, mem_verb,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            raise NotImplementedError

        if self.split_query:
            h_query_pos = query_pos['h_query']
            o_query_pos = query_pos['o_query']
            v_query_pos = query_pos['v_query']
        else:
            h_query_pos = o_query_pos = v_query_pos = query_pos
            
        tgt_sub = self.forward_one_branch_post(
            tgt_sub, mem_sub, 'sub', tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask, pos, h_query_pos)
        tgt_obj = self.forward_one_branch_post(
            tgt_obj, mem_obj, 'obj', tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask, pos, o_query_pos)
        tgt_verb = self.forward_one_branch_post(
            tgt_verb, mem_verb, 'verb', tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask, pos, v_query_pos)

        if self.interactive:
            so_fused = self.so_fuse(torch.cat([tgt_sub, tgt_obj], dim=-1))
            tgt_verb2 = self.cross_attn_v2so(
                query=self.with_pos_embed(tgt_verb, v_query_pos),
                key=self.with_pos_embed(so_fused, (h_query_pos + o_query_pos)),
                value=so_fused, attn_mask=None,
                key_padding_mask=tgt_key_padding_mask)[0]
            tgt_verb = self.norm_v2so(tgt_verb + self.dropout_v2so(tgt_verb2))

        tgt_sub = self.ffn_post(tgt_sub, 'sub')
        tgt_obj = self.ffn_post(tgt_obj, 'obj')
        tgt_verb = self.ffn_post(tgt_verb, 'verb')

        return tgt_sub, tgt_obj, tgt_verb
