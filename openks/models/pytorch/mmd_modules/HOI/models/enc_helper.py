import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .tr_helper import get_activation_fn

__all__ = ['encoder_layer_builder']


def encoder_layer_builder(enc_type: str):
    encoder_entries = {
        'vanilla': EncoderLayerVanilla,
        'hoi_bottleneck': EncoderLayerBottleneck,
    }
    return encoder_entries[enc_type]


class EncoderLayerVanilla(nn.Module):

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

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src), inplace=True)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2), inplace=True)))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class EncoderLayerBottleneck(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=256,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.sub_proj = nn.Linear(d_model, d_model)
        self.obj_proj = nn.Linear(d_model, d_model)
        self.verb_proj = nn.Linear(d_model, d_model)

        self.cross_attn_verb = nn.MultiheadAttention(d_model, nhead, dropout)
        self.norm_verb = nn.LayerNorm(d_model)
        self.dropout_verb = nn.Dropout(dropout)
        self.reduce_dim = nn.Linear(2 * d_model, d_model)

        self.linear1 = self.dropout = self.linear2 = None
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def eye_init(self):
        nn.init.eye_(self.sub_proj.weight)
        nn.init.constant_(self.sub_proj.bias, 0)
        nn.init.eye_(self.obj_proj.weight)
        nn.init.constant_(self.obj_proj.bias, 0)
        nn.init.eye_(self.verb_proj.weight)
        nn.init.constant_(self.verb_proj.bias, 0)

    def forward_cross_attn(self, src_sub, src_obj, src_verb, pos, src_key_padding_mask):
        embed_sub = src_sub
        embed_obj = src_obj

        dst_verb = torch.cat([embed_sub, embed_obj], dim=-1)
        dst_verb = self.reduce_dim(dst_verb)
        if self.normalize_before:
            embed_verb = self.cross_attn_pre(src=src_verb, dst=dst_verb, tgt='verb', pos=pos,
                                             dst_key_padding_mask=src_key_padding_mask)
        else:
            embed_verb = self.cross_attn_post(src=src_verb, dst=dst_verb, tgt='verb', pos=pos,
                                              dst_key_padding_mask=src_key_padding_mask)
        return embed_sub, embed_obj, embed_verb

    def cross_attn_post(self, src, dst, tgt,
                        dst_mask: Optional[Tensor] = None,
                        dst_key_padding_mask: Optional[Tensor] = None,
                        pos: Optional[Tensor] = None):
        src2 = getattr(self, f'cross_attn_{tgt}')(
            query=self.with_pos_embed(src, pos),
            key=self.with_pos_embed(dst, pos),
            value=dst, attn_mask=dst_mask,
            key_padding_mask=dst_key_padding_mask)[0]
        src = getattr(self, f'norm_{tgt}')(src + getattr(self, f'dropout_{tgt}')(src2))

        return src

    def cross_attn_pre(self, src, dst, tgt,
                       dst_mask: Optional[Tensor] = None,
                       dst_key_padding_mask: Optional[Tensor] = None,
                       pos: Optional[Tensor] = None):
        src2 = getattr(self, f'cross_attn_{tgt}')(
            query=self.with_pos_embed(getattr(self, f'norm_{tgt}')(src), pos),
            key=self.with_pos_embed(dst, pos),
            value=dst, attn_mask=dst_mask,
            key_padding_mask=dst_key_padding_mask)[0]
        src = src + getattr(self, f'dropout_{tgt}')(src2)

        return src

    def forward_post(self, src_sub, src_obj, src_verb,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        src_sub = self.sub_proj(src_sub)
        q_s = k_s = self.with_pos_embed(src_sub, pos)
        src_sub2 = self.self_attn(q_s, k_s, value=src_sub, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src_sub = self.norm1(src_sub + self.dropout1(src_sub2))

        src_obj = self.obj_proj(src_obj)
        q_o = k_o = self.with_pos_embed(src_obj, pos)
        src_obj2 = self.self_attn(q_o, k_o, value=src_obj, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src_obj = self.norm1(src_obj + self.dropout1(src_obj2))

        src_verb = self.verb_proj(src_verb)
        q_v = k_v = self.with_pos_embed(src_verb, pos)
        src_verb2 = self.self_attn(q_v, k_v, value=src_verb, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]
        src_verb = self.norm1(src_verb + self.dropout1(src_verb2))

        src_sub, src_obj, src_verb = self.forward_cross_attn(src_sub, src_obj, src_verb, pos,
                                                             src_key_padding_mask)
        src_sub2 = self.linear2(self.dropout(self.activation(self.linear1(src_sub), inplace=True)))
        src_sub = self.norm2(src_sub + self.dropout2(src_sub2))

        src_obj2 = self.linear2(self.dropout(self.activation(self.linear1(src_obj), inplace=True)))
        src_obj = self.norm2(src_obj + self.dropout2(src_obj2))

        src_verb2 = self.linear2(self.dropout(self.activation(self.linear1(src_verb), inplace=True)))
        src_verb = self.norm2(src_verb + self.dropout2(src_verb2))

        return src_sub, src_obj, src_verb

    def forward_pre(self, src_sub, src_obj, src_verb,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src_sub = self.sub_proj(src_sub)
        src_sub2 = self.norm_sub(src_sub)
        q_s = k_s = self.with_pos_embed(src_sub2, pos)
        src_sub2 = self.self_attn(q_s, k_s, value=src_sub2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src_sub = src_sub + self.dropout_sub(src_sub2)

        src_obj = self.obj_proj(src_obj)
        src_obj2 = self.norm_obj(src_obj)
        q_o = k_o = self.with_pos_embed(src_obj2, pos)
        src_obj2 = self.self_attn(q_o, k_o, value=src_obj2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src_obj = src_obj + self.dropout_sub(src_obj2)

        src_verb = self.verb_proj(src_verb)
        src_verb2 = self.norm_verb(src_verb)
        q_v = k_v = self.with_pos_embed(src_verb2, pos)
        src_verb2 = self.self_attn(q_v, k_v, value=src_verb2, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]
        src_verb = src_verb + self.dropout_verb(src_verb2)

        src_sub, src_obj, src_verb = self.forward_cross_attn(src_sub, src_obj, src_verb, pos,
                                                             src_key_padding_mask)
        src_sub2 = self.norm2(src_sub)
        src_sub2 = self.linear2(self.dropout(self.activation(self.linear1(src_sub2), inplace=True)))
        src_sub = src_sub + self.dropout2(src_sub2)

        src_obj2 = self.norm2(src_obj)
        src_obj2 = self.linear2(self.dropout(self.activation(self.linear1(src_obj2), inplace=True)))
        src_obj = src_obj + self.dropout2(src_obj2)

        src_verb2 = self.norm2(src_verb)
        src_verb2 = self.linear2(self.dropout(self.activation(self.linear1(src_verb2), inplace=True)))
        src_verb = src_verb + self.dropout2(src_verb2)

        return src_sub, src_obj, src_verb

    def forward(self, src_sub, src_obj, src_verb,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src_sub, src_obj, src_verb, src_mask,
                                    src_key_padding_mask, pos)
        return self.forward_post(src_sub, src_obj, src_verb, src_mask,
                                 src_key_padding_mask, pos)
