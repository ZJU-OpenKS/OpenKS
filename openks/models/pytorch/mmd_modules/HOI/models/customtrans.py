import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from .transformer import _get_clones, _get_activation_fn
import pdb


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class CustomTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

#         neck_encoder_layer = NeckEncoderLayer(d_model, nhead, dim_feedforward,
#                                               dropout, activation, normalize_before)
#         self.neck_encoder = NeckEncoder(neck_encoder_layer, 1, encoder_norm)

        # self.h_proj = nn.Linear(d_model, d_model)
        # self.o_proj = nn.Linear(d_model, d_model)
        # self.v_proj = nn.Linear(d_model, d_model)

        decoder_layer = NeckDecoderLayer(d_model, nhead, dim_feedforward,
                                         dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = NeckDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                   return_intermediate=return_intermediate_dec)

        # before decoder, we need a projection to embedding query into different space...
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # memory projection
        # self.proj_h1 = nn.Linear(d_model, dim_feedforward)
        # self.proj_h2 = nn.Linear(dim_feedforward, d_model)
        # self.dropout_h = nn.Dropout(dropout)
        # self.dropout_h2 = nn.Dropout(dropout)
        # self.norm_h = nn.LayerNorm(d_model)
        # 
        # self.proj_o1 = nn.Linear(d_model, dim_feedforward)
        # self.proj_o2 = nn.Linear(dim_feedforward, d_model)
        # self.dropout_o = nn.Dropout(dropout)
        # self.dropout_o2 = nn.Dropout(dropout)
        # self.norm_o = nn.LayerNorm(d_model)

        self.proj_ho1 = nn.Linear(d_model, dim_feedforward)
        self.proj_ho2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_ho = nn.Dropout(dropout)
        self.dropout_ho2 = nn.Dropout(dropout)
        self.norm_ho = nn.LayerNorm(d_model)

        self.proj_v1 = nn.Linear(d_model, dim_feedforward)
        self.proj_v2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_v = nn.Dropout(dropout)
        self.dropout_v2 = nn.Dropout(dropout)
        self.norm_v = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed_h, query_embed_o, query_embed_v, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_v = query_embed_v.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt_h = torch.zeros_like(query_embed_h)
        tgt_o = torch.zeros_like(query_embed_o)
        tgt_v = torch.zeros_like(query_embed_v)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # memory emd
        # memory_h = self.proj_h2(self.dropout_h(self.activation(self.proj_h1(memory))))
        # memory_h = memory + self.dropout_h2(memory_h)
        # memory_h = self.norm_h(memory_h)
        # 
        # memory_o = self.proj_o2(self.dropout_o(self.activation(self.proj_o1(memory))))
        # memory_o = memory + self.dropout_o2(memory_o)
        # memory_o = self.norm_o(memory_o)

        memory_ho = self.proj_ho2(self.dropout_ho(self.activation(self.proj_ho1(memory))))
        memory_ho = memory + self.dropout_ho2(memory_ho)
        memory_ho = self.norm_ho(memory_ho)

        memory_v = self.proj_v2(self.dropout_v(self.activation(self.proj_v1(memory))))
        memory_v = memory + self.dropout_v2(memory_v)
        memory_v = self.norm_v(memory_v)

#         memory_h, memory_o, memory_v = self.neck_encoder(memory, src_key_padding_mask=mask, pos=pos_embed)

        # memory_h = self.h_proj(memory)
        # memory_o = self.o_proj(memory)
        # memory_v = self.v_proj(memory)

        hs_h, hs_o, hs_v = self.decoder(tgt_h, tgt_o, tgt_v,
                                        memory_ho, memory_v,
                                        memory_key_padding_mask=mask,
                                        pos=pos_embed,
                                        human_query_pos=query_embed_h,
                                        object_query_pos=query_embed_o,
                                        verb_query_pos=query_embed_v)
        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return (hs_h.transpose(1, 2),
                hs_o.transpose(1, 2),
                hs_v.transpose(1, 2),
                memory.permute(1, 2, 0).view(bs, c, h, w))


class NeckEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output_h = output_o = output_v = src

        for layer in self.layers:
            output_h, output_o, output_v = layer(output_h, output_o, output_v, src_mask=mask,
                                                 src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output_h, output_o, output_v = self.norm(output_h), self.norm(output_o), self.norm(output_v)

        return output_h, output_o, output_v


class NeckDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt_h, tgt_o, tgt_v,
                memory_ho, memory_v,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                human_query_pos: Optional[Tensor] = None,
                object_query_pos: Optional[Tensor] = None,
                verb_query_pos: Optional[Tensor] = None
                ):
        # pdb.set_trace()
        output_h, output_o, output_v = tgt_h, tgt_o, tgt_v

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []

        for layer in self.layers:
            output_h, output_o, output_v = layer(output_h, output_o, output_v,
                                                 memory_ho, memory_v,
                                                 tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask,
                                                 pos=pos,
                                                 human_query_pos=human_query_pos,
                                                 object_query_pos=object_query_pos,
                                                 verb_query_pos=verb_query_pos)
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_h))
                intermediate_o.append(self.norm(output_o))
                intermediate_v.append(self.norm(output_v))

            if self.norm is not None:
                # raise Exception("I'm lazy and don't wanna implement this...")
                # $20 is $20...
                output_h = self.norm(output_h)
                output_o = self.norm(output_o)
                output_v = self.norm(output_v)

                if self.return_intermediate:
                    intermediate_h.pop()
                    intermediate_o.pop()
                    intermediate_v.pop()
                    intermediate_h.append(output_h)
                    intermediate_o.append(output_o)
                    intermediate_v.append(output_v)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_h, output_o, output_v


class NeckEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # MHA v->ho
        self.v2ho_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 3 projection~
        self.h_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Implementation of shared Feedforward model
        self.linear1_sh = nn.Linear(d_model, dim_feedforward)
        self.dropout_sh = nn.Dropout(dropout)
        self.linear2_sh = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)

        self.norm_ho = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src_h,
                     src_o,
                     src_v,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # first do three projection to project global context into three branch
        h = self.h_proj(src_h)
        o = self.o_proj(src_o)
        v = self.v_proj(src_v)

        q_h = k_h = self.with_pos_embed(h, pos)
        q_o = k_o = self.with_pos_embed(o, pos)
        q_v = k_v = self.with_pos_embed(v, pos)

        src_h = self.self_attn(q_h, k_h, value=h, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        src_o = self.self_attn(q_o, k_o, value=o, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        src_v = self.self_attn(q_v, k_v, value=v, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]

        src_h = h + self.dropout1(src_h)
        src_o = o + self.dropout1(src_o)
        src_v = v + self.dropout1(src_v)

        src_h = self.norm1(src_h)
        src_o = self.norm1(src_o)
        src_v = self.norm1(src_v)

        # src_ho = self.norm_ho(torch.cat((src_h, src_o), dim=-1))
        src_ho = src_h + src_o
        src_ho = self.norm_ho(src_ho)
        src_ho_comb = self.linear2(self.dropout(self.activation(self.linear1(src_ho))))
        src_ho = src_ho + self.dropout2(src_ho_comb)
        src_ho = self.norm2(src_ho)

        # v->ho
        # q = k = self.with_pos_embed(src_ho, pos)
        # q = self.with_pos_embed(src_ho, pos)
        # Key should be src_v
        # k = self.with_pos_embed(src_v, pos)
        # correct v->ho
        q = self.with_pos_embed(src_v, pos)
        k = self.with_pos_embed(src_ho, pos)
        src_v2 = self.v2ho_attn(q, k, value=src_v, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src_v = src_v + self.dropout3(src_v2)
        src_v = self.norm3(src_v)

        # shared FFN
        src_v2 = self.linear2_sh(self.dropout_sh(self.activation(self.linear1_sh(src_v))))
        src_v = src_v + self.dropout4(src_v2)
        src_v = self.norm4(src_v)

        src_o2 = self.linear2_sh(self.dropout_sh(self.activation(self.linear1_sh(src_o))))
        src_o = src_o + self.dropout4(src_o2)
        src_o = self.norm4(src_o)

        src_h2 = self.linear2_sh(self.dropout_sh(self.activation(self.linear1_sh(src_h))))
        src_h = src_h + self.dropout4(src_h2)
        src_h = self.norm4(src_h)
        return src_h, src_o, src_v

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
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src_h,
                src_o,
                src_v,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            raise Exception("No Implementation!!!")
            # return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src_h, src_o, src_v, src_mask, src_key_padding_mask, pos)


class NeckDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn_h = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn_o = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_ho = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt_ho, tgt_v,
                     memory_ho, memory_v,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     human_query_pos: Optional[Tensor] = None,
                     object_query_pos: Optional[Tensor] = None,
                     verb_query_pos: Optional[Tensor] = None
                     ):
        q_h = k_h = self.with_pos_embed(tgt_h, human_query_pos)
        q_o = k_o = self.with_pos_embed(tgt_o, object_query_pos)
        q_v = k_v = self.with_pos_embed(tgt_v, verb_query_pos)

        # # memory emd
        # memory_h = self.proj_h2(self.dropout_h(self.activation(self.proj_h1(memory))))
        # memory_h = memory + self.dropout_h2(memory_h)
        # memory_h = self.norm_h(memory_h)
        #
        # memory_o = self.proj_o2(self.dropout_o(self.activation(self.proj_o1(memory))))
        # memory_o = memory + self.dropout_o2(memory_o)
        # memory_o = self.norm_o(memory_o)
        #
        # memory_v = self.proj_v2(self.dropout_v(self.activation(self.proj_v1(memory))))
        # memory_v = memory + self.dropout_v2(memory_v)
        # memory_v = self.norm_v(memory_v)

        # shared self attention
        tgt_h2 = self.self_attn(q_h, k_h, value=tgt_h, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt_o2 = self.self_attn(q_o, k_o, value=tgt_o, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt_v2 = self.self_attn(q_v, k_v, value=tgt_v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]

        tgt_h = tgt_h + self.dropout1(tgt_h2)
        tgt_h = self.norm1(tgt_h)
        tgt_o = tgt_o + self.dropout1(tgt_o2)
        tgt_o = self.norm1(tgt_o)
        tgt_v = tgt_v + self.dropout1(tgt_v2)
        tgt_v = self.norm1(tgt_v)

        # Cross Att for each branch
        tgt_h2 = self.multihead_attn_h(query=self.with_pos_embed(tgt_h, human_query_pos),
                                       key=self.with_pos_embed(memory_h, pos),
                                       value=memory_h, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt_o2 = self.multihead_attn_o(query=self.with_pos_embed(tgt_o, object_query_pos),
                                       key=self.with_pos_embed(memory_o, pos),
                                       value=memory_o, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt_v2 = self.multihead_attn_v(query=self.with_pos_embed(tgt_v, verb_query_pos),
                                       key=self.with_pos_embed(memory_v, pos),
                                       value=memory_v, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]

        tgt_h = tgt_h + self.dropout2(tgt_h2)
        tgt_h = self.norm2(tgt_h)
        tgt_o = tgt_o + self.dropout2(tgt_o2)
        tgt_o = self.norm2(tgt_o)
        tgt_v = tgt_v + self.dropout2(tgt_v2)
        tgt_v = self.norm2(tgt_v)

        tgt_h2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_h))))
        tgt_h = tgt_h + self.dropout3(tgt_h2)
        tgt_h = self.norm3(tgt_h)
        tgt_o2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_o))))
        tgt_o = tgt_o + self.dropout3(tgt_o2)
        tgt_o = self.norm3(tgt_o)
        tgt_v2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_v))))
        tgt_v = tgt_v + self.dropout3(tgt_v2)
        tgt_v = self.norm3(tgt_v)
        return tgt_h, tgt_o, tgt_v

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
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt_h, tgt_o, tgt_v,
                memory_ho, memory_v,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                human_query_pos: Optional[Tensor] = None,
                object_query_pos: Optional[Tensor] = None,
                verb_query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            raise Exception("No Implementation!!!")
            # return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
            #                         tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt_h, tgt_o, tgt_v,
                                 memory_ho, memory_v,
                                 tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos,
                                 human_query_pos, object_query_pos, verb_query_pos)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
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
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
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
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
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
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
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
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
