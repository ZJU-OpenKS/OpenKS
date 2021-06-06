import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class ActionDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_h = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_o = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mem_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # hidden_dim = d_model
        # self.human_query = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        # self.object_query = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        # self.act_query = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # memory projection
        self.proj_h1 = nn.Linear(d_model, dim_feedforward)
        self.proj_h2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_h = nn.Dropout(dropout)
        self.dropout_h2 = nn.Dropout(dropout)
        self.norm_h = nn.LayerNorm(d_model)

        self.proj_o1 = nn.Linear(d_model, dim_feedforward)
        self.proj_o2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_o = nn.Dropout(dropout)
        self.dropout_o2 = nn.Dropout(dropout)
        self.norm_o = nn.LayerNorm(d_model)

        self.proj_v1 = nn.Linear(d_model, dim_feedforward)
        self.proj_v2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_v = nn.Dropout(dropout)
        self.dropout_v2 = nn.Dropout(dropout)
        self.norm_v = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgts, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # mem_h = self.human_query(memory)
        # mem_o = self.object_query(memory)
        # mem_v = self.act_query(memory)

        # memory emd
        memory_h = self.proj_h2(self.dropout_h(self.activation(self.proj_h1(memory))))
        memory_h = memory + self.dropout_h2(memory_h)
        mem_h = self.norm_h(memory_h)

        memory_o = self.proj_o2(self.dropout_o(self.activation(self.proj_o1(memory))))
        memory_o = memory + self.dropout_o2(memory_o)
        mem_o = self.norm_o(memory_o)

        memory_v = self.proj_v2(self.dropout_v(self.activation(self.proj_v1(memory))))
        memory_v = memory + self.dropout_v2(memory_v)
        mem_v = self.norm_v(memory_v)

        # shared att for hov branch
        q_mh = k_mh = self.with_pos_embed(mem_h, pos)
        q_mo = k_mo = self.with_pos_embed(mem_o, pos)
        q_mv = k_mv = self.with_pos_embed(mem_v, pos)
        memory_h = self.mem_attn(q_mh, k_mh, value=mem_h, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)[0]
        memory_o = self.mem_attn(q_mo, k_mo, value=mem_o, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)[0]
        memory_v = self.mem_attn(q_mv, k_mv, value=mem_v, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)[0]
        memory_h = mem_h + self.dropout1(memory_h)
        memory_o = mem_o + self.dropout1(memory_o)
        memory_v = mem_v + self.dropout1(memory_v)
        memory_h = self.norm1(memory_h)
        memory_o = self.norm1(memory_o)
        memory_v = self.norm1(memory_v)

        tgt_hs = []
        tgt_os = []
        tgt_vs = []
        for tgt in tgts:
            tgt = tgt.transpose(0, 1)
            tgth2 = self.multihead_attn_h(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory_h, pos),
                                       value=memory_h, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgto2 = self.multihead_attn_o(query=self.with_pos_embed(tgt, query_pos),
                                        key=self.with_pos_embed(memory_o, pos),
                                        value=memory_o, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
            tgtv2 = self.multihead_attn_v(query=self.with_pos_embed(tgt, query_pos),
                                        key=self.with_pos_embed(memory_v, pos),
                                        value=memory_v, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]

            tgt_h = tgt + self.dropout2(tgth2)
            tgt_h = self.norm2(tgt_h)
            tgt_o = tgt + self.dropout2(tgto2)
            tgt_o = self.norm2(tgt_o)
            tgt_v = tgt + self.dropout2(tgtv2)
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

            tgt_hs.append(tgt_h)
            tgt_os.append(tgt_o)
            tgt_vs.append(tgt_v)

        tgt_hs = torch.stack(tgt_hs).transpose(1, 2)
        tgt_os = torch.stack(tgt_os).transpose(1, 2)
        tgt_vs = torch.stack(tgt_vs).transpose(1, 2)
        return tgt_hs, tgt_os, tgt_vs

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
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

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

'''
class ActionDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mem_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model*2, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        #
        # # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        # # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)
        #
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
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt3 = torch.cat([tgt, tgt2], dim=-1)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt3))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt2)

        # tgt = torch.cat([tgt, tgt2], dim=-1)

        # tgt = F.normalize(tgt, dim=-1)

        # tgt2 = self.multihead_attn(query=tgt,
        #                            key=memory,
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
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
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
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
'''
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
