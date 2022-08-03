# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import math
import copy
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..transformer.attention import MultiHeadAttention as MyMultiHeadAttention


# copy from proposal codes TODO!
def decode_scores_boxes(output_dict, end_points, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias=False, quality_channel=False):
    pred_boxes = output_dict['pred_boxes']
    batch_size = pred_boxes.shape[0]
    num_proposal = pred_boxes.shape[1]
    bbox_args_shape = 3+num_heading_bin*2+num_size_cluster*4
    if quality_channel:
        bbox_args_shape += 1
    assert pred_boxes.shape[-1] == bbox_args_shape, 'pred_boxes.shape wrong'

    if center_with_bias:
        # print('CENTER ADDING VOTE-XYZ', flush=True)
        # print('Using Center With Bias', output_dict.keys())
        if 'transformer_weighted_xyz' in output_dict.keys():
            end_points['transformer_weighted_xyz_all'] = output_dict['transformer_weighted_xyz_all']  # just for visualization
            transformer_xyz = output_dict['transformer_weighted_xyz']
            # print(transformer_xyz[0, :4], base_xyz[0, :4], 'from vote helper', flush=True)
            # print(center.shape, transformer_xyz.shape)
            transformer_xyz = nn.functional.pad(transformer_xyz, (0, 3+num_heading_bin*2+num_size_cluster*4-transformer_xyz.shape[-1]))
            pred_boxes = pred_boxes + transformer_xyz  # residual
        else:
            raise NotImplementedError('You should add it to the transformer final xyz')
            base_xyz = nn.functional.pad(base_xyz, (0, num_heading_bin*2+num_size_cluster*4))
            pred_boxes = pred_boxes + base_xyz  # residual
    else:
        raise NotImplementedError('center without bias(for decoder): not Implemented')

    center = pred_boxes[:,:,0:3] # (batch_size, num_proposal, 3) TODO RESIDUAL
    end_points['center'] = center

    heading_scores = pred_boxes[:,:,3:3+num_heading_bin]  # theta; todo change it
    heading_residuals_normalized = pred_boxes[:,:,3+num_heading_bin:3+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = pred_boxes[:,:,3+num_heading_bin*2:3+num_heading_bin*2+num_size_cluster]
    # Bxnum_proposalxnum_size_clusterx3 TODO NEXT WORK REMOVE BBOX-SIZE-DEFINED
    size_residuals_normalized = pred_boxes[:,:,3+num_heading_bin*2+num_size_cluster:3+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3])

    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    mean_size = torch.from_numpy(mean_size_arr.astype(np.float32)).type_as(pred_boxes).unsqueeze(0).unsqueeze(0)
    end_points['size_residuals'] = size_residuals_normalized * mean_size
    # print(3+num_heading_bin*2+num_size_cluster*4, ' <<< bbox heading and size tensor shape')
    return end_points


class Transformer3D(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0,
                 activation="gelu", normalize_before=False,
                 return_intermediate_dec=False, have_encoder=True, have_decoder=True, attention_type='default', deformable_type=None, offset_size=3):
        super().__init__()

        self.have_encoder = have_encoder
        assert not have_encoder
        self.have_decoder = have_decoder
        if have_decoder:
            print('[Attention:] The Transformer Model Have Decoder Module')
            self.offset_size = offset_size
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, attention_type=attention_type, deformable_type=deformable_type, offset_size=offset_size)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)
            self.attention_type = attention_type

        # self._reset_parameters()  # for fc
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # TODO ADD STATIC_FEAT(WEIGHTED SUM or fc)
    def forward(self, src, mask, query_embed, pos_embed, static_feat=None, src_mask=None, src_position=None, seed_position=None, seed_feat=None, seed_embed=None, decode_vars=None):
        # flatten BxNxC to NxBxC
        B, N, C = src.shape
        src = src.permute(1, 0, 2)
        if pos_embed is not None:
            pos_embed = pos_embed.permute(1, 0, 2)

        if seed_feat is None:
            memory = src
        else:
            memory = seed_feat.permute(1, 0, 2)
        # print('encoder done ???')
        if not self.have_decoder:  # TODO LOCAL ATTENTION
            return memory.permute(1, 0, 2)  # just return it

        # to get decode layer TODO
        if self.attention_type.split(';')[-1] == 'deformable':
            assert query_embed is None, 'deformable: query embedding should be None'
            query_embed = torch.zeros_like(src)
            tgt = src
            tgt_mask = src_mask
        else:  # just Add It
            raise NotImplementedError('Not Deformable')

        if src_position is not None:
            src_position = src_position.permute(1, 0, 2)
        if seed_position is not None:
            seed_position = seed_position.permute(1, 0, 2)
            tgt_position = nn.functional.pad(src_position, (0, self.offset_size-3))
            pos_embed = seed_embed
            if pos_embed is not None:
                pos_embed = pos_embed.permute(1, 0, 2)
        else:
            tgt_position = src_position
            seed_position = src_position

        # self-attention:  tgt -> tgt
        # cross-attention: src/memory -> tgt
        decoder_output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mask,
                                      pos=pos_embed, query_pos=query_embed, src_position=seed_position, tgt_position=tgt_position, decode_vars=decode_vars)
        # print(hs.transpose(1,2).shape, memory.shape, '<< final encoder and decode shape', flush=True)

        if src_position is not None:
            hs, finpos = decoder_output
            # print(hs.shape, memory.shape, finpos.shape, '<<< fin pos shape', flush=True)
            # print((finpos[-1] - src_position).max(), '  <<<  finpos shift', flush=True)
            return hs.transpose(1, 2), memory.permute(1, 0, 2), finpos.transpose(1, 2) # .view(B, N, C)
        else:
            hs = decoder_output
        return hs.transpose(1, 2), memory.permute(1, 0, 2)  # .view(B, N, C)


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
            # print(output, '<< ENCODER output layer??')

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
                query_pos: Optional[Tensor] = None,
                src_position: Optional[Tensor] = None,
                tgt_position: Optional[Tensor] = None,
                decode_vars: Optional = None):
        output = tgt

        intermediate, intermediate_pos = [], []

        for layer in self.layers:
            output, nxt_position = layer(output, memory, tgt_mask=tgt_mask,
                                         memory_mask=memory_mask,
                                         tgt_key_padding_mask=tgt_key_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask,
                                         pos=pos, query_pos=query_pos, src_position=src_position, tgt_position=tgt_position, decode_vars=decode_vars)
            # print((tgt_position-nxt_position).abs().max(), '<< xyz, bias, from transformer')
            # print(output.shape, '<< output shape', tgt_position.shape, '<< tgt shape', flush=True)
            tgt_position = nxt_position
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_pos.append(tgt_position)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_pos)

        return output.unsqueeze(0), tgt_position.unsqueeze(0)


def attn_with_batch_mask(layer_attn, q, k, src, src_mask, src_key_padding_mask):
    bs, src_arr, attn_arr = q.shape[1], [], []
    for i in range(bs):
        key_mask, attn_mask = None, None
        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask[i:i+1]
        if src_mask is not None:
            attn_mask = src_mask[i]
        batch_attn = layer_attn(q[:, i:i+1, :], k[:, i:i+1, :], value=src[:, i:i+1, :], attn_mask=attn_mask,
                                key_padding_mask=key_mask)
        # print(batch_attn[1].sum(dim=-1))  # TODO it is okay to make a weighted sum
        # print(batch_attn[1], attn_mask, flush=True
        src_arr.append(batch_attn[0])
        attn_arr.append(batch_attn[1])
    src2 = torch.cat(src_arr, dim=1)
    attn = torch.cat(attn_arr, dim=0)
    return src2, attn


class MultiheadPositionalAttention(nn.Module):  # nearby points
    def __init__(self, d_model, nhead, dropout, attn_type='nearby'):  # nearby; interpolation
        super().__init__()
        assert attn_type in ['nearby', 'nearby_20','interpolation', 'interpolation_10', 'interpolation_20', 'dist', 'dist_10',
                             'input', 'multiply', 'multiply_20', 'multiply_all', 'myAdd', 'myAdd_20', 'myAdd_all', 'myAdd_5_faster']
        self.attn_type = attn_type
        self.nhead = nhead
        if 'multiply' in self.attn_type or 'myAdd' in self.attn_type:
            self.attention = MyMultiHeadAttention(d_model, d_k=d_model//nhead, d_v=d_model//nhead, h=nhead, dropout=dropout)
        else:
            self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    @staticmethod
    def rotz_batch_pytorch(t):
        """
        Rotation about the z-axis
        :param t: (x1,x2,...,xn)
        :return: output:(x1,x2,...,xn,3,3)
        """
        input_shape = t.shape  # (B, num_proposal)
        output = torch.zeros(tuple(list(input_shape)+[3,3])).type_as(t)
        c = torch.cos(t)
        s = torch.sin(t)
        # Attention ~ 这里的rot_mat是已经转置过的matrix，是为了进行 x'A' = (Ax)'
        # [[cos(t), -sin(t), 0],
        #  [sin(t), cos(t),   0],
        #  [0,     0,        1]]
        output[...,0,0] = c
        output[...,0,1] = -s
        output[...,1,0] = s
        output[...,1,1] = c
        output[...,2,2] = 1
        return output

    def forward(self, query, key, value, attn_mask, key_padding_mask, src_position, tgt_position, decode_vars=None):  # TODO Check Decode Vars
        if self.attn_type in ['input']: # just using attn_mask from input
            return attn_with_batch_mask(self.attention, q=query, k=key, src=value, src_mask=attn_mask,
                                        src_key_padding_mask=key_padding_mask)
        # print(query.shape, key.shape, value.shape, '<< cross attn shape', flush=True)
        N, B, C = src_position.shape
        N2, B2, C2 = tgt_position.shape
        # use just xyz
        assert C == 3 and C2 == 3
        if C != 3 and C2 != 3 and C == C2:
            C2 = C = 3  # only xyz is useful
            src_position = src_position[:, :, :3]
            tgt_position = tgt_position[:, :, :3]
        # Using Just XYZ
        assert B2 == B and C2 == C
        Y = src_position[:, None, :, :].repeat(1, N2, 1, 1)
        X = tgt_position[None, :, :, :].repeat(N, 1, 1, 1)
        dist = torch.sum((X - Y).pow(2), dim=-1)
        dist = dist.permute(2, 0, 1)

        if self.attn_type in ['multiply', 'multiply_20', 'multiply_all', 'myAdd', 'myAdd_20', 'myAdd_all', 'myAdd_5_faster']:  # similiar as pointnet
            assert attn_mask is None, 'positional attn: mask should be none'
            near_kth = 5
            kth_split = self.attn_type.split('_')
            if len(kth_split) != 1:  # not default
                if kth_split[1] != 'all':
                    near_kth = int(kth_split[1])
                else:
                    near_kth = dist.shape[-1]
            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=-1, largest=False, sorted=False)
            # weight
            dist_min = dist_min.sqrt()
            dist_recip = 1 / (dist_min + 1e-1)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # print(dist_min.max(), dist_recip.max(), norm.max())
            # print(dist_min.min(), dist_recip.min(), norm.min())
            weight = (dist_recip / norm).detach()  # B * N * near_kth ; grad notuseful
            # src_mask
            # print(query.shape, key.shape, value.shape, '<< attention shape', src_mask.shape, '<< mask', flush=True)
            if 'myAdd' in self.attn_type:
                if 'faster' in self.attn_type:
                    ret = self.attention.forward_faster(queries=query.permute(1, 0, 2), keys=key.permute(1, 0, 2), values=value.permute(1, 0, 2),
                                                        attention_pos=dist_pos, attention_weights=weight, way='add')
                else:
                    src_mask = torch.zeros(dist.shape).to(dist.device) - 1e9
                    src_mask.scatter_(2, dist_pos, weight)
                    src_mask = src_mask.permute(0, 2, 1) # V*k'*q
                    ret = self.attention(queries=query.permute(1, 0 ,2), keys=key.permute(1, 0, 2), values=value.permute(1, 0, 2),
                                         attention_weights=src_mask[:, None, :, :], way='add')
            else:
                src_mask = torch.zeros(dist.shape).to(dist.device)
                src_mask.scatter_(2, dist_pos, weight)
                src_mask = src_mask.permute(0, 2, 1) # V*k'*q
                ret = self.attention(queries=query.permute(1, 0 ,2), keys=key.permute(1, 0, 2), values=value.permute(1, 0, 2),
                                     attention_weights=src_mask[:, None, :, :], way='mul')
            ret = ret.permute(1, 0, 2)
            return [ret, None]
        else:
            raise NotImplementedError(self.attn_type)

mark = False
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, attention_type='default', deformable_type=None, offset_size=3):
        super().__init__()
        attn_split = attention_type.split(';')
        if len(attn_split) == 1:
            attention_input = 'input'
        else:
            attention_input = attn_split[0]
            print('Attention input type', attention_input)
            assert len(attn_split) == 2, 'len(attention_type) should be 1 or 2'
        attention_type = attn_split[-1]
        self.attention_type = attention_type
        print('transformer: Using Decoder transformer type', attention_input, attention_type)
        if attention_type == 'default':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, attn_type=attention_input)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        elif attn_split[-1] == 'deformable':
            if offset_size != 3:
                self.linear_offset = MLP(d_model, d_model, offset_size, 3, norm=nn.LayerNorm)
            else:
                self.linear_offset = nn.Linear(d_model, offset_size)  # center forward
                self.linear_offset.weight.data.zero_()
                self.linear_offset.bias.data.zero_()
            # print(self.linear_offset.weight.data.max(), '<< linear OFFSET WIEGHT  !')
            assert deformable_type is not None
            src_attn_type = deformable_type
            self.self_attn = MultiheadPositionalAttention(d_model, nhead, dropout=dropout, attn_type=attention_input)
            self.multihead_attn = MultiheadPositionalAttention(d_model, nhead, dropout=dropout, attn_type=src_attn_type)
        else:
            raise NotImplementedError(attention_type)
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
                     query_pos: Optional[Tensor] = None,
                     src_position: Optional[Tensor] = None,
                     tgt_position: Optional[Tensor] = None,
                     decode_vars: Optional = None):
        if self.attention_type == 'default':
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        elif self.attention_type.split(';')[-1] == 'deformable':
            q = k = self.with_pos_embed(tgt, query_pos)
            attn = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask,
                                  src_position=tgt_position,
                                  tgt_position=tgt_position)
            tgt2 = attn[0]
            if len(attn) == 3:
                # print('attn from output! TODO')
                tgt_position = attn[2]
            else:
                assert len(attn) == 2, 'attn len should not be 2 or 3'
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # TODO src_position_attention checking
            offset = self.linear_offset(tgt)
            # print(offset.shape, ' <<< offset')
            # print(offset[:5, 1, :6], '<< offset', flush=True)
            # print(self.linear_offset.weight.data.max(), self.linear_offset.bias.data.max(), ' << linear_offset shape max')
            # print(offset.shape, tgt_position.shape, offset.max(), '<< offset shape', flush=True)
            tgt_position = tgt_position + offset
            attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask,
                                       src_position=src_position,
                                       tgt_position=tgt_position,
                                       decode_vars=decode_vars)
            tgt2 = attn[0]
            # print(tgt2, '<< tgt2')
            if len(attn) == 3:
                # print('attn from input! TODO')
                tgt_position = attn[2]
            else:
                assert len(attn) == 2, 'attn len should not be 2 or 3'
        else:
            raise NotImplementedError(self.attention_type)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, tgt_position


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                src_position: Optional[Tensor] = None,
                tgt_position: Optional[Tensor] = None,
                decode_vars: Optional = None):

        if self.normalize_before:
            raise NotImplementedError('todo: detr - decoder - normalize_before (wrong when normalize_before_with_tgt_position_encoding)')
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, src_position, tgt_position, decode_vars)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    transformer_type = args.get('transformer_type', 'enc_dec')
    print('[build transformer] Using transformer type', transformer_type)
    print(args, '<< transformer config')
    if transformer_type == 'enc_dec':
        return Transformer3D(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif transformer_type == 'enc':
        return Transformer3D(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=False,
            have_decoder=False,
        )
    elif transformer_type.split(';')[-1] == 'deformable':
        return Transformer3D(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            have_encoder=False,
            have_decoder=True,  # using input position
            attention_type=transformer_type,
            deformable_type=args.get('deformable_type','nearby'),
            offset_size=args.get('offset_size', 3)
        )
    else:
        raise NotImplementedError(transformer_type)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    print(activation, '<< transformer activation', flush=True)  # TODO REMOVE IT
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
        return gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if norm is not None:
            print('Using Norm << MLP', flush=True)
            self.norm = nn.ModuleList(norm(hidden_dim) for i in range(num_layers-1))
        else:
            self.norm = None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.norm is not None:
                x = F.relu(self.norm[i](layer(x))) if i < self.num_layers - 1 else layer(x)
            else:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == '__main__':
    from thop import profile
    model = TransformerDecoderLayer(d_model=288, nhead=4, dropout=0.1, dim_feedforward=2048, activation='gelu', attention_type='myAdd;deformable', deformable_type='myAdd')
    tgt = torch.randn(256, 1, 256)
    memory = torch.randn(256, 1, 288)
    query_pos = torch.randn(256, 1, 288)
    tgt_position = torch.randn(256, 1, 3)
    src_position = torch.randn(256, 1, 3)
    # profile(model, (tgt=tgt, memory=memory, query_pos=query_pos, tgt_position=tgt_position, src_position=src_position))
    profile(model, (tgt, None, memory, None, None, None, query_pos, src_position, tgt_position))
