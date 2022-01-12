from typing import Optional

import torch
from torch import nn, Tensor

from .tr_helper import get_clones
from .enc_helper import encoder_layer_builder
from .dec_helper import decoder_layer_builder


class TransformerHOI(nn.Module):

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 nhead_hoi=4,
                 dropout=0.1,
                 dim_vanilla_ffn=2048,
                 dim_hoi_ffn=256,
                 activation="relu",
                 hoi_encoder_type='hoi',
                 vanilla_decoder_type='vanilla',
                 hoi_decoder_type='hoi',
                 num_vanilla_encoders=6,
                 num_hoi_encoders=3,
                 num_vanilla_decoders=6,
                 num_hoi_decoders=3,
                 normalize_before=False,
                 return_intermediate_dec=False,
                 split_query=False,
                 interact_query=False):
        super().__init__()

        vanilla_encoder_layer = encoder_layer_builder('vanilla')(
            d_model, nhead, dim_vanilla_ffn, dropout, activation, normalize_before)
        hoi_encoder_layer = encoder_layer_builder('hoi_bottleneck')(
            d_model, nhead_hoi, dim_hoi_ffn, dropout, activation, normalize_before)
        self.encoder = EncoderHOI(d_model, vanilla_encoder_layer, num_vanilla_encoders,
                                  hoi_encoder_layer, num_hoi_encoders, normalize_before)

        vanilla_decoder_layer = decoder_layer_builder(vanilla_decoder_type)(
            d_model, nhead, dim_vanilla_ffn, dropout, activation, normalize_before,
            split_query=split_query, interact_query=interact_query)
        hoi_decoder_layer = decoder_layer_builder(hoi_decoder_type)(
            d_model, nhead_hoi, dim_hoi_ffn, dropout, activation, normalize_before,
            split_query=split_query, interact_query=interact_query)
        self.decoder = DecoderHOI(d_model, vanilla_decoder_layer, num_vanilla_decoders,
                                  hoi_decoder_layer, num_hoi_decoders, normalize_before, 
                                  return_intermediate=return_intermediate_dec, split_query=split_query)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.split_query = split_query

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.encoder.eye_init()
        self.decoder.eye_init()

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if not self.split_query:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        else:
            for k, v in query_embed.items():
                query_embed[k] = v.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        if not self.split_query:
            tgt = torch.zeros_like(query_embed)
        else:
            tgt = torch.zeros_like(query_embed['h_query'])

        mem, mem_sub, mem_obj, mem_verb = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed)
        hs_sub, hs_obj, hs_verb = self.decoder(
            tgt, mem, mem_sub, mem_obj, mem_verb, pos=pos_embed,
            mem_key_padding_mask=mask, query_pos=query_embed)

        return (hs_sub.transpose(1, 2), hs_obj.transpose(1, 2), hs_verb.transpose(1, 2)), \
               (mem_sub, mem_obj, mem_verb)


class EncoderHOI(nn.Module):

    def __init__(self, d_model, vanilla_encoder_layer, num_vanilla_layers,
                 hoi_encoder_layer, num_hoi_layers, normalize_before=False):
        super().__init__()
        self.layers = get_clones(vanilla_encoder_layer, num_vanilla_layers)
        self.hoi_layers = get_clones(hoi_encoder_layer, num_hoi_layers)

        self.norm = self.norm_sub = self.norm_obj = self.norm_verb = None
        if normalize_before:
            self.norm = nn.LayerNorm(d_model)
            self.norm_sub = nn.LayerNorm(d_model)
            self.norm_obj = nn.LayerNorm(d_model)
            self.norm_verb = nn.LayerNorm(d_model)

    def eye_init(self):
        for layer in self.hoi_layers:
            layer.eye_init()

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, pos=pos,
                           src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        embed_sub = embed_obj = embed_verb = output
        for layer in self.hoi_layers:
            embed_sub, embed_obj, embed_verb = layer(
                embed_sub, embed_obj, embed_verb, src_mask=mask, pos=pos,
                src_key_padding_mask=src_key_padding_mask)
        if self.norm_sub is not None:
            embed_sub = self.norm_sub(embed_sub)
            embed_obj = self.norm_obj(embed_obj)
            embed_verb = self.norm_verb(embed_verb)

        return output, embed_sub, embed_obj, embed_verb


class DecoderHOI(nn.Module):

    def __init__(self, d_model, vanilla_decoder_layer, num_vanilla_layers, hoi_decoder_layer,
                 num_hoi_layers, normalize_before=False, return_intermediate=False, split_query=False):
        super().__init__()
        if not isinstance(hoi_decoder_layer, decoder_layer_builder('vanilla_bottleneck')):
            num_hoi_layers = num_hoi_layers * 3

        self.layers = get_clones(vanilla_decoder_layer, num_vanilla_layers)
        self.hoi_layers = get_clones(hoi_decoder_layer, num_hoi_layers)

        self.norm = self.norm_sub1 = self.norm_obj1 = self.norm_verb1 = None
        self.norm_sub2 = self.norm_obj2 = self.norm_verb2 = None
        if isinstance(vanilla_decoder_layer, decoder_layer_builder('vanilla')):
            self.norm = nn.LayerNorm(d_model)
        if normalize_before:
            self.norm_sub1 = nn.LayerNorm(d_model)
            self.norm_obj1 = nn.LayerNorm(d_model)
            self.norm_verb1 = nn.LayerNorm(d_model)
            self.norm_sub2 = nn.LayerNorm(d_model)
            self.norm_obj2 = nn.LayerNorm(d_model)
            self.norm_verb2 = nn.LayerNorm(d_model)

        self.return_intermediate = return_intermediate
        self.split_query = split_query

    def eye_init(self):
        pass

    def foward_vanilla(self, tgt, mem_sub, mem_obj, mem_verb,
                       tgt_mask: Optional[Tensor] = None,
                       mem_mask: Optional[Tensor] = None,
                       tgt_key_padding_mask: Optional[Tensor] = None,
                       mem_key_padding_mask: Optional[Tensor] = None,
                       pos: Optional[Tensor] = None,
                       query_pos: Optional[Tensor] = None):
        intermediate = []

        out_sub = out_obj = out_verb = tgt
        for layer_id, layer in enumerate(self.layers):
            if isinstance(layer, decoder_layer_builder('vanilla')):
                out_sub = layer(out_sub, mem_sub, tgt_mask=tgt_mask,
                                memory_mask=mem_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=mem_key_padding_mask,
                                pos=pos, query_pos=query_pos)
                out_obj = layer(out_obj, mem_obj, tgt_mask=tgt_mask,
                                memory_mask=mem_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=mem_key_padding_mask,
                                pos=pos, query_pos=query_pos)
                out_verb = layer(out_verb, mem_verb, tgt_mask=tgt_mask,
                                 memory_mask=mem_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=mem_key_padding_mask,
                                 pos=pos, query_pos=query_pos)
            else:
                out_sub, out_obj, out_verb = layer(
                    out_sub, out_obj, out_verb, mem_sub, mem_obj, mem_verb,
                    tgt_mask=tgt_mask, memory_mask=mem_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=mem_key_padding_mask, pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                if self.norm_sub1 is not None:
                    intermediate.append((
                        self.norm_sub1(out_sub), self.norm_obj1(out_obj), self.norm_verb1(out_verb)))
                else:
                    intermediate.append((out_sub, out_obj, out_verb))
        return (out_sub, out_obj, out_verb), intermediate

    def forward_hoi(self, out_sub, out_obj, out_verb, mem_sub, mem_obj, mem_verb,
                    tgt_mask: Optional[Tensor] = None,
                    mem_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    mem_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        intermediate = []
        if len(self.hoi_layers) <= 0:
            return (out_sub, out_obj, out_verb), intermediate

        if isinstance(self.hoi_layers[0], decoder_layer_builder('vanilla')):
            for dec_sub, dec_obj, dec_verb in zip(
                    self.hoi_layers[::3], self.hoi_layers[1::3], self.hoi_layers[2::3]):
                out_sub = dec_sub(out_sub, mem_sub, tgt_mask=tgt_mask,
                                  memory_mask=mem_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=mem_key_padding_mask,
                                  pos=pos, query_pos=query_pos)
                out_obj = dec_obj(out_obj, mem_obj, tgt_mask=tgt_mask,
                                  memory_mask=mem_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=mem_key_padding_mask,
                                  pos=pos, query_pos=query_pos)
                out_verb = dec_verb(out_verb, mem_verb, tgt_mask=tgt_mask,
                                    memory_mask=mem_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=mem_key_padding_mask,
                                    pos=pos, query_pos=query_pos)
                if self.return_intermediate:
                    if self.norm_sub2 is not None:
                        intermediate.append((
                            self.norm_sub2(out_sub), self.norm_obj2(out_obj), self.norm_verb2(out_verb)))
                    else:
                        intermediate.append((out_sub, out_obj, out_verb))
        else:
            for layer in self.hoi_layers:
                out_sub, out_obj, out_verb = layer(
                    out_sub, out_obj, out_verb, mem_sub, mem_obj, mem_verb,
                    tgt_mask=tgt_mask, memory_mask=mem_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=mem_key_padding_mask, pos=pos, query_pos=query_pos)
                if self.return_intermediate:
                    if self.norm_sub2 is not None:
                        intermediate.append((
                            self.norm_sub2(out_sub), self.norm_obj2(out_obj), self.norm_verb2(out_verb)))
                    else:
                        intermediate.append((out_sub, out_obj, out_verb))
        return (out_sub, out_obj, out_verb), intermediate

    def forward(self, tgt, mem, mem_sub, mem_obj, mem_verb,
                tgt_mask: Optional[Tensor] = None,
                mem_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                mem_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        (out_sub, out_obj, out_verb), intermediate = self.foward_vanilla(
            tgt, mem_sub, mem_obj, mem_verb, tgt_mask, mem_mask,
            tgt_key_padding_mask, mem_key_padding_mask, pos, query_pos)
        del mem

        (out_sub, out_obj, out_verb), intermediate_hoi = self.forward_hoi(
            out_sub, out_obj, out_verb, mem_sub, mem_obj, mem_verb, tgt_mask,
            mem_mask, tgt_key_padding_mask, mem_key_padding_mask, pos, query_pos)
        intermediate.extend(intermediate_hoi)
        del intermediate_hoi

        if self.norm_sub2 is not None:
            out_sub = self.norm_sub2(out_sub)
            out_obj = self.norm_obj2(out_obj)
            out_verb = self.norm_verb2(out_verb)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append((out_sub, out_obj, out_verb))

        if self.return_intermediate:
            out_sub = torch.stack([outs[0] for outs in intermediate])
            out_obj = torch.stack([outs[1] for outs in intermediate])
            out_verb = torch.stack([outs[2] for outs in intermediate])

        return out_sub, out_obj, out_verb


def build_transformer(args):
    if args.hoi:
        return TransformerHOI(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            nhead_hoi=args.nheads_hoi,
            dim_vanilla_ffn=args.dim_feedforward,
            dim_hoi_ffn=args.dim_feedforward_hoi,
            num_vanilla_encoders=args.enc_layers,
            num_hoi_encoders=args.hoi_enc_layers,
            num_vanilla_decoders=args.dec_layers,
            num_hoi_decoders=args.hoi_dec_layers,
            hoi_encoder_type=args.hoi_enc_type,
            vanilla_decoder_type=args.vanilla_dec_type,
            hoi_decoder_type=args.hoi_dec_type,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            split_query=args.split_query,
            interact_query=args.interact_query)

    raise ValueError('not implement!')
