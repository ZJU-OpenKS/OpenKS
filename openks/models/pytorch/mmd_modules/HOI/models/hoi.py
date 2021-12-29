import torch
from torch import nn

from ..util.misc import NestedTensor, nested_tensor_from_tensor_list
from .tr_helper import MLP


class DETRHOI(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes,
                 num_queries, aux_loss=False, split_query=False, use_matching=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.use_matching = use_matching
        self.split_query = split_query

        hidden_dim = transformer.d_model
        if self.split_query:
            self.h_query_embed = nn.Embedding(num_queries, hidden_dim)
            self.o_query_embed = nn.Embedding(num_queries, hidden_dim)
            self.v_query_embed = nn.Embedding(num_queries, hidden_dim)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if self.use_matching:
            self.matching_embed = MLP(hidden_dim*2, hidden_dim, 2, 2)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        if not self.split_query:
            query_embed = self.query_embed.weight
        else:
            query_embed = {
                'h_query': self.h_query_embed.weight,
                'o_query': self.o_query_embed.weight,
                'v_query': self.v_query_embed.weight
            }
        hs, mem = self.transformer(
            self.input_proj(src), mask, query_embed, pos[-1])
        return self.prepare_for_triplet_output(hs, mem, pos, mask)

    def prepare_for_triplet_output(self, hs, mem, pos, mask):
        hs_sub, hs_obj, hs_verb = hs
        mem_sub, mem_obj, mem_verb = mem

        outputs_obj_class = self.obj_class_embed(hs_obj)
        outputs_verb_class = self.verb_class_embed(hs_verb)
        outputs_sub_coord = self.sub_bbox_embed(hs_sub).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs_obj).sigmoid()

        out = {
            'pred_obj_logits': outputs_obj_class[-1],
            'pred_verb_logits': outputs_verb_class[-1],
            'pred_sub_boxes': outputs_sub_coord[-1],
            'pred_obj_boxes': outputs_obj_coord[-1]
        }

        if self.use_matching:
            outputs_matching = self.matching_embed(torch.cat([hs_sub, hs_obj], dim=-1))
            out['pred_matching_logits'] = outputs_matching[-1]

        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_matching)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class,
                      outputs_sub_coord, outputs_obj_coord, outputs_matching=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.use_matching:
            return [{'pred_obj_logits': a,
                     'pred_verb_logits': b,
                     'pred_sub_boxes': c,
                     'pred_obj_boxes': d,
                     'pred_matching_logits': e}
                    for a, b, c, d, e in
                    zip(outputs_obj_class[:-1],
                        outputs_verb_class[:-1],
                        outputs_sub_coord[:-1],
                        outputs_obj_coord[:-1],
                        outputs_matching[:-1])
                    ]
        else:
            return [{
                'pred_obj_logits': a,
                'pred_verb_logits': b,
                'pred_sub_boxes': c,
                'pred_obj_boxes': d
            } for a, b, c, d in zip(
                outputs_obj_class[:-1],
                outputs_verb_class[:-1],
                outputs_sub_coord[:-1],
                outputs_obj_coord[:-1])
            ]


def build_detr(args, backbone, transformer, num_classes):
    if args.hoi:
        return DETRHOI(
            backbone,
            transformer,
            num_obj_classes=args.num_obj_classes,
            num_verb_classes=args.num_verb_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            split_query=args.split_query,
            use_matching=args.use_matching)

    raise ValueError('not implement!')
