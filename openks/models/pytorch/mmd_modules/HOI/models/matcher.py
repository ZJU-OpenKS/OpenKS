"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, union_box


class HungarianMatcherHOI(nn.Module):

    def __init__(self,
                 cost_obj_class: float = 1,
                 cost_verb_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 box_matcher='split_max'):
        super().__init__()

        assert cost_obj_class != 0 or cost_verb_class != 0 or \
               cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert box_matcher in ['split_max', 'union_split_sum']
        self.box_matcher = box_matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2]

        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
        out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
        tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])

        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
        tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
        cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                            (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                            ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
        cost_boxes = self.get_cost_boxes(out_sub_bbox, tgt_sub_boxes, out_obj_bbox, tgt_obj_boxes)

        C = self.cost_obj_class * cost_obj_class + self.cost_verb_class * cost_verb_class + cost_boxes
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['obj_labels']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def get_cost_boxes(self, out_sub_bbox, tgt_sub_boxes, out_obj_bbox, tgt_obj_boxes):
        out_union_bbox = tgt_union_boxes = None
        if self.box_matcher in ['union_split_sum']:
            out_union_bbox = union_box(out_sub_bbox, out_obj_bbox)
            tgt_union_boxes = union_box(tgt_sub_boxes, tgt_obj_boxes)

        cost_bbox = self.get_cost_bbox(out_sub_bbox, tgt_sub_boxes, out_obj_bbox, tgt_obj_boxes,
                                       out_union_bbox=out_union_bbox, tgt_union_boxes=tgt_union_boxes)
        cost_giou = self.get_cost_giou(out_sub_bbox, tgt_sub_boxes, out_obj_bbox, tgt_obj_boxes,
                                       out_union_bbox=out_union_bbox, tgt_union_boxes=tgt_union_boxes)
        return self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

    def get_cost_bbox(self, out_sub_bbox, tgt_sub_boxes, out_obj_bbox, tgt_obj_boxes,
                      out_union_bbox=None, tgt_union_boxes=None):
        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        cost_union_bbox = None if out_union_bbox is None else torch.cdist(out_union_bbox, tgt_union_boxes, p=1)

        if cost_sub_bbox.shape[1] == 0:
            return cost_sub_bbox

        if self.box_matcher == 'split_max':
            return torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]
        else:
            return 0.33 * cost_sub_bbox + 0.33 * cost_obj_bbox + 0.33 * cost_union_bbox

    def get_cost_giou(self, out_sub_bbox, tgt_sub_boxes, out_obj_bbox, tgt_obj_boxes,
                      out_union_bbox=None, tgt_union_boxes=None):
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        cost_union_giou = None if out_union_bbox is None else -generalized_box_iou(
            box_cxcywh_to_xyxy(out_union_bbox), box_cxcywh_to_xyxy(tgt_union_boxes))

        if cost_sub_giou.shape[1] == 0:
            return cost_sub_giou

        if self.box_matcher == 'split_max':
            return torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]
        else:
            return 0.33 * cost_sub_giou + 0.33 * cost_obj_giou + 0.33 * cost_union_giou


def build_matcher(args):
    if args.hoi:
        if not args.use_matching:
            return HungarianMatcherHOI(
                cost_obj_class=args.set_cost_obj_class,
                cost_verb_class=args.set_cost_verb_class,
                cost_bbox=args.set_cost_bbox,
                cost_giou=args.set_cost_giou,
                box_matcher=args.box_matcher)
        else:
            raise ValueError('not implement!')
    else:
        raise ValueError('not implement!')
