import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util import box_ops


class PostProcessHOI(nn.Module):

    def __init__(self, use_matching, subject_category_id):
    # def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.use_matching = use_matching

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = \
            outputs['pred_obj_logits'], outputs['pred_verb_logits'], \
            outputs['pred_sub_boxes'], outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes), f'{len(out_obj_logits)} != {len(target_sizes)}'
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        if self.use_matching:
            out_matching_logits = outputs['pred_matching_logits']
            matching_scores = F.softmax(out_matching_logits, -1)[..., 1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_ops.box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_ops.box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(obj_scores)):
            range(len(obj_scores))
            os, ol, vs, sb, ob = obj_scores[index], obj_labels[index], verb_scores[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            if self.use_matching:
                ms = matching_scores[index]
                vs = vs * ms.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build_postprocess(args):
    if args.hoi:
        return {'hoi': PostProcessHOI(args.use_matching, args.subject_category_id)}

    raise ValueError('not implement!')
