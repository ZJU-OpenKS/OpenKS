import torch

from .backbone import build_backbone
from .transformer import build_transformer
from .matcher import build_matcher
from .hoi import build_detr
from .criterion import build_criterion
from .postprocess import build_postprocess


def build_weight_dict(args):
    weight_dict = {}
    if args.hoi:
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
        weight_dict['loss_verb_ce'] = args.verb_loss_coef
        weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
        weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
        weight_dict['loss_sub_giou'] = args.giou_loss_coef
        weight_dict['loss_obj_giou'] = args.giou_loss_coef
        if args.use_matching:
            weight_dict['loss_matching'] = args.matching_loss_coef
    else:
        weight_dict['loss_ce'] = 1
        weight_dict['loss_bbox'] = args.bbox_loss_coef
        weight_dict['loss_giou'] = args.giou_loss_coef
        if args.masks:
            weight_dict["loss_mask"] = args.mask_loss_coef
            weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    return weight_dict


def build_model(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = build_detr(args, backbone, transformer, num_classes)

    matcher = build_matcher(args)
    weight_dict = build_weight_dict(args)
    criterion = build_criterion(args, matcher, weight_dict,
                                num_classes).to(device)
    postprocess = build_postprocess(args)

    return model, criterion, postprocess
