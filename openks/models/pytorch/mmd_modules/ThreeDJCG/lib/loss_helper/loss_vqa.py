# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.nn_distance import nn_distance, huber_loss
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.ap_helper_fcos import parse_predictions
# from .loss import SoftmaxRankingLoss, SigmoidRankingLoss
from .loss import SoftmaxRankingFocalLoss, SigmoidRankingFocalLoss
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.box_util import rotz_batch_pytorch
from .loss_detection import compute_vote_loss, compute_objectness_loss, compute_box_loss, compute_box_and_sem_cls_loss


def compute_vqa_loss(data_dict, config, no_reference=False):  # todo: no_reference not used
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """


    # unpack
    pred_related_object_confidence = data_dict['vqa_pred_related_object_confidence']
    pred_answer = data_dict['vqa_pred_answer']

    # predicted bbox
    pred_heading = data_dict['pred_heading'].detach().cpu().numpy() # B,num_proposal
    pred_center = data_dict['pred_center'].detach().cpu().numpy() # (B, num_proposal)
    pred_box_size = data_dict['pred_size'].detach().cpu().numpy() # (B, num_proposal, 3)

    gt_center_list = data_dict['center_label'].cpu().numpy()
    gt_heading_class_list = data_dict['heading_class_label'].cpu().numpy()
    gt_heading_residual_list = data_dict['heading_residual_label'].cpu().numpy()
    gt_size_class_list = data_dict['size_class_label'].cpu().numpy()
    gt_size_residual_list = data_dict['size_residual_label'].cpu().numpy()

    gt_related_object = data_dict['vqa_related_object_id']
    gt_answer = data_dict['vqa_answer_id']
    # convert gt bbox parameters to bbox corners
    batch_size, lang_num_max, num_proposals = pred_related_object_confidence.shape[:3]
    lang_num = data_dict["lang_num"]
    max_iou_rate_25 = 0
    max_iou_rate_5 = 0

    # object score loss (matching)
    criterion_related_object = SigmoidRankingFocalLoss()
    criterion_vqa_answer = SoftmaxRankingFocalLoss()
    related_object_loss, vqa_answer_loss = 0., 0.
    related_objects_number = 0
    gt_related_object_label = np.zeros((batch_size, lang_num_max, num_proposals, 4))
    gt_answer_class = np.zeros((batch_size, lang_num_max, pred_answer.shape[-1]))
    gt_related_object_mask = np.ones((batch_size, lang_num_max, 4))  # remove the unlabeled data
    for i in range(batch_size):
        gt_obb_batch = config.param2obb_batch(gt_center_list[i][:, 0:3], gt_heading_class_list[i],
                                        gt_heading_residual_list[i],
                                        gt_size_class_list[i], gt_size_residual_list[i])
        gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
        pred_bbox_batch = get_3d_box_batch(pred_box_size[i], pred_heading[i], pred_center[i])
        for j in range(lang_num[i]):
            for x in range(4):
                if gt_related_object[i][j][x] is None:
                    gt_related_object_mask[i, j, x] = 0
                    continue
                for y in gt_related_object[i][j][x]:
                    # convert the bbox parameters to bbox corners
                    ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[y], (num_proposals, 1, 1)))

                    ious_ind = ious.argmax()
                    max_ious = ious[ious_ind]
                    # should we add the objectness mask?

                    related_objects_number += 1
                    if max_ious >= 0.25:
                        gt_related_object_label[i, j, ious.argmax(), x] = 1  # treat the bbox with highest iou score as the gt
                        max_iou_rate_25 += 1
                    if max_ious >= 0.5:
                        gt_related_object_label[i, j, ious>0.5, x] = 1  # treat the bbox with highest iou score as the gt
                        max_iou_rate_5 += 1
            gt_answer_class[i, j, gt_answer[i][j]] = 1
        # import ipdb
        # ipdb.set_trace()

        # reference loss
        related_object_loss += criterion_related_object(pred_related_object_confidence[i, :lang_num[i]],
                                                        torch.FloatTensor(gt_related_object_label[i, :lang_num[i]]).to(pred_related_object_confidence.device),
                                                        torch.FloatTensor(gt_related_object_mask[i, :lang_num[i]])[:, None, :].to(pred_related_object_confidence.device))
        vqa_answer_loss += criterion_vqa_answer(pred_answer[i, :lang_num[i]],
                                                torch.FloatTensor(gt_answer_class[i, :lang_num[i]]).to(pred_related_object_confidence.device))
    data_dict["related_object_labels"] = gt_related_object_label
    related_object_loss = related_object_loss  # should / 256 ,but it is too small

    if related_objects_number == 0:
        print('[Loss Helper Warning!] No related object', flush=True)
        max_iou_rate_25 = max_iou_rate_5 = 1e-9
        related_objects_number = 1e-9

    # related_object_max_iou_rate_0.25
    # related_object_max_iou25_AR
    data_dict['related_object_max_iou25_AR'] = max_iou_rate_25 / related_objects_number
    data_dict['related_object_max_iou50_AR'] = max_iou_rate_5 / related_objects_number  # sum(lang_num.cpu().numpy())

    # related_object_loss = related_object_loss * 256  # add weight; for the reason that there are very little positive samples
    related_object_loss = related_object_loss / batch_size
    vqa_answer_loss = vqa_answer_loss / batch_size

    data_dict['related_object_loss'] = related_object_loss
    data_dict['vqa_loss'] = vqa_answer_loss
    return related_object_loss, vqa_answer_loss


def compute_lang_classification_loss(data_dict):  # auxiliary loss
    # 1. language class loss (initial, in the langauge module)
    # 2. object class loss (second, cross-modal object class loss)
    criterion_lang_type = torch.nn.CrossEntropyLoss()
    criterion_related_sem_cls = SigmoidRankingFocalLoss()

    pred_lang_class = data_dict['vqa_question_lang_scores']
    pred_lang_sem_cls = data_dict['vqa_pred_lang_sem_cls']

    gt_lang_class = data_dict['vqa_question_id']
    gt_lang_sem_cls_list = data_dict['vqa_related_object_sem_id']

    batch_size, lang_num_max = gt_lang_class.shape
    gt_lang_sem_cls = np.zeros((batch_size, lang_num_max, pred_lang_sem_cls.shape[2], 4))
    gt_lang_sem_cls_mask = np.ones((batch_size, lang_num_max, 4))

    lang_num = data_dict["lang_num"]
    lang_type_loss, related_object_sem_cls_loss = 0., 0.

    for i in range(batch_size):
        num = lang_num[i]
        lang_type_loss += criterion_lang_type(pred_lang_class[i, :num], gt_lang_class[i, :num])  # question_class
        for j in range(lang_num[i]):
            for x in range(4):
                if gt_lang_sem_cls_list[i][j][x] is None:
                    gt_lang_sem_cls_mask[i,j,x] = 0
                    continue
                for y in gt_lang_sem_cls_list[i][j][x]:
                    gt_lang_sem_cls[i,j,y,x] = 1
        related_object_sem_cls_loss += criterion_related_sem_cls(pred_lang_sem_cls[i, :num],
                                                                 torch.FloatTensor(gt_lang_sem_cls[i, :num]).to(pred_lang_sem_cls.device),
                                                                 torch.FloatTensor(gt_lang_sem_cls_mask[i, :num])[:, None, :].to(pred_lang_sem_cls.device))
    # related_object_sem_cls_loss = related_object_sem_cls_loss * 18
    # related_object_sem_cls_loss = related_object_sem_cls_loss / pred_lang_sem_cls.shape[2]  # mean; mask shape is [..., 1, ...]

    lang_type_loss = lang_type_loss / batch_size
    related_object_sem_cls_loss = related_object_sem_cls_loss / batch_size

    data_dict['lang_type_loss'] = lang_type_loss
    data_dict['related_object_sem_cls_loss'] = related_object_sem_cls_loss
    return lang_type_loss, related_object_sem_cls_loss


def get_loss(data_dict, config, detection=True, qa=True, use_lang_classifier=True):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(objectness_label.float())/float(total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]

    # Box loss and sem cls loss
    heading_cls_loss, heading_reg_loss, size_distance_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * sem_cls_loss
    box_loss = box_loss + 20 * size_distance_loss

    # objectness; Nothing
    obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==data_dict["objectness_label"].long()).float()*data_dict["objectness_mask"])/(torch.sum(data_dict["objectness_mask"])+1e-6)
    data_dict["obj_acc"] = obj_acc

    if detection:
        data_dict["vote_loss"] = vote_loss
        data_dict["objectness_loss"] = objectness_loss
        data_dict["heading_cls_loss"] = heading_cls_loss
        data_dict["heading_reg_loss"] = heading_reg_loss
        data_dict["size_distance_loss"] = size_distance_loss
        data_dict["sem_cls_loss"] = sem_cls_loss
        data_dict["box_loss"] = box_loss
    else:
        device = vote_loss.device()
        data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_distance_loss"] = torch.zeros(1)[0].to(device)
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["box_loss"] = torch.zeros(1)[0].to(device)

    related_object_loss, answer_loss = compute_vqa_loss(data_dict, config)
    lang_type_loss, related_object_sem_cls_loss = compute_lang_classification_loss(data_dict)

    # related_object_loss = 0

    data_dict['vqa_all_loss'] = related_object_loss + answer_loss
    # TODO: lang_type_loss is wrong?
    # data_dict['lang_loss'] = lang_type_loss + related_object_sem_cls_loss
    # TODO: related_object_sem_cls_loss works?
    data_dict['lang_loss'] = related_object_sem_cls_loss
    # data_dict['lang_loss'] = torch.zeros_like(answer_loss)

    # Final loss function
    # loss = data_dict['vote_loss'] + 0.1 * data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1 * data_dict['sem_cls_loss'] + 0.03 * data_dict["ref_loss"] + 0.03 * data_dict["lang_loss"]
    loss = 0

    # Final loss function
    if detection:
        # sem_cls loss is included in the box_loss
        # detection_loss = detection_loss + 0.1 * data_dict['sem_cls_loss']
        detection_loss = data_dict["vote_loss"] + 0.1*data_dict["objectness_loss"] + 1.0*data_dict["box_loss"]
        detection_loss *= 10 # amplify
        loss = loss + detection_loss

    if qa:
        loss = loss + 0.5 * data_dict["vqa_all_loss"]
    if use_lang_classifier:
        loss = loss + 0.5 * data_dict["lang_loss"]
    data_dict["loss"] = loss

    return data_dict
