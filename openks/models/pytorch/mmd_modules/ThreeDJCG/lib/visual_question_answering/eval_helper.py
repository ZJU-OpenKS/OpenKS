# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.nn_distance import nn_distance, huber_loss
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.ap_helper_fcos import parse_predictions, APCalculator
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch, box3d_iou_batch_tensor

from openks.models.pytorch.mmd_modules.ThreeDJCG.data.scannet.model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()  # For NMS

Cal25 = APCalculator(ap_iou_thresh=0.25)
Cal50 = APCalculator(ap_iou_thresh=0.5)
step_number, mAP25_metrics, mAP50_metrics = 0, {'mAP': 0, 'AR': 0}, {'mAP': 0, 'AR': 0}
step_output_freq = 50

def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)

    return corners_3d


@torch.no_grad()
def get_eval(data_dict, config, reference, use_lang_classifier=False, use_oracle=False, use_cat_rand=False,
             use_best=False, scanrefer_eval=False, mask=False, eval25=False, refresh_map=0):  # refresh: for evaluation; 1: immediately refresh; -1: do not refresh
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()
    # objectness_labels_batch = data_dict['objectness_label'].long()  # GT

    # config
    post_processing = {
        "remove_empty_box": True,
        "use_3d_nms": True,
        "nms_iou": 0.25,
        "use_old_type_nms": False,
        "cls_nms": True,
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }

    # if post_processing:  # Must Have NMS Mask; otherwise the iou precision and recall will be not right
    if data_dict['istrain'][0] == 0:  # Must Have NMS Mask; otherwise the iou precision and recall will be not right
        _ = parse_predictions(data_dict, post_processing)
        nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

        # construct valid mask
        pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        # label_masks = (objectness_labels_batch == 1).float()
    else:
        # raise NotImplementedError('NMS must be used')  # use all bboxes
        # construct valid mask
        pred_masks = (objectness_preds_batch == 1).float()
        # label_masks = (objectness_labels_batch == 1).float()

    pred_masks = pred_masks.detach()

    # Predict Answer
    pred_answer = data_dict['vqa_pred_answer'].detach().cpu().numpy()
    pred_related_object_confidence = data_dict['vqa_pred_related_object_confidence'].sigmoid().detach()

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
    test_type = data_dict["test_type_id"]
    # convert gt bbox parameters to bbox corners
    batch_size, lang_num_max, num_proposals = pred_related_object_confidence.shape[:3]
    lang_num = data_dict["lang_num"]

    if not scanrefer_eval:  # testing scanvqa
        AP_25, AR_25 = [], []
        AP_5, AR_5 = [], []
        answer_acc = []
        test_type_num = 4
        test_type_acc = []
        for i in range(test_type_num):
            test_type_list = []
            test_type_acc.append(test_type_list)
        # object score (matching)
        for i in range(batch_size):
            gt_obb_batch = config.param2obb_batch(gt_center_list[i][:, 0:3], gt_heading_class_list[i],
                                                  gt_heading_residual_list[i],
                                                  gt_size_class_list[i], gt_size_residual_list[i])
            gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
            pred_bbox_batch = get_3d_box_batch(pred_box_size[i], pred_heading[i], pred_center[i])

            gt_bbox_batch = torch.from_numpy(gt_bbox_batch).float().to(pred_masks.device)
            pred_bbox_batch = torch.from_numpy(pred_bbox_batch).float().to(pred_masks.device)
            for j in range(lang_num[i]):
                pred_related_batch = pred_related_object_confidence[i, j]
                eps, thres = 1e-9, 0.5
                acc_iou_rate_25, acc_iou_rate_5, all_positive, all_related = eps, eps, eps, eps
                positive_mask = (pred_related_batch>thres) * pred_masks[i, :, None]  # pred_masks!
                all_positive = sum(sum(positive_mask)) + eps

                batch_gt_map_cls, batch_pred_map_cls = [], []
                for x in range(4):
                    related_mask = pred_masks[i] * positive_mask[:, x]
                    if gt_related_object[i][j][x] is None:
                        continue
                    for y in gt_related_object[i][j][x]:
                        # convert the bbox parameters to bbox corners
                        # ious = box3d_iou_batch_tensor(pred_bbox_batch, np.tile(gt_bbox_batch[y], (num_proposals, 1, 1)))
                        ious = box3d_iou_batch_tensor(pred_bbox_batch, gt_bbox_batch[y, None].repeat(num_proposals, 1, 1))
                        ious = ious * related_mask

                        ious_ind = ious.argmax()
                        max_ious = ious[ious_ind]
                        # should we add the objectness mask?

                        all_related += 1  # tp+fn
                        if max_ious >= 0.25:
                            acc_iou_rate_25 += 1
                        if max_ious >= 0.5:
                            acc_iou_rate_5 += 1

                        batch_gt_map_cls.append((x, gt_bbox_batch[y].detach().cpu().numpy()))
                    batch_pred_map_cls.extend([(x, pred_bbox_batch[i].detach().cpu().numpy(), pred_related_batch[i, x].detach().cpu().numpy())
                                               for i in range(related_mask.shape[0]) if related_mask[i]])
                if eval25:
                    Cal25.step([batch_pred_map_cls], [batch_gt_map_cls])
                Cal50.step([batch_pred_map_cls], [batch_gt_map_cls])
                # print('all_positive', all_positive, 'acc_iou_rate', acc_iou_rate_25, acc_iou_rate_5, all_related, flush=True)
                AP_25.append(acc_iou_rate_25/all_positive)
                AR_25.append(acc_iou_rate_25/all_related)
                AP_5.append(acc_iou_rate_5/all_positive)
                AR_5.append(acc_iou_rate_5/all_related)
                # print(acc_iou_rate_25, acc_iou_rate_5, all_positive, all_related, 'ap, ar = ', AP_25[-1], AR_25[-1], flush=True)  # debugging
                id_pred = np.argmax(pred_answer[i, j])
                answer_acc.append((id_pred == gt_answer[i][j]))

                test_type_acc[test_type[i][j]].append((id_pred == gt_answer[i][j]))

        # store
        # data_dict["pred_mask"] = torch.tensor(pred_masks).cuda()
        data_dict["pred_mask"] = pred_masks.detach().clone()
        # data_dict["label_mask"] = label_masks
        data_dict['answer_acc'] = answer_acc
        data_dict['test_type_acc'] = test_type_acc
        data_dict["AP_0.25"] = AP_25
        data_dict["AR_0.25"] = AR_25
        data_dict["AP_0.5"] = AP_5
        data_dict["AR_0.5"] = AR_5

        global step_number, mAP25_metrics, mAP50_metrics
        if ((step_number % step_output_freq == 0 or data_dict['istrain'][0] == 0) and refresh_map==0) or refresh_map==1:  # refresh: refresh per 50 iter
            if eval25:
                mAP25_metrics = Cal25.compute_metrics()
                Cal25.reset()
            mAP50_metrics = Cal50.compute_metrics()
            Cal50.reset()
        step_number = step_number + 1

        data_dict['mAP50_metrics'] = mAP50_metrics
        if eval25:
            data_dict['mAP25_metrics'] = mAP25_metrics
        data_dict['mAP_0.25'] = mAP25_metrics['mAP']
        data_dict['mAP_0.5'] = mAP50_metrics['mAP']
        # print(mAP50_metrics['AR'], data_dict["AR_0.5"], flush=True)

        data_dict["vqa_acc"] = answer_acc

        data_dict["number_acc"] = test_type_acc[0]
        data_dict["color_acc"] = test_type_acc[1]
        data_dict["yes_no_acc"] = test_type_acc[2]
        data_dict["other_acc"] = test_type_acc[3]

    else:
        ious = []
        multiple = []
        others = []
        pred_bboxes = []
        gt_bboxes = []
        ref_acc = []
        related_object_labels = data_dict["related_object_labels"]
        #print("pred_ref", pred_ref.shape, gt_ref.shape)
        for i in range(batch_size):
            # compute the iou
            gt_obb_batch = config.param2obb_batch(gt_center_list[i][:, 0:3], gt_heading_class_list[i],
                                                  gt_heading_residual_list[i],
                                                  gt_size_class_list[i], gt_size_residual_list[i])
            gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
            pred_bbox_batch = get_3d_box_batch(pred_box_size[i], pred_heading[i], pred_center[i])

            gt_bbox_batch = torch.from_numpy(gt_bbox_batch).float().to(pred_masks.device)
            pred_bbox_batch = torch.from_numpy(pred_bbox_batch).float().to(pred_masks.device)
            for j in range(lang_num[i]):
                x = 2 if mask else 0
                pred_related_batch = pred_related_object_confidence[i, j, :, x]
                pred_bbox_id = torch.argmax(pred_related_batch * pred_masks[i, :], -1)
                gt_bbox_id = gt_related_object[i][j][x]
                assert len(gt_bbox_id) == 1, 'one grounded object!'
                gt_bbox_id = gt_bbox_id[0]
                pred_bbox = pred_bbox_batch[pred_bbox_id].cpu().numpy()
                gt_bbox = gt_bbox_batch[gt_bbox_id].cpu().numpy()
                iou = eval_ref_one_sample(pred_bbox, gt_bbox)
                ious.append(iou)

                # NOTE: get_3d_box() will return problematic bboxes
                pred_bboxes.append(pred_bbox)
                gt_bboxes.append(gt_bbox)

                # construct the multiple mask
                multiple.append(data_dict["unique_multiple_list"][i][j])

                # construct the others mask
                flag = 1 if data_dict["object_cat_list"][i][j] == 17 else 0
                others.append(flag)

                ref_acc.append(related_object_labels[i, j, pred_bbox_id, x])

        # if reference and use_lang_classifier:
        #     object_cat = torch.tensor(data_dict["object_cat_list"]).reshape(batch_size*len_nun_max)
        #     data_dict["lang_acc"] = (torch.argmax(data_dict['lang_scores'], 1) == object_cat).float().mean()
        # else:
        #     data_dict["lang_acc"] = torch.zeros(1)[0].cuda()

        data_dict["ref_acc"] = ref_acc
        # store
        data_dict["pred_mask"] = pred_masks.detach().clone()
        data_dict["ref_iou"] = ious
        data_dict["ref_iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
        data_dict["ref_iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]
        data_dict["ref_multiple_mask"] = multiple
        data_dict["ref_others_mask"] = others
        data_dict["pred_bboxes"] = pred_bboxes
        data_dict["gt_bboxes"] = gt_bboxes

    # lang
    if reference and use_lang_classifier:
        gt_lang_class = data_dict['vqa_question_id']
        data_dict["lang_acc"] = (torch.argmax(data_dict['vqa_question_lang_scores'], -1) == gt_lang_class).float().mean()
    else:
        data_dict["lang_acc"] = torch.zeros(1)[0].cuda()

    # --------------------------------------------
    # Some other statistics  (votenet)
    obj_pred_val = torch.argmax(data_dict['objectness_scores'], 2)  # B,K
    obj_acc = torch.sum(
        (obj_pred_val == data_dict['objectness_label'].long()).float() * data_dict['objectness_mask']) / (
                          torch.sum(data_dict['objectness_mask']) + 1e-6)
    data_dict['obj_acc'] = obj_acc
    # detection semantic classification
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1,
                                 data_dict['object_assignment'])  # select (B,K) from (B,K2)
    sem_cls_pred = data_dict['sem_cls_scores'].argmax(-1)  # (B,K)
    sem_match = (sem_cls_label == sem_cls_pred).float()
    data_dict["sem_acc"] = (sem_match * data_dict["pred_mask"]).sum() / data_dict["pred_mask"].sum()

    return data_dict

