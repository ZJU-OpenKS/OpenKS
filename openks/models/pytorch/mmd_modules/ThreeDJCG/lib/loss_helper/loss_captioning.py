# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from .loss_detection import compute_vote_loss, compute_objectness_loss, compute_box_loss, compute_box_and_sem_cls_loss


FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


def compute_cap_loss(data_dict, config, weights):
    """ Compute cluster caption loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cap_loss, cap_acc
    """

    # unpack
    batch_size, len_nun_max = data_dict["lang_feat_list"].shape[:2]
    pred_caps = data_dict["lang_cap"] # (B * len_nun_max, num_words - 1, num_vocabs)
    num_words = data_dict["lang_len_list"].reshape(batch_size * len_nun_max).max()
    #target_caps = data_dict["lang_ids"][:, 1:num_words] # (B * len_nun_max, num_words - 1)
    target_caps = data_dict["lang_ids_list"][:, :, 1:num_words]  # (B, len_nun_max, num_words - 1)

    _, _, num_vocabs = pred_caps.shape
    pred_caps = pred_caps.reshape(batch_size, len_nun_max, -1, num_vocabs) # (B, len_nun_max, num_words - 1, num_vocabs)

    # caption loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    cap_loss_all = 0.
    lang_num = data_dict["lang_num"]
    good_bbox_masks = data_dict["good_bbox_masks"].unsqueeze(1).repeat(1, num_words-1) # (B * len_nun_max, num_words - 1)
    good_bbox_masks = good_bbox_masks.reshape(batch_size, len_nun_max, -1).float() # (B, len_nun_max, num_words - 1)
    #print("pred_caps", pred_caps.shape)
    #print("target_caps", target_caps.shape)
    for i in range(batch_size):
        #print("pred_caps1", pred_caps[i, :lang_num[i]].reshape(-1, num_vocabs).shape)
        #print("target_caps1", target_caps[i, :lang_num[i]].reshape(-1).shape)
        cap_loss = criterion(pred_caps[i, :lang_num[i]].reshape(-1, num_vocabs), target_caps[i, :lang_num[i]].reshape(-1))
        good_bbox_mask = good_bbox_masks[i, :lang_num[i]].reshape(-1).float()
        #print("cap_loss",cap_loss.shape)
        #print("good_bbox_mask", good_bbox_mask.shape)
        cap_loss = torch.sum(cap_loss * good_bbox_mask) / (torch.sum(good_bbox_mask) + 1e-6)
        cap_loss_all = cap_loss_all + cap_loss

    cap_loss = cap_loss_all / batch_size

    cap_acc_all = 0.
    num_good_bbox = data_dict["good_bbox_masks"].sum()
    good_bbox_masks = data_dict["good_bbox_masks"].reshape(batch_size, len_nun_max)
    if num_good_bbox > 0:  # only apply loss on the good boxes
        for i in range(batch_size):
            pred_cap = pred_caps[i, :lang_num[i]]
            target_cap = target_caps[i, :lang_num[i]]
            good_bbox_mask = good_bbox_masks[i, :lang_num[i]]
            if good_bbox_mask.sum() > 0:
                #print("pred_cap", pred_cap.shape)
                #print("target_cap", target_cap.shape)
                #print("good_bbox_mask", good_bbox_mask.shape)
                pred_cap = pred_cap[good_bbox_mask]  # num_good_bbox
                target_cap = target_cap[good_bbox_mask]  # num_good_bbox
                # caption acc
                #print("pred_cap1", pred_cap.shape)
                #print("target_cap1", target_cap.shape)
                pred_cap = pred_cap.reshape(-1, num_vocabs).argmax(-1)  # num_good_bbox * (num_words - 1)
                target_cap = target_cap.reshape(-1)  # num_good_bbox * (num_words - 1)
                masks = target_cap != 0
                masked_pred_caps = pred_cap[masks]
                masked_target_caps = target_cap[masks]
                cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
            else:
                cap_acc = 0
            cap_acc_all = cap_acc_all + cap_acc
        cap_acc = cap_acc_all / batch_size
    else:
        cap_acc = torch.zeros(1)[0].cuda()
    return cap_loss, cap_acc


def get_object_cap_loss(data_dict, config, weights, classify=True, caption=True):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    if classify:
        cls_loss, cls_acc = compute_object_cls_loss(data_dict, weights)

        data_dict["cls_loss"] = cls_loss
        data_dict["cls_acc"] = cls_acc
    else:
        data_dict["cls_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cls_acc"] = torch.zeros(1)[0].cuda()

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cap_acc"] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = data_dict["cls_loss"] + data_dict["cap_loss"]

    # loss *= 10 # amplify

    data_dict["loss"] = loss

    return data_dict


def get_scene_cap_loss(data_dict, device, config, weights, 
    detection=True, caption=True):
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
        device = vote_loss.device
        data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_distance_loss"] = torch.zeros(1)[0].to(device)
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["box_loss"] = torch.zeros(1)[0].to(device)

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].to(device)
        data_dict["cap_acc"] = torch.zeros(1)[0].to(device)
        data_dict["pred_ious"] = torch.zeros(1)[0].to(device)

    # Final loss function
    # loss = data_dict["vote_loss"] + 0.1 * data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"] + data_dict["cap_loss"]

    if detection:
        loss = data_dict["vote_loss"] + 0.1*data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"]
        # loss = data_dict["vote_loss"] + 1.0*data_dict["objectness_loss"] + 1.0*data_dict["box_loss"]
        loss *= 10 # amplify
        if caption:
            loss += 0.2*data_dict["cap_loss"]
    else:
        loss = 0.2*data_dict["cap_loss"]

    data_dict["loss"] = loss

    return data_dict

