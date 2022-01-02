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
#sys.path.append(os.path.join(os.getcwd(), os.pardir, "openks/models/pytorch/mmd_modules/ThreeDJCG")) # HACK add the lib folder
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.nn_distance import nn_distance, huber_loss
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.ap_helper import parse_predictions
from .loss import SoftmaxRankingLoss
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.config import CONF
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.box_util import rotz_batch_pytorch

FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """
    # Load ground truth votes and assign them to seed points
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1]  # B,num_seed,3
    vote_xyz = data_dict['vote_xyz']  # B,num_seed*vote_factor,3
    seed_inds = data_dict['seed_inds'].long()  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += data_dict['seed_xyz'].repeat(1, 1, 3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size * num_seed, -1,
                                     3)  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size * num_seed, GT_VOTE_FACTOR,
                                               3)  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)

    return vote_loss

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['aggregated_vote_xyz']
    gt_center = data_dict['center_label'][:, :, 0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B, K)).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

    # Set assignment
    object_assignment = ind1  # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment


def compute_box_loss(data_dict, config, distance_huber_loss_thres=0.15, use_centerness=False):
    num_heading_bin = config.num_heading_bin

    gt_heading_class = data_dict['gt_assigned_heading_class']  # (B, N)
    gt_heading_residual = data_dict['gt_assigned_heading_residual']  # (B, N, 3)
    gt_distance = data_dict['gt_assigned_distance']  # (B, N, 6)
    gt_centerness =  data_dict['gt_assigned_centerness'].view(-1)  # (B*N,)
    objectness_label =  data_dict['objectness_label'].float()  # (B, N)
    objectness_mask = data_dict['objectness_mask'].float()  # (B, N)
    batch_size = objectness_label.shape[0]
    num_proposal = objectness_label.shape[1]

    # heading class loss + heading residual loss
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict['heading_scores'].transpose(2,1), gt_heading_class)  # (B, N)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_normalized_label = gt_heading_residual / (np.pi/num_heading_bin)
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, num_proposal, num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, gt_heading_class.unsqueeze(-1), 1)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0)  # (B, N)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 6 distance loss ~ smooth-L1 loss / huber loss
    rois = data_dict['rois']  # (B, N, 6)
    distance_loss = torch.mean(huber_loss(rois - gt_distance, delta=distance_huber_loss_thres), -1)  # (B, N)
    distance_loss = torch.sum(distance_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    assert (not use_centerness)
    # centerness loss ~ BCE loss
    # centerness_scores = data_dict['centerness_scores_0'].view(-1)  # (B*N,)
    # criterion_centerness = nn.BCEWithLogitsLoss(reduction='none')
    # centerness_loss = criterion_centerness(centerness_scores, gt_centerness)
    # centerness_loss = torch.sum(centerness_loss * objectness_mask.view(-1))/(torch.sum(objectness_mask)+1e-6)
    return heading_class_loss, heading_residual_normalized_loss, distance_loss


def recover_assigned_gt_bboxes(data_dict, config, object_assignment):
    # Return: Assigned Objects
    num_heading_bin = config.num_heading_bin
    mean_size_arr = config.mean_size_arr
    num_size_cluster = config.num_size_cluster

    aggregated_vote_xyz = data_dict['aggregated_vote_xyz'].clone()  # (B, N, 3)
    batch_size = object_assignment.shape[0]
    num_proposal = object_assignment.shape[1]

    # gather gt_center_label by object_assignment
    gt_center = torch.gather(data_dict['center_label'], dim=1, index=object_assignment.unsqueeze(-1).repeat(1,1,3))  # (B, N, 3)

    # gather gt bbox heading
    heading_class_label = torch.gather(data_dict['heading_class_label'], 1, object_assignment)  # (B, K)
    heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1, object_assignment)  # (B, K)

    if num_heading_bin != 1:  # for SUN RGBD
        gt_heading = heading_class_label.float() * ((2*np.pi)/float(num_heading_bin)) + heading_residual_label
    else:  # for ScanNetV2
        gt_heading = torch.zeros((batch_size, num_proposal)).cuda()

    # gather gt bbox size
    size_class_label = torch.gather(data_dict['size_class_label'], 1, object_assignment)  # (B, K)
    size_residual_label = torch.gather(data_dict['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3))  # (B, N, 3)
    mean_size_arr_expanded = torch.ones((batch_size, num_proposal, num_size_cluster, 3)).cuda() * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)  # (B, N, num_size_cluster, 3)
    mean_size = torch.gather(mean_size_arr_expanded, 2, size_class_label.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).squeeze(2)  # (B, N, 3)
    gt_bbox_size = mean_size + size_residual_label  # (B, N, 3)
    gt_bbox_size_half = gt_bbox_size / 2  # (B, N, 3)

    # Compute GT distance to 6 faces
    aggregated_vote_xyz -= gt_center  # (B, N, 3) ~ 中心化
    aggregated_vote_xyz = aggregated_vote_xyz.view(-1, 3).unsqueeze(1)  # (B*N, 1, 3)
    # Rotate aggregated_vote_xyz towards negative gt_heading
    R = rotz_batch_pytorch(-gt_heading.float()).view(-1, 3, 3)  # (B*N, 3, 3)
    aggregated_vote_xyz = torch.matmul(aggregated_vote_xyz, R)  # (B*N, 1, 3)
    aggregated_vote_xyz = aggregated_vote_xyz.squeeze(1).view(batch_size, num_proposal, 3)  # (B, N, 3)
    bld = gt_bbox_size_half + aggregated_vote_xyz  # (B, N, 3) ~ back_left_down distance
    fru = gt_bbox_size_half - aggregated_vote_xyz  # (B, N, 3) ~ front_right_up distance
    gt_distance = torch.cat((bld, fru), dim=2)  # (B, N, 6) ~ back_left_down_front_right_up

    # Compute inside label ~ whether the aggregated_vote_xyz in assigned bbox or not
    inside_label = torch.eq(torch.sum(torch.gt(gt_distance, 0), dim=2), 6).long()  # (B, N)

    # Compute centerness label
    distance_min = torch.min(bld, fru)  # (B, N, 3)
    distance_max = torch.max(bld, fru)  # (B, N, 3)
    distance_min_over_max = distance_min / (distance_max+1e-6)  # (B, N, 3)
    distance_min_over_max = torch.clamp(distance_min_over_max, min=1e-6)
    exponent = 3
    if exponent == 1:
        gt_centerness = distance_min_over_max[:,:,0]*distance_min_over_max[:,:,1]*distance_min_over_max[:,:,2]
    elif exponent in [2, 3]:
        gt_centerness = torch.pow(distance_min_over_max[:,:,0]*distance_min_over_max[:,:,1]*distance_min_over_max[:,:,2], exponent=1/exponent)
    else:
        raise NotImplementedError
    gt_centerness *= inside_label.float()

    return gt_center, heading_class_label, heading_residual_label, gt_heading, gt_distance, inside_label, gt_centerness, gt_bbox_size


# compute_box_and_sem_cls_loss
def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_reg_loss (distance loss)
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]

    # Set assignment
    # (B,K) with values in 0,1,...,K2-1
    # recover_assigned_gt_assigned_bboxes
    gt_assigned_center, gt_assigned_heading_class, gt_assigned_heading_residual, gt_assigned_heading, gt_assigned_distance, \
            inside_label, gt_assigned_centerness, gt_assigned_bbox_size = recover_assigned_gt_bboxes(data_dict, config, object_assignment)

    data_dict['gt_assigned_center'] = gt_assigned_center
    data_dict['gt_assigned_heading_class'] = gt_assigned_heading_class
    data_dict['gt_assigned_heading_residual'] = gt_assigned_heading_residual
    data_dict['gt_assigned_heading'] = gt_assigned_heading
    data_dict['gt_assigned_distance'] = gt_assigned_distance
    data_dict['gt_assigned_centerness'] = gt_assigned_centerness

    heading_class_loss, heading_residual_normalized_loss, distance_loss = compute_box_loss(data_dict, config, distance_huber_loss_thres=0.15)

    objectness_label = data_dict["objectness_label"]
    # (object level) Semantic cls loss
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict['sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label.float()) / (torch.sum(objectness_label) + 1e-6)
    return heading_class_loss, heading_residual_normalized_loss, distance_loss, sem_cls_loss

