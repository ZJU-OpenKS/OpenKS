# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _transpose_and_gather_feat
import torch.nn.functional as F


def intersection(bb1, bb2):
  l = torch.max(bb1[:,0,:], bb2[:,0,:])
  r = torch.min(bb1[:,1,], bb2[:,1,:])
  insec = (r-l).clamp(min=0)
  return insec

def union(bb1, bb2, insec):
  area1 = bb1[:,1,:] - bb1[:,0,:]
  area2 = bb2[:,1,:] - bb2[:,0,:]
  return area1 + area2 - insec

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt, mask):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  
  pred = pred.view(-1, pred.size(-2), pred.size(-1))
  gt = gt.view(-1, gt.size(-2), gt.size(-1))

  pred = pred[mask.view(-1)]
  gt = gt[mask.view(-1)]

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target, mask):
    return self.neg_loss(out, target, mask)

class RelLoss(nn.Module):
  def __init__(self):
    super(RelLoss, self).__init__()
    self._loss = nn.BCEWithLogitsLoss(reduce=False)

  def forward(self, pred, gt):
    pred = pred.view(-1, pred.size(-1))
    gt = gt.view(-1, gt.size(-1))

    # pred = pred[mask.view(-1)]
    # gt = gt[mask.view(-1)]

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
    return loss




class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    # print('out', output.shape)
    pred = _transpose_and_gather_feat(output, ind)
    # print('pred', pred.shape)
    # print('mask',mask.shape)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class TempLoss(nn.Module):
  def __init__(self):
    super(TempLoss, self).__init__()

  def forward(self, cls_score, offset, t_gt):
    t_gt = t_gt.float()
    batch, _, _ = offset.size()
    anchors = torch.zeros_like(offset)
    for n in range(5):
      anchors[:,0,n*64:n*64+64] = torch.arange(0,64).float().cuda()-(2**(n+1)-1)
      anchors[:,1,n*64:n*64+64] = torch.arange(0,64).float().cuda()+(2**(n+1)-1)
    valid_mask_1 = anchors[:,:1,:]>=0
    valid_mask_2 = anchors[:,1:,:]<=63
    valid_mask = (valid_mask_1 * valid_mask_2)
    anchors = anchors[valid_mask.expand_as(anchors)].view(batch,2, -1)
    cls_temp = nn.Sigmoid()(cls_score[valid_mask]).view(batch, -1)
    offset_temp = offset[valid_mask.expand_as(offset)].view(batch,2, -1)
    t_gt = t_gt.unsqueeze(-1).expand_as(anchors)
    inter = intersection(anchors, t_gt)
    u = union(anchors, t_gt, inter)
    tiou = inter/u
    cls_loss = -((1-tiou)*torch.log(1-cls_temp) + tiou*torch.log(cls_temp)).mean()
    pos_ind = tiou>0.5
    if pos_ind.sum() == 0:
      max_ind = torch.argmax(tiou)
      pos_ind = pos_ind.view(-1)
      pos_ind[max_ind] = 1
      pos_ind = pos_ind.view(tiou.size())
    # neg_num = pos_ind.sum().long()*10
    # neg_ind = tiou<0.3
    # cls_loss_neg = -(torch.log(1-cls_temp[neg_ind]))
    # cls_loss_neg,_ = torch.topk(cls_loss_neg,k=neg_num,dim=-1)
    # cls_loss_pos = -(torch.log(cls_temp[pos_ind]))
    # cls_loss = torch.cat([cls_loss_neg,cls_loss_pos],dim=0).mean()
    # _, sorted_index = torch.sort(cls_temp,dim=-1,descending=True)
    # sorted_index = sorted_index.unsqueeze(1).expand_as(anchors)
    pos_ind = pos_ind.unsqueeze(1).expand_as(anchors)
    pos_anchors = anchors[pos_ind].view(2,-1)
    t_gt = t_gt[pos_ind].view(2,-1)
    # pos_anchors = anchors.gather(dim=-1, index=sorted_index)[:,:,:10].contiguous().view(2, -1)
    # t_gt = t_gt.gather(dim=-1, index=sorted_index)[:,:,:10].contiguous().view(2, -1)
    pos_gt = torch.zeros_like(pos_anchors)
    pos_gt[0,:] = (t_gt[0,:] - pos_anchors[0,:])/64
    pos_gt[1,:] = (t_gt[1,:] - pos_anchors[1,:])/64
    offset_loss = F.l1_loss(offset_temp[pos_ind].view(2, -1), pos_gt, size_average=False)
    offset_loss = offset_loss/pos_ind.sum().float()
    return cls_loss, offset_loss



class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss_tm(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss_tm, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = output.view(output.size(0),-1)
    pred = pred.gather(-1, ind)
    # pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred , target, size_average=True)
    # loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
