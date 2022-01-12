from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from ..models.losses import FocalLoss
from ..models.losses import TempLoss, RegL1Loss, RegLoss,RelLoss, NormRegL1Loss, RegWeightedL1Loss, RegWeightedL1Loss_tm
from ..models.decode import ctdet_decode
from ..models.utils import _sigmoid
from ..utils.debugger import Debugger
from ..utils.post_process import ctdet_post_process
from ..utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg

    # self.crit_relevance = RelLoss()
    # self.crit_relevance = torch.nn.BCEWithLogitsLoss(size_average=True)
    # self.crit_offset = torch.nn.L1Loss(reduction='sum')
    # self.crit_offset = RegWeightedL1Loss_tm()

    self.crit_temp = TempLoss()

    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss, rel_loss, offset_loss = 0, 0, 0,0,0

    batch['reg_mask'] = batch['reg_mask'].view(-1)
# 
    # print(batch['reg_mask'])

    for key in batch.keys():
      if key not in ['tm', 't_gt', 'frame_id_init', 'tm_l','tm_mask','tm_ind','tm_offset','input', 'tokens', 'meta','reg_mask', 'input_mask']:
        # print(key)
        # print(batch[key].shape)
        batch[key] = batch[key].view([-1,1]+list(batch[key].size()[2:]))
        # batch[key] = batch[key][batch['reg_mask']]
        # print(key, batch[key].shape)



    for s in range(opt.num_stacks):
      output = outputs[s]
      # for key in output.keys():
        # print(key, output[key].shape)
      #   output[key] = output[key][batch['reg_mask']]
      batch['reg_mask'] = batch['reg_mask'].view(-1, 1)
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])
      # print(output['hm'].shape, output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      if batch['reg_mask'].any():
        hm_loss += self.crit(output['hm'], batch['hm'], batch['reg_mask']) / opt.num_stacks
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          if batch['reg_mask'].any():
            wh_loss += self.crit_reg(
              output['wh'], batch['reg_mask'],
              batch['ind'], batch['wh']) / opt.num_stacks
      
      if opt.reg_offset and opt.off_weight > 0:
        if batch['reg_mask'].any():
          off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                               batch['ind'], batch['reg']) / opt.num_stacks

      # print('checkpoint:1')
      # rel_loss += self.crit_relevance(_sigmoid(output['relevance']), batch['tm'].float())
      # print('checkpoint:2')
      # offset_loss += self.crit_offset(output['offset'], batch['tm_mask'], batch['tm_ind'], batch['tm_l'])
      # print('checkpoint:3')
      rel_loss, offset_loss = self.crit_temp(output['relevance'], output['offset'], batch['t_gt'])

    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.rel_weight*rel_loss + offset_loss
    # loss = opt.rel_weight*rel_loss + offset_loss
    loss_stats = {'loss': loss, 'o_loss': offset_loss,'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'rel_loss': rel_loss, 'off_loss': off_loss}
    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'o_loss', 'hm_loss', 'wh_loss', 'rel_loss', 'off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]