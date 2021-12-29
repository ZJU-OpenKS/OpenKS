from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from ..models.data_parallel import DataParallel
from ..utils.utils import AverageMeter
import torch.nn as nn
import gc

def intersection(bb1, bb2):
  x_min = torch.max(bb1[:,0], bb2[:,0])
  x_max = torch.min(bb1[:,2], bb2[:,2])
  y_min = torch.max(bb1[:,1], bb2[:,1])
  y_max = torch.min(bb1[:,3], bb2[:,3])
  insec_w = (x_max - x_min).clamp(min=0)
  insec_h = (y_max - y_min).clamp(min=0)
  return insec_w * insec_h

def union(bb1, bb2, insec):
  area1 = (bb1[:,2] - bb1[:,0]) * (bb1[:,3] - bb1[:,1])
  area2 = (bb2[:,2] - bb2[:,0]) * (bb2[:,3] - bb2[:,1])
  return area1 + area2 - insec


def accuracy(output, gt, mask, opt, th=[0.3,0.5]):
  if mask.any():
    temp = output['hm'].view(-1,opt.output_h, opt.output_w)[mask]
    # print('temp:', temp.shape)
    ind = torch.argmax(temp.view(temp.size(0),-1),1)
    wh = output['wh'].view(-1,2,opt.output_h,opt.output_w)[mask]
    # print('wh:', wh.shape)
    # wh = torch.index_select(wh.view(wh.size(0),wh.size(1),-1), dim=-1, index=ind)
    reg = output['reg'].view(-1,2,opt.output_h,opt.output_w)[mask]
    # reg = torch.index_select(reg.view(reg.size(0),reg.size(1),-1), dim=-1, index=ind)
    w_index = torch.arange(0,opt.output_w).float().cuda().view(1,1,opt.output_w).expand(wh.size(0),opt.output_h,opt.output_w)
    h_index = torch.arange(0,opt.output_h).float().cuda().view(1,opt.output_h,1).expand(wh.size(0),opt.output_h,opt.output_w)

    wh_index = torch.stack([w_index, h_index], dim=1)

    bboxes = torch.cat ([wh_index + reg - wh/2, wh_index+ reg + wh/2], dim=1).view(wh_index.size(0), 4, -1)

    bbox_list = []
    # print(bboxes.shape, ind)
    for k, i in enumerate(ind):
      bbox_list.append(bboxes[k,:,i])
    bboxes = torch.stack(bbox_list, dim=0)
    gt_boxes = gt.view(-1,6)[:,:4]

    # print('bb:', bboxes.shape, bboxes)

    # print('gt:', gt_boxes.shape, gt_boxes)


    insec = intersection(bboxes, gt_boxes)

    u = union(bboxes, gt_boxes, insec)

    iou = insec/u

    num = iou.size(0)

    acc_list = []
    for t in th:
      corr = iou[iou>=t].size(0)
      acc = corr/num*100
      acc_list.append(acc)
  # print(iou)

  else:
    num = 0
    acc_list = []
    for t in th:
      acc = 0
      acc_list.append(acc)
  return acc_list, num


def rel_acc(output, target, th=0.5):
  num = output.shape[0]
  acc=0
  # corr = 0
  # target_temp = target.sum(dim=-1)
  # for b in range(target_temp.shape[0]):
  #   if target_temp[b,0] > 0:
  #     t_max = torch.argmax(output[b,0,:])
  #     if target[b,0,t_max] > 0:
  #       corr+=1
  #   elif (output[b,0,:]<0.25).all():
  #       corr+=1
  #   if target_temp[b,1] > 0:
  #     t_max = torch.argmax(output[b,1,:])
  #     if target[b,1,t_max] > 0:
  #       corr+=1
  #   elif (output[b,1,:]<0.25).all():
  #       corr+=1
  # t_max = torch.argmax(output,dim=-1)
  # corr=(target.gather(-1,t_max.view(-1,1,1))>0).sum().float()
  # acc = corr/ float(num) *100
  # for i in range(num):
  #   print(target[i,0,:],output[i,0,:],target.gather(-1,t_max.view(-1,1,1))[i],t_max[i],output[i,0,t_max[i]])
  return acc, num

def t_iou(cls_score, se_gt, offset, target, tm_ind, t_gt):
  gt_iou, iou=0,0
  # tc = tm_ind[:,0].float()
  # target= target.float()
  # preds_temp = preds[:,0,:].gather(-1,tm_ind[:,0].view(-1,1))
  # # print(preds_temp.shape, target[:,:1].shape)
  # insec = torch.min(preds_temp,target[:,:1])
  # union = torch.max(preds_temp,target[:,:1])
  # gt_iou = insec/union*100
  # gt_iou_mask = gt_iou>0
  # gt_iou = gt_iou[gt_iou_mask].mean().detach().item()
  # if gt_iou > 0:
  #   pass
  # else:
  #   gt_iou = 0
  # gt_s = tc - target[:,:1].view(-1)/2
  # gt_e = tc + target[:,:1].view(-1)/2
  # inds = torch.argmax(se_scores, dim=-1)
  # pred_l = preds[:,0,:].gather(-1,inds.view(-1,1)).view(-1)
  # pred_s = inds.float() - pred_l
  # pred_e = inds.float() + pred_l
  # insec = (torch.min(gt_e, pred_e)-torch.max(gt_s,pred_s)).clamp(min=0)
  # union = torch.max(gt_e, pred_e)-torch.min(gt_s,pred_s)
  # iou = (insec/union).mean().detach().item() * 100
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
  max_ind = torch.argmax(cls_temp,dim=-1,keepdim=True).unsqueeze(1)
  max_ind = max_ind.expand(max_ind.size(0),offset_temp.size(1),max_ind.size(2))
  offset_temp = offset_temp.gather(dim=-1,index=max_ind)
  anchors = anchors.gather(dim=-1,index=max_ind)
  preds = anchors+offset_temp*64
  insec = (torch.min(anchors[:,1,0], t_gt[:,1]) - torch.max(anchors[:,0,0], t_gt[:,0])).clamp(min=0)
  union = t_gt[:,1]-t_gt[:,0] + anchors[:,1,0]- anchors[:,0,0] - insec
  gt_iou = (insec/union).mean().detach().item()*100
  insec = (torch.min(preds[:,1,0], t_gt[:,1]) - torch.max(preds[:,0,0], t_gt[:,0])).clamp(min=0)
  union = t_gt[:,1]-t_gt[:,0] + preds[:,1,0]- preds[:,0,0] - insec
  iou = (insec/union).mean().detach().item()*100

  return gt_iou, iou, offset.size(0), offset.size(0)





class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'], batch['tokens'], batch['input_mask'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

  def train(self, mode=True):
    super(ModelWithLoss, self).train(mode)
    self.training = mode
    for module in self.children():
        module.train(mode)
    for m in self.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for params in m.parameters():
                params.requires_grad = False

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    # if epoch == 1:
    #   self.opt.rel_weight = 0
    # else:
    #   self.opt.rel_weight = 1
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    accuracy_stat_3 = AverageMeter()
    accuracy_stat_5 = AverageMeter()
    accuracy_rel = AverageMeter()
    gt_iou = AverageMeter()
    pred_iou = AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      # print(123, batch['meta']['sents'])

      for k in batch:
        if k != 'meta'and k!='sents':
          if isinstance(batch[k],list):
            for t in range(len(batch[k])):
              # print(t,k)
              for j in range(len(batch[k][t])):
                # print(j,k)
                batch[k][t][j] = batch[k][t][j].to(device=opt.device, non_blocking=True)
          else:
            batch[k] = batch[k].to(device=opt.device, non_blocking=True)

    

      # print(batch['sents'])
      output, loss, loss_stats = model_with_loss(batch)

      (acc_3, acc_5),num = accuracy(output, batch['gt_det'].view(-1,6)[batch['reg_mask'].view(-1)], batch['reg_mask'].view(-1), self.opt)

      acc_rel, rel_num = rel_acc(output['relevance'], batch['tm'])

      gt_iou_stat, iou_stat, gt_iou_num, iou_num = t_iou(output['relevance'], batch['tm'], output['offset'], batch['tm_l'], batch['tm_ind'],batch['t_gt']) 

      
      try:
        loss = loss.mean()
        if phase == 'train':
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
      except:
        pass
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      accuracy_stat_3.update(acc_3, num)
      accuracy_stat_5.update(acc_5, num)
      accuracy_rel.update(acc_rel, rel_num)
      gt_iou.update(gt_iou_stat, gt_iou_num)
      pred_iou.update(iou_stat, iou_num)
      Bar.suffix = Bar.suffix + '|{} {:.2f} '.format('acc@3', accuracy_stat_3.avg)
      Bar.suffix = Bar.suffix + '|{} {:.2f} '.format('acc@5', accuracy_stat_5.avg)
      Bar.suffix = Bar.suffix + '|{} {:.2f} '.format('acc_rel', accuracy_rel.avg)
      # Bar.suffix = Bar.suffix + '|{} {:.2f} '.format('acc_rel_neg', accuracy_rel_neg.avg)
      # Bar.suffix = Bar.suffix + '|{} {:.2f} '.format('acc_rel_pos', accuracy_rel_pos.avg)
      for l in avg_loss_stats:
        try:
          avg_loss_stats[l].update(
            loss_stats[l].mean().item(), batch['input'].size(0))
        except:
          avg_loss_stats[l].update(
            loss_stats[l], 0)
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      Bar.suffix = Bar.suffix + '|{} {:.2f} '.format('gt_iou', gt_iou.avg)
      Bar.suffix = Bar.suffix + '|{} {:.2f} '.format('pred_iou', pred_iou.avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)

      
      del output, loss, loss_stats
      # for obj in gc.get_objects():
      #   try:
      #       if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
      #           print(type(obj), obj.size())
      #   except:
      #       pass
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    ret['acc@3'] = accuracy_stat_3.avg
    ret['acc@5'] = accuracy_stat_5.avg
    ret['acc_rel'] = accuracy_rel.avg
    ret['gt_iou'] = gt_iou.avg
    ret['pred_iou'] = pred_iou.avg
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)