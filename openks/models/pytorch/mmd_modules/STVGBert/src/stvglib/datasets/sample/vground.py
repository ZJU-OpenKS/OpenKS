from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, gaussian_temproal_radius, draw_se_map
from utils.image import draw_dense_reg
import math

class VGroundDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    num_frames = 64
    num_classes = self.num_classes
    img_id = self.images[index]
    frame_count = img_id['frame_count']
    used_s = img_id['used_segment']['begin_fid']
    used_e = img_id['used_segment']['end_fid']
    seg_length = used_e - used_s
    gap = seg_length/(num_frames-1)
    fid_s = img_id['temporal_gt']['begin_fid']
    fid_e = img_id['temporal_gt']['end_fid']
    tc = ((fid_e + fid_s)/2 - used_s)/seg_length * num_frames
    # gt_flag=False
    # r_s_s = max(1,fid_s-num_frames*gap+1)
    # r_s_e = max(min(fid_s, frame_count-num_frames*gap+1),1)
    # r_e_s = max(1, fid_e-num_frames*gap+1)
    # r_e_e = max(min(frame_count-num_frames*gap+1, fid_e),1)
    # if fid_e-fid_s > (num_frames-1)*gap :
    #   mid_id = np.random.randint(fid_s,fid_e-(num_frames-1)*gap+1)
    # else:
    #   mid_id = fid_s
    # print(r_s,r_e, fid_s-num_frames*gap+1, frame_count-num_frames*gap+1, fid_e)
    # frame_id_candidates = []
    # frame_id_candidates + ([np.random.randint(fid_s+1 - 8*(num_frames-1), min(fid_s+1, fid_e-1 - 8*(num_frames-1)))] * 3)
    # frame_id_candidates.append(np.random.randint(fid_s+1, max(fid_e-1 - 8*(num_frames-1), fid_s+2)))
    # frame_id_candidates+([np.random.randint(max(fid_s+2, fid_e-1 - 8*(num_frames-1)), fid_e-1)]*3)
    # frame_id_neg_cadidates = []
    # frame_id_neg_cadidates.append(np.random.randint(fid_s-8*(num_frames-1), fid_s+1))
    # frame_id_neg_cadidates.append(np.random.randint(fid_e, frame_count+1))
    # frame_id_candidates.append(frame_id_neg_cadidates[np.random.randint(0, len(frame_id_neg_cadidates))])
    
    # frame_id_candidates.append(np.random.randint(r_s_s, r_s_e+1))
    # frame_id_candidates.append(np.random.randint(r_e_s, r_e_e+1))
    # frame_id_candidates.append(mid_id)

    # frame_id_init = frame_id_candidates[np.random.randint(0,3)]
    # frame_id_init = max(1, frame_id_init)
    # frame_id_init = min(fid_s+1, frame_id_init)
    # if fid_e-fid_s <20:
    #   frame_id_init = fid_s+1
    frame_id_init = used_s+1
    frame_id_list = [int(frame_id_init+j*gap) for j in range(num_frames)]
    # for f_id in frame_id_list:
    #   if f_id in range(fid_s+1, fid_e):
    #     gt_flag=True
    #     break
    # print(frame_id_candidates, frame_id, fid_s, fid_e)
    # if len(img_id['questions']) > 0:
    #   cap_idx = np.random.randint(0,len(img_id['questions']))
    #   sent = img_id['questions'][cap_idx]
    #   id_padding = len(img_id['captions'])
    # else:
    #   cap_idx = np.random.randint(0,len(img_id['captions']))
    #   sent = img_id['captions'][cap_idx]
    #   id_padding=0

    cap_idx = np.random.randint(0,len(img_id['captions']))
    sent = img_id['captions'][cap_idx]
    id_padding=0


    tokens = self.entries[index]['token'][id_padding+cap_idx]
    input_mask = self.entries[index]['input_mask'][id_padding+cap_idx]
    segment_ids = self.entries[index]['segment_ids'][id_padding+cap_idx]
    input_h, input_w = self.opt.input_h, self.opt.input_w
    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio


    
    

    contain_gt = False
    gt_num = 0

    # while not contain_gt:
    gt_num+=1
    frame_id = frame_id_init
    inp_list = []
    c0_factor = np.random.randn()
    c1_factor = np.random.randn()
    s_factor = np.random.randn()
    flip_flag = np.random.random()

    cs = 0
    ce = 0

    hm = np.zeros((num_frames, output_h, output_w), dtype=np.float32)
    tm = np.zeros((1,num_frames), dtype=np.float32)
    tm_offset = np.zeros((2,num_frames), dtype=np.float32)
    wh = np.zeros((num_frames, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((num_frames, 2), dtype=np.float32)
    ind = np.zeros((num_frames), dtype=np.int64)
    tm_ind = np.zeros((2), dtype=np.int64)
    reg_mask = np.zeros((num_frames), dtype=np.uint8)
    cat_spec_wh = np.zeros((num_frames, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((num_frames, num_classes * 2), dtype=np.uint8)
    tm_l = np.zeros((2), dtype=np.float32)
    tm_mask = np.zeros((2), dtype=np.uint8)
    gt_det = np.zeros((num_frames,6), dtype=np.float32)

    tm[0,int(tc)]=1
    rad = gaussian_temproal_radius((fid_e-fid_s)/gap)
    rad = max(0, int(rad))
    draw_se_map(tm[0], int(tc), rad)
    tm_l[0] = (fid_e - fid_s)/seg_length*num_frames
    tm_l[1] = tc - int(tc)
    tm_ind[0] = int(tc)
    tm_ind[1] = int(tc) + num_frames
    # print(seg_length, gap, (fid_e-fid_s)/gap, rad, tm[0])
    
    for t in range(num_frames):
      frame_id = frame_id_list[t]
      frame_id = max(frame_id, 1)
      frame_id = min(frame_id, frame_count)

      # tm_offset[0,t] = (frame_id-1-fid_s)/float(seg_length)
      # tm_offset[1,t] = (fid_s-frame_id-1)/float(seg_length)

      # print(frame_id)
      img_path = os.path.join(self.img_dir, img_id['vid'], '{:04d}.jpg'.format(frame_id))
      anns = self.annots[img_id['vid']]['trajectories'][frame_id-1]
    # num_objs = min(len(anns), self.max_objs)

    # print(img_path)
      if os.path.isfile(img_path):
        img = cv2.imread(img_path)
      else:
        print(img_path +' dose not exsits')

    # img_temp = img.copy()

      try:
        height, width = img.shape[0], img.shape[1]
      except:
        print(img_path)
        raise
      c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
      if self.opt.keep_res:
        input_h = (height | self.opt.pad) + 1
        input_w = (width | self.opt.pad) + 1
        s = np.array([input_w, input_h], dtype=np.float32)
      else:
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.opt.input_h, self.opt.input_w
      
      flipped = False
      if self.split == 'train':
        if not self.opt.not_rand_crop:
          s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
          w_border = self._get_border(128, img.shape[1])
          h_border = self._get_border(128, img.shape[0])
          c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
          c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        else:
          sf = self.opt.scale
          cf = self.opt.shift
          c[0] += s * np.clip(c0_factor*cf, -2*cf, 2*cf)
          c[1] += s * np.clip(c1_factor*cf, -2*cf, 2*cf)
          s = s * np.clip(s_factor*sf + 1, 1 - sf, 1 + sf)
        
        if flip_flag < self.opt.flip:
          flipped = True
          img = img[:, ::-1, :]
          c[0] =  width - c[0] - 1
        

      trans_input = get_affine_transform(
        c, s, 0, [input_w, input_h])
      inp = cv2.warpAffine(img, trans_input, 
                           (input_w, input_h),
                           flags=cv2.INTER_LINEAR)
      # inp_temp = inp.copy()
      inp = (inp.astype(np.float32) / 255.)
      if self.split == 'train' and not self.opt.no_color_aug:
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
      inp = (inp - self.mean) / self.std
      inp = inp.transpose(2, 0, 1)

      
      num_classes = self.num_classes
      trans_output = get_affine_transform(c, s, 0, [output_w, output_h])


   #  n_a = {'description': 'n/a',
   # 'type': 'n/a',
   # 'target_id': 99}
   #  sents = [n_a, n_a, n_a]
   #  num_sents = 0
   #  sents_list = img_id['captions'] #+ img_id['questions']
   #  # if len(img_id['questions'])==0:
   #  #   sents_list=img_id['captions']
   #  # else:
   #  #   sents_list=img_id['questions']
   #  for i, sent in enumerate(sents_list):
   #    sents[i] = sent
   #    num_sents +=1
    
      draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

      # gt_det = []
      
      # word_idx_list = []

      # hm_list = []
      # wh_list = []
      # dense_wh_list = []
      # reg_list = []
      # ind_list = []
      # reg_mask_list = []
      # cat_spec_wh_list = []
      # cat_spec_mask_list = []


    

    # for k in range(num_sents):
      # words = sents[k]['description'][:-1].split(' ')
      # word_idx = []
      # for word in words:
      #   word_idx.append(self.voc[word])
      # word_idx.append(self.voc[sents[k]['description'][-1]])
      # word_idx_list.append(word_idx)


      # print(fid_s, fid_e)
      # print(frame_id_list)
      if frame_id in range(fid_s+1, fid_e):
        # gt_num +=1
        for j in range(len(anns)):
          if anns[j]['tid'] == sent['target_id']:
            ann = anns[j]
        bbox = np.array([ann['bbox']['xmin'],ann['bbox']['ymin'],ann['bbox']['xmax'],ann['bbox']['ymax']], dtype=np.float32)
        # dcpt = sents[k]['description']
        if flipped:
          bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h > 0 and w > 0:

          # if (frame_id -1) in (range(fid_s, fid_s+gap-1)):
          #   cs=t
          #   tm[0,cs] = 1.
          #   rad = gaussian_temproal_radius((fid_e-fid_s)/gap)
          #   rad = max(0, int(rad))
          #   draw_se_map(tm[0], cs, rad)
          #   tm_l[0] = (fid_e-frame_id+1)/float(64*gap)
          #   tm_mask[0] = 1
          #   tm_ind[0]=t
          #   # print(fid_s, fid_e, rad)
          #   # print(frame_id_list)
          #   # print(tm)
          # if frame_id in  range(fid_e-gap+1, fid_e):
          #   ce = t
          #   tm[1,ce] = 1.
          #   rad = gaussian_temproal_radius((fid_e-fid_s)/gap)
          #   rad = max(0, int(rad))
          #   draw_se_map(tm[1], ce, rad)
          #   tm_l[1] = (frame_id - 1 - fid_s)/float(64*gap)
          #   tm_mask[1] = 1
          #   if tm_ind[0]==0:
          #     tm_ind[1]=t+num_frames
            # print(fid_s, fid_e, rad)
            # print(frame_id_list)
            # print(tm)
          radius = gaussian_radius((math.ceil(h), math.ceil(w)))
          radius = max(0, int(radius))
          radius = self.opt.hm_gauss if self.opt.mse_loss else radius
          ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
          ct_int = ct.astype(np.int32)
          draw_gaussian(hm[t], ct_int, radius)
          wh[t] = 1. * w, 1. * h
          ind[t] = ct_int[1] * output_w + ct_int[0]
          reg[t] = ct - ct_int
          reg_mask[t] = 1
          cat_spec_wh[t,:] = wh[t]
          cat_spec_mask[t,:] = 1
          # hm_list.append(hm)
          # wh_list.append(wh)
          # ind_list.append(ind)
          # reg_list.append(reg)
          # reg_mask_list.append(reg_mask)
          # cat_spec_wh_list.append(cat_spec_wh)
          # cat_spec_mask_list.append(cat_spec_mask)
          if self.opt.dense_wh:
            draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[t], radius)
            dense_wh_list.append(dense_wh)
          gt_det[t,:]=np.array([ct[0] - w / 2, ct[1] - h / 2, 
                         ct[0] + w / 2, ct[1] + h / 2, 1, t], dtype=np.float32)

      inp_list.append(inp)

    # contain_gt = reg_mask.any() or (not gt_flag)
    # if gt_num>20000:
    #   print(gt_num)
    #   print(frame_id_init, fid_s+1, fid_e, frame_count)
    #   print(frame_id_candidates, frame_id, fid_s, fid_e)
    #   raise()
      # else:
      #   print('bbox:',bbox)
      #   cv2.imwrite('./{}_ori.jpg'.format(img_id['vid']), img_temp)
      #   cv2.imwrite('./{}_tran.jpg'.format(img_id['vid']), inp_temp)

    # print(sents, num_sents, reg_mask)
    
    inp = np.stack(inp_list,axis=0)
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'frame_id_init':np.array([frame_id_init]), 't_gt': np.array([(fid_s-used_s)/gap,(fid_e-used_s)/gap]),'tm_ind':tm_ind, 'tm_l':tm_l, 'tm_mask':tm_mask, 'wh': wh, 'tm':tm, 'tm_offset': tm_offset, 'tokens': tokens, 'input_mask': input_mask, 'segment_ids': segment_ids}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    # gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
    #            np.zeros((1, 6), dtype=np.float32)
    ret.update({'gt_det': gt_det})
    if self.opt.debug > 0 or not self.split == 'train':
      # gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
      #          np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret