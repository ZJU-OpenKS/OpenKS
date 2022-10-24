from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .init_paths import *

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from .stvglib.external.nms import soft_nms
from .stvglib.opts import opts
from .stvglib.logger import Logger
from .stvglib.utils.utils import AverageMeter
from .stvglib.datasets.dataset_factory import dataset_factory
from .stvglib.detectors.detector_factory import detector_factory
import pickle

no_question_list = [5, 42, 100, 122, 151, 201, 207, 219, 234, 246, 274, 284, 310, 357, 360, 375, 376, 383, 384, 440, 461, 464, 466, 469, 488, 513, 514, 528, 556, 557, 558, 569, 589, 608, 612, 618, 672, 689, 690, 767, 801, 811, 819, 932, 933, 934, 951, 952, 994, 1031, 1114, 1119, 1139, 1140, 1152, 1158, 1204, 1210, 1242, 1243, 1244, 1250, 1255, 1263, 1271, 1296, 1313, 1314, 1322, 1337, 1338, 1339, 1360, 1370, 1477, 1491, 1535, 1547, 1589, 1607, 1671, 1685, 1689, 1731, 1741, 1748, 1778, 1779, 1780, 1781, 1782, 1787, 1791, 1800, 1814, 1861, 1890, 1901, 1902, 1947, 1948, 1954, 1974, 1975, 1976, 1977, 1997, 2059, 2072, 2077, 2147, 2166, 2174, 2175, 2176, 2258, 2265, 2301, 2305, 2372, 2418, 2446, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2483, 2501, 2502, 2503, 2586, 2592, 2625, 2626, 2627, 2659, 2681, 2696, 2700, 2733, 2754, 2757, 2776, 2783, 2797, 2810, 2811, 2813, 2816, 2841, 2905, 2964, 2989, 3002, 3027, 3028, 3029, 3030, 3116, 3173, 3183, 3191, 3195, 3244, 3248, 3262, 3271, 3291, 3293, 3306, 3334, 3340, 3358, 3417, 3422, 3457, 3458, 3470, 3471, 3472, 3480, 3490, 3526, 3539, 3568, 3572, 3666, 3741, 3787, 3817, 3829, 3841, 3890, 3938, 3939, 3940, 3984, 4004, 4006, 4056, 4097, 4104, 4111, 4127, 4141, 4164, 4182, 4188, 4231, 4248, 4314, 4315, 4316, 4317, 4318, 4383, 4446, 4513, 4594]

test_set = 'cap'

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.entries = dataset.entries
    # self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    if test_set == 'cap' or len(img_id['questions'])==0:
      text_num = len(img_id['captions'])
      tokens = self.entries[index]['token'][:text_num,:]
      text_mask = self.entries[index]['input_mask'][:text_num,:]
    else:
      text_num = len(img_id['captions']+img_id['questions'])
      cap_num = len(img_id['captions'])
      tokens = self.entries[index]['token'][cap_num:text_num,:]
      text_mask = self.entries[index]['input_mask'][cap_num:text_num,:]
    # img_info = self.load_image_func(ids=[img_id])[0]
    f_s = img_id['used_segment']['begin_fid']
    f_e = img_id['used_segment']['end_fid']
    num_frames = 64
    gap = (f_e-f_s+1)/(num_frames-1)
    images, meta = {}, {}
    image_list = []
    meta_list = []

    for scale in opt.test_scales:
      for i in range(num_frames):
        f_id = int(f_s + i*gap)+1
        f_id = min(f_id, img_id['frame_count'])
        img_path = os.path.join(self.img_dir, img_id['vid'], '{:04d}.jpg'.format(f_id))
        image = cv2.imread(img_path)
        try:
          _,_,_ = image.shape
        except:
          print(img_path)
          raise()
        if opt.task == 'ddd':
          images[scale], meta[scale] = self.pre_process_func(
            image, scale, img_info['calib'])
        else:
          img, meta_img = self.pre_process_func(image, scale)
          image_list.append(img)
          meta_list.append(meta_img)
          # images[scale], meta[scale] = self.pre_process_func(image, scale)
      images[scale] = torch.cat(image_list,dim=0)
      meta[scale] = meta_list
    return img_id, tokens, text_mask, {'images': images, 'image': image_list, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = []
  rel = []
  offset = []
  output = []
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, tokens, text_mask, pre_processed_images) in enumerate(data_loader):
    # if img_id['da_id'] not in no_question_list or test_set == 'cap':
    ret = detector.run(pre_processed_images, tokens, text_mask)

    # da_id = img_id['da_id'].numpy()[0]
    # if da_id not in results.keys():
    #   results[da_id] = []
    # if da_id not in rel.keys():
    #   rel[da_id] = []
    # if da_id not in output.keys():
    #   output[da_id] = []
    # if da_id not in offset.keys():
    #   offset[da_id] = []
    results.append(ret['results'])
    rel.append(ret['rel'])
    # output[da_id].append(ret['output'])
    offset.append(ret['offset'])

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])

  # with open('outputs/output_{}_{}.pkl'.format(opt.exp_id,str(da_id)), 'wb') as f:
  #   pickle.dump(output[da_id], f)
    bar.next()
  bar.finish()
  with open('results_{}.pkl'.format(opt.exp_id), 'wb') as f:
    pickle.dump(results, f)

  with open('rel_{}.pkl'.format(opt.exp_id), 'wb') as f:
    pickle.dump(rel, f)

  with open('offset_{}.pkl'.format(opt.exp_id), 'wb') as f:
    pickle.dump(offset, f)

  # dataset.run_eval(results, opt.save_dir)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)


