# Copyright (c) 2021 OpenKS Authors, Visual Computing Group, Beihang University.
# All rights reserved.
import os
import sys
import logging
import argparse
import datetime
import time
import json
from tqdm import tqdm
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data

import cv2
from progress.bar import Bar
import pickle

from ..model import VisualConstructionModel


        
# Add lib to PYTHONPATH
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'mmd_modules/STVGBert/src/lib')
add_path(lib_path)
print(lib_path)
##train_related
from .mmd_modules.STVGBert.src.lib.opts import opts
from .mmd_modules.STVGBert.src.lib.models.model import create_model, load_model, save_model
from .mmd_modules.STVGBert.src.lib.models.data_parallel import DataParallel
from .mmd_modules.STVGBert.src.lib.logger import Logger
from .mmd_modules.STVGBert.src.lib.datasets.dataset_factory import get_dataset
from .mmd_modules.STVGBert.src.lib.trains.train_factory import train_factory

##evaluate_related
# from .mmd_modules.STVGBert.src.lib.external.nms import soft_nms
# from .mmd_modules.STVGBert.src.lib.utils.utils import AverageMeter
# from .mmd_modules.STVGBert.src.lib.datasets.dataset_factory import dataset_factory
# from .mmd_modules.STVGBert.src.lib.detectors.detector_factory import detector_factory
from .mmd_modules.STVGBert.src.test import test, prefetch_test


@VisualConstructionModel.register("STVGBert", "PyTorch")
class VisualRelationTorch(VisualConstructionModel):
    # TODO distributed learning is not complete.
    def __init__(self, name: str = 'pytorch-default', use_distributed: bool = False, args = None):
        self.name = name
        self.opt = self.parse_args(args)
        print("sss")
        print(self.opt)
   
    
    def parse_args(self, args):
        print("make parser")
        parser = argparse.ArgumentParser(description="STVGBert Model")
        parser.add_argument('--stvgbert', action='store_true',
                                help="A flag denotes the STVGBert training")
        # basic experiment setting
        parser.add_argument('--task', default='ctdet',
                                help='ctdet | ddd | multi_pose | exdet')
        parser.add_argument('--dataset', default='vidstg',
                                help='coco | kitti | coco_hp | pascal')
        parser.add_argument('--exp_id', default='vidstg_svgnet_resize_7')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--debug', type=int, default=0,
                                help='level of visualization.'
                                    '1: only show the final detection results'
                                    '2: show the network output features'
                                    '3: use matplot to display' # useful when lunching training with ipython notebook
                                    '4: save all visualizations to disk')
        parser.add_argument('--demo', default='', 
                                help='path to image/ image folders/ video. '
                                    'or "webcam"')
        parser.add_argument('--load_model', default='',
                                help='path to pretrained model')
        parser.add_argument('--resume', action='store_true',
                                help='resume an experiment. '
                                    'Reloaded the optimizer parameter and '
                                    'set load_model to model_last.pth '
                                    'in the exp dir if load_model is empty.') 

        # system
        parser.add_argument('--gpus', default='3', 
                                help='-1 for CPU, use comma for multiple gpus')
        parser.add_argument('--num_workers', type=int, default=4,
                                help='dataloader threads. 0 for single-thread.')
        parser.add_argument('--not_cuda_benchmark', action='store_true',
                                help='disable when the input size is not fixed.')
        parser.add_argument('--seed', type=int, default=317, 
                                help='random seed') # from CornerNet

        # log
        parser.add_argument('--print_iter', type=int, default=0, 
                                help='disable progress bar and print to screen.')
        parser.add_argument('--hide_data_time', action='store_true',
                                help='not display time during training.')
        parser.add_argument('--save_all', action='store_true',
                                help='save model to disk every 5 epochs.')
        parser.add_argument('--metric', default='loss', 
                                help='main metric to save best model')
        parser.add_argument('--vis_thresh', type=float, default=0.3,
                                help='visualization threshold.')
        parser.add_argument('--debugger_theme', default='white', 
                                choices=['white', 'black'])
        
        # model
        parser.add_argument('--arch', default='resdcn_101', 
                                help='model architecture. Currently tested'
                                    'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                    'dlav0_34 | dla_34 | hourglass')
        parser.add_argument('--head_conv', type=int, default=-1,
                                help='conv layer channels for output head'
                                    '0 for no conv layer'
                                    '-1 for default setting: '
                                    '64 for resnets and 256 for dla.')
        parser.add_argument('--down_ratio', type=int, default=4,
                                help='output stride. Currently only supports 4.')
        parser.add_argument('--bert_config_path', type=str, default='/data/sunweiqi/OpenKS/openks/models/pytorch/mmd_modules/STVGBert/src/bert_base_6layer_6conect.json',
                                help='Bert Config Path.')
        parser.add_argument('--bert_pretrained_model_path', type=str, default='/data/sunweiqi/OpenKS/openks/models/pytorch/mmd_modules/STVGBert/pytorch_model_8.bin',
                                help='Bert Pretrained Model Path.')

        # input
        parser.add_argument('--input_res', type=int, default=256, 
                                help='input height and width. -1 for default from '
                                'dataset. Will be overriden by input_h | input_w')
        parser.add_argument('--input_h', type=int, default=-1, 
                                help='input height. -1 for default from dataset.')
        parser.add_argument('--input_w', type=int, default=-1, 
                                help='input width. -1 for default from dataset.')
        
        # train
        parser.add_argument('--lr', type=float, default=1e-5, 
                                help='learning rate for batch size 32.')
        parser.add_argument('--lr_step', type=str, default='10',
                                help='drop learning rate by 10.')
        parser.add_argument('--num_epochs', type=int, default=15,
                                help='total training epochs.')
        parser.add_argument('--batch_size', type=int, default=1,
                                help='batch size')
        parser.add_argument('--master_batch_size', type=int, default=1,
                                help='batch size on the master gpu.')
        parser.add_argument('--num_iters', type=int, default=-1,
                                help='default: #samples / batch_size.')
        parser.add_argument('--val_intervals', type=int, default=1,
                                help='number of epochs to run validation.')
        parser.add_argument('--trainval', action='store_true',
                                help='include validation in training and '
                                    'test on test set')

        # test
        parser.add_argument('--flip_test', action='store_true',
                                help='flip data augmentation.')
        parser.add_argument('--test_scales', type=str, default='1',
                                help='multi scale test augmentation.')
        parser.add_argument('--nms', action='store_true',
                                help='run nms in testing.')
        parser.add_argument('--K', type=int, default=1,
                                help='max number of output objects.') 
        parser.add_argument('--not_prefetch_test', action='store_true',
                                help='not use parallal data pre-processing.')
        parser.add_argument('--fix_res', action='store_true',
                                help='fix testing resolution or keep '
                                    'the original resolution')
        parser.add_argument('--keep_res', action='store_true',
                                help='keep the original resolution'
                                    ' during validation.')
        parser.add_argument('--test_name', type=str, default='',
                                help='test result name.')

        # dataset
        parser.add_argument('--not_rand_crop', action='store_true', default=True,
                                help='not use the random crop data augmentation'
                                    'from CornerNet.')
        parser.add_argument('--shift', type=float, default=0.1,
                                help='when not using random crop'
                                    'apply shift augmentation.')
        parser.add_argument('--scale', type=float, default=0.4,
                                help='when not using random crop'
                                    'apply scale augmentation.')
        parser.add_argument('--rotate', type=float, default=0,
                                help='when not using random crop'
                                    'apply rotation augmentation.')
        parser.add_argument('--flip', type = float, default=0.5,
                                help='probability of applying flip augmentation.')
        parser.add_argument('--no_color_aug', action='store_true', default=True,
                                help='not use the color augmenation '
                                    'from CornerNet')
        parser.add_argument('--data_dir', type = str, default='/data2/yangming/datasets/',
                                help='dataset path')
        # multi_pose
        parser.add_argument('--aug_rot', type=float, default=0, 
                                help='probability of applying '
                                    'rotation augmentation.')
        # ddd
        parser.add_argument('--aug_ddd', type=float, default=0.5,
                                help='probability of applying crop augmentation.')
        parser.add_argument('--rect_mask', action='store_true',
                                help='for ignored object, apply mask on the '
                                    'rectangular region or just center point.')
        parser.add_argument('--kitti_split', default='3dop',
                                help='different validation split for kitti: '
                                    '3dop | subcnn')

        # loss
        parser.add_argument('--mse_loss', action='store_true',
                                help='use mse loss or focal loss to train '
                                    'keypoint heatmaps.')
        # ctdet
        parser.add_argument('--reg_loss', default='l1',
                                help='regression loss: sl1 | l1 | l2')
        parser.add_argument('--hm_weight', type=float, default=1,
                                help='loss weight for keypoint heatmaps.')
        parser.add_argument('--off_weight', type=float, default=1,
                                help='loss weight for keypoint local offsets.')
        parser.add_argument('--wh_weight', type=float, default=0.1,
                                help='loss weight for bounding box size.')
        parser.add_argument('--rel_weight', type=float, default=1,
                                help='loss rel for bounding box size.')
        # multi_pose
        parser.add_argument('--hp_weight', type=float, default=1,
                                help='loss weight for human pose offset.')
        parser.add_argument('--hm_hp_weight', type=float, default=1,
                                help='loss weight for human keypoint heatmap.')
        # ddd
        parser.add_argument('--dep_weight', type=float, default=1,
                                help='loss weight for depth.')
        parser.add_argument('--dim_weight', type=float, default=1,
                                help='loss weight for 3d bounding box size.')
        parser.add_argument('--rot_weight', type=float, default=1,
                                help='loss weight for orientation.')
        parser.add_argument('--peak_thresh', type=float, default=0.2)
        
        # task
        # ctdet
        parser.add_argument('--norm_wh', action='store_true',
                                help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        parser.add_argument('--dense_wh', action='store_true',
                                help='apply weighted regression near center or '
                                    'just apply regression on center point.')
        parser.add_argument('--cat_spec_wh', action='store_true',
                                help='category specific bounding box size.')
        parser.add_argument('--not_reg_offset', action='store_true',
                                help='not regress local offset.')
        # exdet
        parser.add_argument('--agnostic_ex', action='store_true',
                                help='use category agnostic extreme points.')
        parser.add_argument('--scores_thresh', type=float, default=0.1,
                                help='threshold for extreme point heatmap.')
        parser.add_argument('--center_thresh', type=float, default=0.1,
                                help='threshold for centermap.')
        parser.add_argument('--aggr_weight', type=float, default=0.0,
                                help='edge aggregation weight.')
        # multi_pose
        parser.add_argument('--dense_hp', action='store_true',
                                help='apply weighted pose regression near center '
                                    'or just apply regression on center point.')
        parser.add_argument('--not_hm_hp', action='store_true',
                                help='not estimate human joint heatmap, '
                                    'directly use the joint offset from center.')
        parser.add_argument('--not_reg_hp_offset', action='store_true',
                                help='not regress local offset for '
                                    'human joint heatmaps.')
        parser.add_argument('--not_reg_bbox', action='store_true',
                                help='not regression bounding box size.')
        
        # ground truth validation
        parser.add_argument('--eval_oracle_hm', action='store_true', 
                                help='use ground center heatmap.')
        parser.add_argument('--eval_oracle_wh', action='store_true', 
                                help='use ground truth bounding box size.')
        parser.add_argument('--eval_oracle_offset', action='store_true', 
                                help='use ground truth local heatmap offset.')
        parser.add_argument('--eval_oracle_kps', action='store_true', 
                                help='use ground truth human pose offset.')
        parser.add_argument('--eval_oracle_hmhp', action='store_true', 
                                help='use ground truth human joint heatmaps.')
        parser.add_argument('--eval_oracle_hp_offset', action='store_true', 
                                help='use ground truth human joint local offset.')
        parser.add_argument('--eval_oracle_dep', action='store_true', 
                                help='use ground truth depth.')
        opt = parser.parse_args(args)
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset
        opt.reg_bbox = not opt.not_reg_bbox
        opt.hm_hp = not opt.not_hm_hp
        opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

        if opt.head_conv == -1: # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64
        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        # opt.data_dir = os.path.join('/home/rusu5516/')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)
        
        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                        else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        print(opt)
        return opt

    
    def evaluate(self):
        if self.opt.not_prefetch_test:
            test(self.opt)
        else:
            prefetch_test(self.opt)
    
    def train(self):
        torch.manual_seed(self.opt.seed)
        torch.backends.cudnn.benchmark = not self.opt.not_cuda_benchmark and not self.opt.test
        # torch.backends.cudnn.benchmark = False
        Dataset = get_dataset(self.opt.dataset, self.opt.task)
        self.opt = opts().update_dataset_info_and_set_heads(self.opt, Dataset)
        print(self.opt)

        logger = Logger(self.opt)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str
        self.opt.device = torch.device('cuda' if self.opt.gpus[0] >= 0 else 'cpu')
        
        print('Creating model...')
        model = create_model(self.opt.arch, self.opt.heads, self.opt.head_conv, self.opt.bert_config_path, self.opt.bert_pretrained_model_path)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        large_rate_group = ['hm.','wh.','reg.', 'encoder.LayerNorm', 'deconv_layers', 'convert_2_t']

        optimizer_grouped_parameters = []

        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if any(nd in key for nd in large_rate_group):
                    # if args.learning_rate <= 2e-5:
                    lr = 1e-4
                else:
                    lr = self.opt.lr
                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]
                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]

        
        optimizer = torch.optim.Adam(model.parameters(), self.opt.lr)
        # optimizer = torch.optim.Adam(optimizer_grouped_parameters, opt.lr)
        start_epoch = 0
        if self.opt.load_model != '':
            model, optimizer, start_epoch = load_model(
                model, self.opt.load_model, optimizer, self.opt.resume, self.opt.lr, self.opt.lr_step)

        Trainer = train_factory[self.opt.task]
        trainer = Trainer(self.opt, model, optimizer)
        trainer.set_device(self.opt.gpus, self.opt.chunk_sizes, self.opt.device)

        print('创建数据集')
        val_loader = torch.utils.data.DataLoader(
                Dataset(self.opt, 'val'), 
                batch_size=1, 
                shuffle=False,
                num_workers=1,
                pin_memory=True
        )

        # if self.opt.test:
        #     _, preds = trainer.val(0, val_loader)
        #     val_loader.dataset.run_eval(preds, self.opt.save_dir)
        #     return

        train_loader = torch.utils.data.DataLoader(
                Dataset(self.opt, 'train'), 
                batch_size=self.opt.batch_size, 
                shuffle=True,
                num_workers=self.opt.num_workers,
                pin_memory=True,
                drop_last=True
        )

        print('开始训练')
        best = 1e10
        for epoch in range(start_epoch + 1, self.opt.num_epochs + 1):
            mark = epoch if self.opt.save_all else 'last'
            log_dict_train, _ = trainer.train(epoch, train_loader)
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if self.opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
                save_model(os.path.join(self.opt.save_dir, 'model_{}.pth'.format(mark)), 
                                    epoch, model, optimizer)
                with torch.no_grad():
                    log_dict_val, preds = trainer.val(epoch, val_loader)
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch)
                    logger.write('{} {:8f} | '.format(k, v))
                if log_dict_val[self.opt.metric] < best:
                    best = log_dict_val[self.opt.metric]
                    save_model(os.path.join(self.opt.save_dir, 'model_best.pth'), 
                                        epoch, model)
            else:
                save_model(os.path.join(self.opt.save_dir, 'model_last.pth'), 
                                    epoch, model, optimizer)
            logger.write('\n')
            if epoch in self.opt.lr_step:
                save_model(os.path.join(self.opt.save_dir, 'model_{}.pth'.format(epoch)), 
                                    epoch, model, optimizer)
                lr = self.opt.lr * (0.1 ** (self.opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
                for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
        logger.close()
        print("结束训练")
    
    def run(self, mode="train"):
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.evaluate()
        elif mode == "single":
            raise ValueError("UnImplemented mode!")
    