import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo

import matplotlib
from matplotlib.offsetbox import AnchoredText
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time, math
import copy
import os,errno
# import bisect
from operator import itemgetter
from .models.discriminator import *
import .datasets.imagefolder
import argparse
from PIL import Image, ImageDraw,ImageFont
torch.autograd.set_detect_anomaly(True)
######################################################################
# Prepara Parameters

parser = argparse.ArgumentParser(description='Domain Adaptation')
parser.add_argument('--data_dir', type=str, default='./data/office31')
parser.add_argument('--source_set', type=str, default='amazon')
parser.add_argument('--target_set', type=str, default='webcam')

parser.add_argument('--gpu', type=str, default='2')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_class', type=int, default=31)
parser.add_argument('--base_lr', type=float, default=0.0015)

parser.add_argument('--pretrain_sample', type=int, default=50000)
parser.add_argument('--train_sample', type=int, default=200000)

parser.add_argument('--form_w', type=float, default=0.4)
parser.add_argument('--main_w', type=float, default=-0.8)

parser.add_argument('--wp', type=float, default=0.055)
parser.add_argument('--wt', type=float, default=1)

parser.add_argument('--select', type=str, default='1-2')

parser.add_argument('--usePreT2D', type=bool, default=False)

parser.add_argument('--useT1DorT2', type=str, default="T2")

parser.add_argument('--diffS', type=bool, default=False)

parser.add_argument('--diffDFT2', type=bool, default=False)

parser.add_argument('--useT2CompD', type=bool, default=False)
parser.add_argument('--usemin', type=bool, default=False)

parser.add_argument('--useRatio', type=bool, default=False)

parser.add_argument('--useCurrentIter', type=bool, default=False)
# parser.add_argument('--useEpoch', type=bool, default=False)

parser.add_argument('--useLargeLREpoch', type=bool, default=True)

parser.add_argument('--MaxStep', type=int, default=0)

parser.add_argument('--useSepTrain', type=bool, default=True)

parser.add_argument('--fixW', type=bool, default=False)
parser.add_argument('--decay', type=float, default=0.0003)
parser.add_argument('--nesterov', type=bool, default=True)

parser.add_argument('--ReTestSource', type=bool, default=False)

parser.add_argument('--sourceTestIter', type=int, default=2000)
parser.add_argument('--defaultPseudoRatio', type=float, default=0.2)
parser.add_argument('--totalPseudoChange', type=int, default=100)

parser.add_argument('--beta', type=float, default=1.0)

args = parser.parse_args()

data_dir = '/data2/tianjiayi/dataset/Office31/'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

