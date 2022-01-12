# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

from .lang_feats import LangFeats
from .vilbert import VILBertForVGROUND, BertConfig

import torch
import torch.nn as nn
from .DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo

import copy

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, bert_config_path, bert_pretrained_model_path):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        config = BertConfig.from_json_file(bert_config_path)
        # print('ddd', os.getcwd())
        self.bert = VILBertForVGROUND.from_pretrained(bert_pretrained_model_path, config)
        # self.bert = VILBertForVGROUND(config)

        # self.lf = LangFeats()

        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.tv_3 = nn.ModuleList([nn.Sequential(*[nn.Conv1d(128, 128, kernel_size=3,padding=1, bias=False),
        nn.ReLU(inplace=True), 
        nn.Conv1d(128, 128, kernel_size=3,padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv1d(128, 1, kernel_size=3,padding=1, bias=False)])]*5)

        self.tv_4 = nn.ModuleList([nn.Sequential(*[nn.Conv1d(128, 128, kernel_size=3,padding=1, bias=False),
        nn.ReLU(inplace=True), 
        nn.Conv1d(128, 128, kernel_size=3,padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv1d(128, 2, kernel_size=3,padding=1, bias=False)])]*5)

        # self.t1 = nn.Conv1d(config.v_hidden_size, 512, stride=2,kernel_size=3,padding=1, bias=False)
        # self.t2 = nn.Conv1d(512, 256, stride=2,kernel_size=3,padding=1, bias=False)
        # self.t3 = nn.Conv1d(256, 128, stride=2,kernel_size=3,padding=1, bias=False)
        # self.dt1 = nn.ConvTranspose1d(128,128,kernel_size=4,padding=1,output_padding=0,stride=2)
        # self.dt2 = nn.ConvTranspose1d(128,128,kernel_size=4,padding=1,output_padding=0,stride=2)
        # self.dt3 = nn.ConvTranspose1d(128,128,kernel_size=4,padding=1,output_padding=0,stride=2)
        # self.ds1 = nn.Conv1d(config.v_hidden_size, 128,kernel_size=3,padding=1, bias=False)
        # self.ds2 = nn.Conv1d(512, 128,kernel_size=3,padding=1, bias=False)
        # self.ds3 = nn.Conv1d(256, 128,kernel_size=3,padding=1, bias=False)




        # self.Conv3d_1 = nn.Conv3d(config.v_hidden_size, 256, kernel_size=3,padding=1, bias=False)

        # self.Conv3d_2 = nn.Conv3d(256, 256, kernel_size=3,padding=1, bias=False)

        # self.Conv3d_3 = nn.Conv3d(256, 128, kernel_size=3,padding=1, bias=False)

        self.max = nn.MaxPool3d(kernel_size=(1,8,8))

        self.avg_list = nn.ModuleList([nn.AvgPool1d(kernel_size=2**(n+2)-1, padding=2**(n+1)-1, stride=1) for n in range(5)])

        self.query = nn.Linear(config.hidden_size, 128)

        self.key = nn.Linear(config.v_hidden_size, 128)

        self.value = nn.Linear(config.v_hidden_size, 128)

        # self.mapping3 = nn.Linear(config.hidden_size, 64)

        self.headconv = nn.Conv2d(64, 256,
                    kernel_size=3, padding=1, bias=True)

        # used for deconv layers
        self.inplanes = int(self.inplanes/2)
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        self.deconv_layers2 = copy.deepcopy(self.deconv_layers)

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, sents, text_mask, test=False):
        # langfeats, sents_per_image, sents_start_idx = self.lf(sents)

    
        batch, num, input_c, input_h, input_w = x.size()
        # print('input_size', x.size())

        x = x.view(-1, input_c, input_h, input_w)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        _, feat_dim, h, w = x.size()

        v_in = x.permute(0,2,3,1).contiguous().view(batch, num,-1,feat_dim)

        # x_spatial = x.mean(1)
        # x_temporal = x.mean(2)

        loc = torch.zeros(batch,num,h,w,6).float()

        loc[:,:,:,:,0] = (torch.arange(0,num).float()/num).view(1,-1,1,1)
        loc[:,:,:,:,1] = (torch.arange(0,w).float()/w).view(1,1,1,-1)
        loc[:,:,:,:,2] = (torch.arange(0,h).float()/h).view(1,1,-1,1)
        loc[:,:,:,:,3] = 1/w
        loc[:,:,:,:,4] = 1/h
        loc[:,:,:,:,5] = 1/(w*h)

        loc = loc.view(batch, num,-1,6).cuda()

        ret = {}

        if test:
            for head in self.heads:
                ret[head] = []
            ret['relevance'] = []
            ret['offset'] = []
            for i in range(sents.shape[1]):
                sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v = self.bert(sents[:,i,:], v_in, loc, attention_mask=text_mask[:,i,:])
                # global_v = sequence_output_v.mean(-2)

                sequence_output_t = sequence_output_t.unsqueeze(1).expand(batch,num,sequence_output_t.size(-2), sequence_output_t.size(-1)).view(-1,sequence_output_t.size(-2),sequence_output_t.size(-1))

                query = self.query(sequence_output_t[:,:1,:])
                key = self.key(sequence_output_v).view(batch*num,sequence_output_v.size(-2),-1)
                value = self.value(sequence_output_v).view(batch*num,sequence_output_v.size(-2),-1)
                attention_scores = torch.matmul(query, key.transpose(-1,-2))
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                global_v = torch.matmul(attention_probs,value).view(batch,num,-1).permute(0,2,1).contiguous()

                global_v_list = []
                for avglayer in self.avg_list:
                    global_v_list.append(avglayer(global_v))

                cls_scores = []
                for v, t3 in zip(global_v_list, self.tv_3):
                    cls_scores.append(t3(v))
                cls_scores = torch.cat(cls_scores,dim=-1)

                offset = []
                for v, t4 in zip(global_v_list, self.tv_4):
                    offset.append(t4(v))
                offset = torch.cat(offset, dim=-1)

                ret['relevance'].append(cls_scores)

                ret['offset'].append(offset)


                sequence_output_v = sequence_output_v.view(batch*num,h,w,-1).permute(0,3,1,2).contiguous()
         
                # corr_map = torch.bmm(self.headconv(x).permute(0,2,3,1).contiguous().view(x.size(0),-1,265),self.mapping2(sequence_output_t[:,:1,:]).permute(0,2,1).contiguous())
                
                # corr_map = corr_map.view(batch*num, x.size(-2), x.size(-1),-1).permute(0,3,1,2)
                
                # print('out_V', sequence_output_v.size())

                x2 = self.deconv_layers2(self.relu(self.conv2(x)))

                x = self.deconv_layers(sequence_output_v)

                # for head in self.heads:
                #     if head == 'hm':
                #         ret[head]= corr_map
                #     else:
                #         ret[head] = self.__getattr__(head)(x)

                for head in self.heads:
                    if head == 'hm':
                        ret[head].append(self.__getattr__(head)(x))
                    else:
                        ret[head].append(self.__getattr__(head)(x2))

                return [ret]

            global_vt_list = []

            sequence_output_v_list = []

        
        sequence_output_t, sequence_output_v, _, _ = self.bert(sents, v_in, loc, attention_mask=text_mask)

        sequence_output_t = sequence_output_t.unsqueeze(1).expand(batch,num,sequence_output_t.size(-2), sequence_output_t.size(-1)).view(-1,sequence_output_t.size(-2),sequence_output_t.size(-1))

        query = self.query(sequence_output_t[:,:1,:])
        key = self.key(sequence_output_v).view(batch*num,sequence_output_v.size(-2),-1)
        value = self.value(sequence_output_v).view(batch*num,sequence_output_v.size(-2),-1)
        attention_scores = torch.matmul(query, key.transpose(-1,-2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        global_v = torch.matmul(attention_probs,value).view(batch,num,-1).permute(0,2,1).contiguous()

        global_v_list = []
        for avglayer in self.avg_list:
            global_v_list.append(avglayer(global_v))

        cls_scores = []
        for v, t3 in zip(global_v_list, self.tv_3):
            cls_scores.append(t3(v))
        cls_scores = torch.cat(cls_scores,dim=-1)

        offset = []
        for v, t4 in zip(global_v_list, self.tv_4):
            offset.append(t4(v))
        offset = torch.cat(offset, dim=-1)

        ret['relevance'] = cls_scores

        ret['offset'] = offset

        sequence_output_v = sequence_output_v.view(batch*num,h,w,-1).permute(0,3,1,2).contiguous()
 
        # print('out_V', sequence_output_v.size())

        
        x2 = self.deconv_layers2(self.relu(self.conv2(x)))

        x = self.deconv_layers(sequence_output_v)

        for head in self.heads:
            if head == 'hm':
                # ret[head]= corr_map
                ret[head] = self.__getattr__(head)(x)
            else:
                with torch.no_grad():
                    ret[head] = self.__getattr__(head)(x2)
        # for head in self.heads:
        #     ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            # pretrained_state_dict = torch.load('/home/rusu5516/project/video_grounding/ctdet_coco_resdcn101.pth')
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=256, bert_config_path='', bert_pretrained_model_path=''):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv, bert_config_path=bert_config_path, bert_pretrained_model_path=bert_pretrained_model_path)
  model.init_weights(num_layers)
  return model
