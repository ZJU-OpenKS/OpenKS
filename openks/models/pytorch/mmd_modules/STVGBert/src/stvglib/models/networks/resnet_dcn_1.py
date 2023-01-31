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
from .vilbert5 import VILBertForVGROUND, BertConfig

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

    def __init__(self, block, layers, heads, head_conv):
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
        config = BertConfig.from_json_file('/home/rusu5516/project/st_grouding/bert_base_6layer_6conect.json')

        self.bert = VILBertForVGROUND.from_pretrained('/home/rusu5516/project/st_grouding/pytorch_model_8.bin', config)
        # self.bert = VILBertForVGROUND(config)

        # self.lf = LangFeats()

        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.tv_1 = nn.Sequential(*[nn.Conv1d(config.v_hidden_size, config.v_hidden_size, kernel_size=3,padding=1, bias=False),
        nn.ReLU(inplace=True), 
        nn.Conv1d(config.v_hidden_size, 512, kernel_size=3,padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv1d(512, 64, kernel_size=3,padding=1, bias=False)])

        self.tv_2 = nn.Sequential(*[nn.Conv1d(config.v_hidden_size, config.v_hidden_size, kernel_size=3,padding=1, bias=False),
        nn.ReLU(inplace=True), 
        nn.Conv1d(config.v_hidden_size, 512, kernel_size=3,padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv1d(512, 64, kernel_size=3,padding=1, bias=False)])

        # self.Conv3d_1 = nn.Conv3d(config.v_hidden_size, 256, kernel_size=3,padding=1, bias=False)

        # self.Conv3d_2 = nn.Conv3d(256, 256, kernel_size=3,padding=1, bias=False)

        # self.Conv3d_3 = nn.Conv3d(256, 128, kernel_size=3,padding=1, bias=False)

        self.max = nn.MaxPool3d(kernel_size=(1,8,8))

        self.avg = nn.AvgPool3d(kernel_size=(1,8,8))

        self.mapping = nn.Linear(config.hidden_size, 64)

        self.mapping2 = nn.Linear(config.hidden_size, 64)

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

        # with torch.no_grad(): 
    
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

        v_in = x.permute(0,2,3,1).contiguous().view(batch, num,-1,feat_dim).mean(2)

        # x_spatial = x.mean(1)
        # x_temporal = x.mean(2)

        loc = torch.zeros(batch,num,5).float()

        loc[:,:,0] = 0.5
        loc[:,:,1] = 0.5
        loc[:,:,2] = 1
        loc[:,:,3] = 1
        loc[:,:,4] = 1

        loc = loc.cuda()

        ret = {}

        if test:
            for head in self.heads:
                ret[head] = []
            ret['relevance'] = []
            for i in range(sents.shape[1]):
                sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v = self.bert(sents[:,i,:], v_in, loc, attention_mask=text_mask[:,i,:])
                # global_v = sequence_output_v.mean(-2)

                # global_v = self.relu(sequence_output_v.mean(-2).permute(0,2,1).contiguous())
                # global_v = self.relu(self.tconv1(global_v))
                # global_v = self.tconv2(global_v)

                # global_t = self.mapping(sequence_output_t[:,:1,:]).permute(0,2,1).contiguous()

                # global_vt = torch.cat([global_v,global_t.expand_as(global_v)], dim=1)

                global_v = sequence_output_v.permute(0,2,1).contiguous()

                global_t_1 = self.relu(self.mapping(sequence_output_t[:,:1,:])).permute(0,2,1).contiguous()

                global_t_2 = self.relu(self.mapping2(sequence_output_t[:,:1,:])).permute(0,2,1).contiguous()


                v_s = self.tv_1(global_v)

                s_map = torch.bmm(v_s.permute(0,2,1), global_t_1)

                v_e = self.tv_2(global_v)

                e_map = torch.bmm(v_e.permute(0,2,1), global_t_2)

                ret['relevance'].append(torch.cat([s_map, e_map], dim=-1).permute(0,2,1).contiguous())

                # ret['relevance'].append(torch.bmm(global_v, global_t))
                # ret['relevance'].append(self.tconv3(global_vt))

                sequence_output_v = sequence_output_v.view(batch*num,h,w,-1).permute(0,3,1,2).contiguous()
         
                sequence_output_v = sequence_output_v.view(batch*num,-1,1,1)
                x = self.relu(self.conv2(x))
                x2 = self.deconv_layers2(x)
                x = self.deconv_layers(x + sequence_output_v.expand_as(x))

                # corr_map = torch.bmm(self.headconv(x).permute(0,2,3,1).contiguous().view(x.size(0),-1,265),self.mapping2(sequence_output_t[:,:1,:]).permute(0,2,1).contiguous())
                
                # corr_map = corr_map.view(batch*num, x.size(-2), x.size(-1),-1).permute(0,3,1,2)
                
                # print('out_V', sequence_output_v.size())

                # x = self.deconv_layers(sequence_output_v)

                for head in self.heads:
                    if head == 'hm':
                        # ret[head]= corr_map
                        ret[head].append(self.__getattr__(head)(x))
                    else:
                        ret[head].append(self.__getattr__(head)(x2))

                # for head in self.heads:
                #     ret[head].append(self.__getattr__(head)(x))

                return [ret]

        global_vt_list = []

        sequence_output_v_list = []

        # sequence_output_t_list = []

        # for c_id in range(int(num/4)):
        #     out_t, out_v, _,_ = self.bert(sents, v_in[:,c_id*4:c_id*4+4,:,:], loc[:,c_id*4:c_id*4+4,:,:], attention_mask=text_mask)

        #     global_c_v = out_v.mean(-2).permute(0,2,1).contiguous()

        #     global_c_t = self.relu(self.mapping(out_t[:,:1,:])).permute(0,2,1).contiguous()

        #     global_c_vt = torch.cat([global_c_v,global_c_t.expand_as(global_c_v)], dim=1)

        #     global_vt_list.append(global_c_vt)

        #     sequence_output_v_list.append(out_v)

        #     # sequence_output_t_list.append(out_t)

        # global_vt = torch.cat(global_vt_list, dim=-1)

        # sequence_output_v = torch.cat(sequence_output_v_list, dim=1)

        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v = self.bert(sents, v_in, loc, attention_mask=text_mask)


        global_v = sequence_output_v.permute(0,2,1).contiguous()
        # global_v = self.relu(self.tconv1(global_v))
        # global_v = self.tconv2(global_v).permute(0,2,1).contiguous()

        # print(sequence_output_t.sum(keepdim=True,dim=1).shape)
        # print(text_mask[:,:].sum(dim=1).shape)
        # print((sequence_output_t.sum(keepdim=True,dim=1)/text_mask[:,:].sum(dim=1)).shape)

        global_t_1 = self.relu(self.mapping(sequence_output_t[:,:1,:])).permute(0,2,1).contiguous()

        global_t_2 = self.relu(self.mapping2(sequence_output_t[:,:1,:])).permute(0,2,1).contiguous()

        # global_t_3 = self.relu(self.mapping3(sequence_output_t[:,:1,:])).permute(0,2,1).contiguous()

        # global_vt = torch.cat([global_v,global_t.expand_as(global_v)], dim=1)

        # global_vt = self.relu(self.tconv1(global_vt))
        # global_vt = self.relu(self.tconv2(global_vt))

        # # ret['relevance'] = torch.bmm(global_v, global_t)
        # ret['relevance'] = self.tconv3(global_vt)

        # global_v = sequence_output_v.permute(0,3,1,2).contiguous().view(batch,-1,num,h, w)


        # global_v = self.relu(self.Conv3d_1(global_v))

        # global_v = self.relu(self.Conv3d_2(global_v))

        # global_v = self.Conv3d_3(global_v)

        # global_v = self.avg(global_v).view(batch,128,num)

        v_s = self.tv_1(global_v)

        # v_s = v_s.mean(-1).mean(-1)

        s_map = torch.bmm(v_s.permute(0,2,1), global_t_1)

        # s_map = torch.bmm(global_v[:,:64,:].permute(0,2,1), global_t_1)

        v_e = self.tv_2(global_v)

        # v_e = v_e.mean(-1).mean(-1)

        e_map = torch.bmm(v_e.permute(0,2,1), global_t_2)

        # e_map = torch.bmm(global_v[:,64:,:].permute(0,2,1), global_t_2)

        ret['relevance'] = torch.cat([s_map, e_map], dim=-1).permute(0,2,1).contiguous()

        sequence_output_v = sequence_output_v.view(batch*num,-1,1,1)
 
        # print('out_V', sequence_output_v.size())

        
        x = self.relu(self.conv2(x))
        x2 = self.deconv_layers2(x)

        x = self.deconv_layers(x + sequence_output_v.expand_as(x))

        # print(self.headconv(x).size())

        # corr_map = torch.bmm(x.permute(0,2,3,1).contiguous().view(batch,-1,64),global_t_3)

        # corr_map = corr_map.view(batch*num, x.size(-2), x.size(-1),-1).permute(0,3,1,2)

        # corr_maps = []

        # # print('x_size:',x.shape)
        # # print(sents_per_image)
        # for i in range(x.size(0)):
        #     m = x[i,:,:,:]
        #     # print(m.shape)
        #     for k in range(sents_per_image[i]):
        #         # print("m:",m.shape,k, i)
        #         # print("idx:", sents_start_idx)
        #         # print('langfeat:', langfeats.shape)
        #         corr_maps.append(torch.mv(m.view(m.size(0),-1).transpose(dim0=1,dim1=0), langfeats[sents_start_idx[i]+k,:]).view(m.size(1),m.size(2)))
        #         if sents_per_image[i] - k==1 and k<2:
        #             corr_maps+= [corr_maps[-1] for _ in range(2-k)]

        # # print(len(corr_maps))
        # corr_maps = torch.stack(corr_maps, dim=0)
        # # print(corr_maps.shape)
        # feat_list = []
        # for i in range(len(sents_per_image)):
        #     feat_list += [x[i,:,:,:] for _ in range(3)]

        # x = torch.stack(feat_list, dim=0)

        for head in self.heads:
            if head == 'hm':
                # ret[head]= corr_map
                ret[head] = self.__getattr__(head)(x)
            else:
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


def get_pose_net(num_layers, heads, head_conv=256):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers)
  return model
