import re
import json

import paddle
import paddle.nn as nn
from .ic_conv2d import ICConv2d
from paddle.vision.models import resnet


class BottleneckBlock(resnet.BottleneckBlock):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__(inplanes, planes, stride,
                                              downsample, groups, base_width, dilation, norm_layer)
        global pattern, pattern_index
        pattern_index = pattern_index + 1
        width = int(planes * (base_width / 64.)) * groups
        self.conv2 = ICConv2d(
            pattern[pattern_index], width, width, kernel_size=3, stride=stride, bias_attr=False)


class IC_ResNet(resnet.ResNet):
    def __init__(self, block, depth, pattern_path=None, class_dim=1000, with_pool=True):
        super(IC_ResNet, self).__init__(resnet.BottleneckBlock,
                                        depth, num_classes=class_dim, with_pool=with_pool)
        global pattern, pattern_index
        with open(pattern_path, 'r') as f:
            pattern = json.load(f)
        pattern_index = -1

        self.inplanes = 64
        self.dilation = 1

        layer_cfg = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3]
        }
        layers = layer_cfg[depth]

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        assert len(pattern) == pattern_index + 1


def ic_resnet_50(pretrained=False, **kwargs):
    model = IC_ResNet(
        BottleneckBlock,
        depth=50,
        pattern_path='ic_resnet50_k9.json',
        **kwargs
    )
    if pretrained:
        model.set_dict(paddle.load('ic_resnet50_k9.pdparams'))
    return model