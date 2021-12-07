import re
import json

import paddle
import paddle.nn as nn


class ICConv2d(nn.Layer):
    def __init__(self, pattern_dist, inplanes, planes, kernel_size, stride=1, groups=1, bias_attr=False):
        super(ICConv2d, self).__init__()
        self.conv_list = nn.LayerList()
        self.planes = planes
        for pattern in pattern_dist:
            channel = pattern_dist[pattern]
            pattern_trans = re.findall(r"\d+\.?\d*", pattern)
            pattern_trans[0] = int(pattern_trans[0])+1
            pattern_trans[1] = int(pattern_trans[1])+1
            if channel > 0:
                padding = [0, 0]
                padding[0] = (kernel_size+2*(pattern_trans[0]-1))//2
                padding[1] = (kernel_size+2*(pattern_trans[1]-1))//2
                self.conv_list.append(nn.Conv2D(inplanes, channel, kernel_size=kernel_size, stride=stride,
                                                padding=padding, bias_attr=bias_attr, groups=groups, dilation=pattern_trans))

    def forward(self, x):
        out = []
        for conv in self.conv_list:
            out.append(conv(x))
        out = paddle.concat(out, axis=1)
        assert out.shape[1] == self.planes
        return out
