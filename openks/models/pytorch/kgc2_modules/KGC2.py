# -*- coding: utf-8 -*-
# @Author  : Benjamin
# @FileName: KGC2.py

import numpy as np
import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_
from ...model import KGC2TorchModel


@KGC2TorchModel.register("KGC2", "PyTorch")
class KGC2(KGC2TorchModel):
    def __init__(self, d, d1, d2, **kwargs):
        super(KGC2, self).__init__()

        self.E = nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d2, padding_idx=0)
        k, o = kwargs.get('k', 30), d1
        self.U = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, k * o)),
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.V = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, k * o)),
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.input_dropout = nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = nn.Dropout(kwargs["hidden_dropout2"])
        self.bn0 = nn.BatchNorm1d(d1)
        self.bn1 = nn.BatchNorm1d(d1)
        self.k = k
        self.o = o
        self.loss = nn.BCELoss()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)


    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        r = self.R(r_idx)

        ## MFB
        x = torch.mm(e1, self.U) * torch.mm(r, self.V)
        x = self.hidden_dropout1(x)
        x = x.view(-1, self.o, self.k)
        x = x.sum(-1)
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))

        pred = torch.sigmoid(x)
        return pred

