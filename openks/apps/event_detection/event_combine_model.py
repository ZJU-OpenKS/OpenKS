# -*- coding: utf-8 -*-
#
# Copyright 2022 HangZhou Hikvision Digital Technology Co., Ltd. All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from openks.models.model import TorchModel

logger = logging.getLogger(__name__)


@TorchModel.register("Pre_CatEmbdConcatModel", "PyTorch")
class Pre_CatEmbdConcatModel(TorchModel):
    """
    pre-mode: embedding on category column
    """

    def __init__(self, config):
        super(Pre_CatEmbdConcatModel, self).__init__()
        self.dic_embed = nn.Embedding(config["dic_len"], config["embd_dim"])
        self.dic_embed = self.dic_embed.apply(self._init_weights)
        self.input_size = config["embd_dim"] * (config["cat_num"] + config["time_num"]) + config["dense_num"]
        self.config = config

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inv_cat_data, inv_dense_data, cat_data, dense_data,
                mask_len):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len
        batch_num, seq_len, in_cat_dim = cat_data.size()

        input_embd = self.dic_embed(cat_data).reshape(batch_num, seq_len, -1)
        input_dense = dense_data
        input2 = torch.cat((input_embd, input_dense), dim=2)

        invinput_embd = self.dic_embed(inv_cat_data).reshape(batch_num, seq_len, -1)
        invinput_dense = inv_dense_data
        inv_input2 = torch.cat((invinput_embd, invinput_dense), dim=2)
        return inv_input2, input2


@TorchModel.register("Pre_AllEmbdSumModel", "PyTorch")
class Pre_AllEmbdSumModel(TorchModel):
    """
    pre-mode: embedding on all column
    """

    def __init__(self, config):
        super(Pre_AllEmbdSumModel, self).__init__()
        self.dic_embed = nn.Embedding(config["dic_len"], config["embd_dim"])
        self.dic_embed = self.dic_embed.apply(self._init_weights)

        self.dense_Wne = nn.Parameter(torch.randn(config["dense_num"], config["embd_dim"]))
        self.dense_Wvn = nn.Parameter(torch.randn(config["dense_num"], config["embd_dim"]))

        self.attn_vec = nn.Parameter(torch.randn(config["embd_dim"], 1))
        self.attn_KMatrix = nn.Parameter(torch.randn(config["embd_dim"], config["embd_dim"]))
        self.attn_VMatrix = nn.Parameter(torch.randn(config["embd_dim"], config["embd_dim"]))

        self.input_size = config["embd_dim"]
        self.config = config
        self.MaskOrNot = config["MaskOrNot"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inv_cat_data, inv_dense_data, cat_data, dense_data, inv_attmask, attmask,
                mask_len):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len

        Ect = self.dic_embed(cat_data)  # batch_size*seq_len*cat_num*embed_dim
        Ent = self.dense_Wne + dense_data.unsqueeze(-1) * self.dense_Wvn  # batch_size*seq_len*dense_num*embed_dim
        Et = torch.cat((Ect, Ent), dim=-2)

        Kt = torch.cat((Ect, self.dense_Wne.unsqueeze(0).unsqueeze(0).expand(Ent.size())),
                       dim=-2)  # batch_size*seq_len*(cat_num+dense_num)*embed_dim
        attmask[torch.where(mask_len == 0)] = 1
        attmask2 = (1.0 - attmask)
        attmask[torch.where(mask_len == 0)] = 0
        attmask2[torch.where(attmask2 == 1)] = -float("inf")  # batch_size*seq_len*(cat_num+dense_num)
        att_score = torch.matmul(torch.matmul(Kt, self.attn_KMatrix),
                                 self.attn_vec).squeeze()  # batch_size*seq_len*(cat_num+dense_num)
        if self.MaskOrNot:
            att_score = torch.softmax(att_score + attmask2, dim=-1)
        else:
            att_score = torch.softmax(att_score, dim=-1)
        # att_score[torch.where(attmask == 0)] = 0.0
        # input2 = torch.matmul(att_score.unsqueeze(-2), Et).squeeze()  # batch_size*seq_len*embed_dim
        input2 = torch.matmul(att_score.unsqueeze(-2), torch.matmul(Et,
                                                                    self.attn_VMatrix)).squeeze()  # batch_size*seq_len*embed_dim  20210527 modify
        input2[torch.where(mask_len == 0)] = 0.0

        inv_Ect = self.dic_embed(inv_cat_data)
        inv_Ent = self.dense_Wne + inv_dense_data.unsqueeze(
            -1) * self.dense_Wvn  # batch_size*seq_len*dense_num*embed_dim
        inv_Et = torch.cat((inv_Ect, inv_Ent), dim=-2)
        inv_Kt = torch.cat((inv_Ect, self.dense_Wne.unsqueeze(0).unsqueeze(0).expand(inv_Ent.size())),
                           dim=-2)  # batch_size*seq_len*(cat_num+dense_num)*embed_dim
        inv_attmask[torch.where(mask_len == 0)] = 1
        inv_attmask2 = (1.0 - inv_attmask)
        inv_attmask[torch.where(mask_len == 0)] = 0
        inv_attmask2[torch.where(inv_attmask2 == 1)] = -float("inf")
        inv_att_score = torch.matmul(torch.matmul(inv_Kt, self.attn_KMatrix),
                                     self.attn_vec).squeeze()  # batch_size*seq_len*(cat_num+dense_num)
        if self.MaskOrNot:
            inv_att_score = torch.softmax(inv_att_score + inv_attmask2, dim=-1)
        else:
            inv_att_score = torch.softmax(inv_att_score, dim=-1)
        # inv_att_score[torch.where(inv_attmask == 0)] = 0.0
        # inv_input2 = torch.matmul(inv_att_score.unsqueeze(-2), inv_Et).squeeze()  # batch_size*seq_len*embed_dim
        inv_input2 = torch.matmul(inv_att_score.unsqueeze(-2), torch.matmul(inv_Et,
                                                                            self.attn_VMatrix)).squeeze()  # batch_size*seq_len*embed_dim 20210527 modify
        inv_input2[torch.where(mask_len == 0)] = 0.0
        return inv_input2, input2


@TorchModel.register("SingleHeadSumModel", "PyTorch")
class SingleHeadSumModel(TorchModel):
    """
    single head
    """

    def __init__(self, config):
        super(SingleHeadSumModel, self).__init__()
        self.attn_vec = nn.Parameter(torch.randn(config["embd_dim"], 1))
        self.attn_KMatrix = nn.Parameter(torch.randn(config["embd_dim"], config["embd_dim"]))
        self.attn_VMatrix = nn.Parameter(torch.randn(config["embd_dim"], config["embd_dim"]))

        self.input_size = config["embd_dim"]
        self.config = config
        self.MaskOrNot = config["MaskOrNot"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inv_Et, inv_Kt, Et, Kt, inv_attmask, attmask,
                mask_len):  # Kt：batch_size*seq_len*(cat_num+dense_num)*embed_dim  attmask：batch_size*seq_len*(cat_num+dense_num)
        attmask[torch.where(mask_len == 0)] = 1
        attmask2 = (1.0 - attmask)
        attmask[torch.where(mask_len == 0)] = 0
        attmask2[torch.where(attmask2 == 1)] = -float("inf")  # batch_size*seq_len*(cat_num+dense_num)
        att_score = torch.matmul(torch.matmul(Kt, self.attn_KMatrix),
                                 self.attn_vec).squeeze()  # batch_size*seq_len*(cat_num+dense_num)
        if self.MaskOrNot:
            att_score = torch.softmax(att_score + attmask2, dim=-1)
        else:
            att_score = torch.softmax(att_score, dim=-1)
        # att_score[torch.where(attmask == 0)] = 0.0
        # input2 = torch.matmul(att_score.unsqueeze(-2), Et).squeeze()  # batch_size*seq_len*embed_dim
        input2 = torch.matmul(att_score.unsqueeze(-2), torch.matmul(Et,
                                                                    self.attn_VMatrix)).squeeze()  # batch_size*seq_len*embed_dim  20210527 modify
        input2[torch.where(mask_len == 0)] = 0.0

        inv_attmask[torch.where(mask_len == 0)] = 1
        inv_attmask2 = (1.0 - inv_attmask)
        inv_attmask[torch.where(mask_len == 0)] = 0
        inv_attmask2[torch.where(inv_attmask2 == 1)] = -float("inf")
        inv_att_score = torch.matmul(torch.matmul(inv_Kt, self.attn_KMatrix),
                                     self.attn_vec).squeeze()  # batch_size*seq_len*(cat_num+dense_num)
        if self.MaskOrNot:
            inv_att_score = torch.softmax(inv_att_score + inv_attmask2, dim=-1)
        else:
            inv_att_score = torch.softmax(inv_att_score, dim=-1)
        # inv_att_score[torch.where(inv_attmask == 0)] = 0.0
        # inv_input2 = torch.matmul(inv_att_score.unsqueeze(-2), inv_Et).squeeze()  # batch_size*seq_len*embed_dim
        inv_input2 = torch.matmul(inv_att_score.unsqueeze(-2), torch.matmul(inv_Et,
                                                                            self.attn_VMatrix)).squeeze()  # batch_size*seq_len*embed_dim 20210527 modify
        inv_input2[torch.where(mask_len == 0)] = 0.0
        return inv_input2, input2


@TorchModel.register("Pre_AllEmbdMultiHeadSumModel", "PyTorch")
class Pre_AllEmbdMultiHeadSumModel(TorchModel):
    """
    pre-mode: embedding on all column
    """

    def __init__(self, config):
        super(Pre_AllEmbdMultiHeadSumModel, self).__init__()
        self.num_heads = config["num_heads"]
        self.dic_embed = nn.Embedding(config["dic_len"], config["embd_dim"])
        self.dic_embed = self.dic_embed.apply(self._init_weights)

        self.dense_Wne = nn.Parameter(torch.randn(config["dense_num"], config["embd_dim"]))
        self.dense_Wvn = nn.Parameter(torch.randn(config["dense_num"], config["embd_dim"]))

        self.multiHeadSum = [SingleHeadSumModel(config) for i in range(self.num_heads)]
        self.multiHeadSumModule = nn.ModuleList(self.multiHeadSum)
        self.input_size = config["embd_dim"]
        self.config = config
        self.featureTrans = TransformerEncoder(config["embd_dim"], num_heads=8, dropout=config["dropout_prob"],
                                               feedforward_dim=2 * config["embd_dim"])

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inv_cat_data, inv_dense_data, cat_data, dense_data, inv_attmask, attmask,
                mask_len):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len
        Ect = self.dic_embed(cat_data)  # batch_size*seq_len*cat_num*embed_dim
        Ent = self.dense_Wne + dense_data.unsqueeze(-1) * self.dense_Wvn  # batch_size*seq_len*dense_num*embed_dim
        Et = torch.cat((Ect, Ent), dim=-2)

        Kt = torch.cat((Ect, self.dense_Wne.unsqueeze(0).unsqueeze(0).expand(Ent.size())),
                       dim=-2)  # batch_size*seq_len*(cat_num+dense_num)*embed_dim
        flat_Et = Et.reshape(-1, Et.size()[2], Et.size()[3])  ## (batch_size*seq_len)*(cat_num+dense_num)*embed_dim
        ori_bn, ori_seq, cat_dense_size = attmask.size()
        attmask[torch.where(mask_len == 0)] = 1
        flat_mask = (1 - attmask).bool().reshape(-1, cat_dense_size)
        attmask[torch.where(mask_len == 0)] = 0
        flat_attmask = torch.zeros(cat_dense_size, cat_dense_size).to(flat_mask.device)
        Et = self.featureTrans(flat_Et.permute(1, 0, 2), flat_mask, flat_attmask).permute(1, 0, 2).reshape(ori_bn,
                                                                                                           ori_seq,
                                                                                                           cat_dense_size,
                                                                                                           self.input_size)
        Et[torch.where(mask_len == 0)] = 0
        Et[torch.where(attmask == 0)] = 0
        inv_Ect = self.dic_embed(inv_cat_data)
        inv_Ent = self.dense_Wne + inv_dense_data.unsqueeze(
            -1) * self.dense_Wvn  # batch_size*seq_len*dense_num*embed_dim
        inv_Et = torch.cat((inv_Ect, inv_Ent), dim=-2)
        inv_Kt = torch.cat((inv_Ect, self.dense_Wne.unsqueeze(0).unsqueeze(0).expand(inv_Ent.size())),
                           dim=-2)  # batch_size*seq_len*(cat_num+dense_num)*embed_dim

        flat_inv_Et = inv_Et.reshape(-1, inv_Et.size()[2],
                                     inv_Et.size()[3])  ## (batch_size*seq_len)*(cat_num+dense_num)*embed_dim
        ori_bn, ori_seq, cat_dense_size = inv_attmask.size()
        inv_attmask[torch.where(mask_len == 0)] = 1
        flat_inv_mask = (1 - inv_attmask).bool().reshape(-1, cat_dense_size)
        inv_attmask[torch.where(mask_len == 0)] = 0
        flat_inv_attmask = torch.zeros(cat_dense_size, cat_dense_size).to(flat_inv_mask.device)
        inv_Et = self.featureTrans(flat_inv_Et.permute(1, 0, 2), flat_inv_mask, flat_inv_attmask).permute(1, 0,
                                                                                                          2).reshape(
            ori_bn, ori_seq, cat_dense_size, self.input_size)
        inv_Et[torch.where(mask_len == 0)] = 0
        inv_Et[torch.where(attmask == 0)] = 0

        inv_input2, input2 = [], []
        for layer in self.multiHeadSumModule:
            inv_it2, it2 = layer(inv_Et, inv_Kt, Et, Kt, inv_attmask, attmask, mask_len)
            inv_input2.append(inv_it2)
            input2.append(it2)
        inv_input2 = torch.cat(inv_input2, dim=-1)
        input2 = torch.cat(input2, dim=-1)
        return inv_input2, input2


@TorchModel.register("Mid_Pre2Beh", "PyTorch")
class Mid_Pre2Beh(TorchModel):
    def __init__(self, config, input_size):
        super(Mid_Pre2Beh, self).__init__()
        self.fc1 = nn.Linear(input_size, config["hidden_size"])
        self.dropout1 = nn.Dropout(config["dropout_prob"])
        self.fc1 = self.fc1.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input2):  # input:batch_size*seq_len*input_dim
        out2 = self.dropout1(torch.relu(self.fc1(input2)))
        return out2


@TorchModel.register("Beh_BiLSTMMeanMaxPool", "PyTorch")
class Beh_BiLSTMMeanMaxPool(TorchModel):
    def __init__(self, config):
        super(Beh_BiLSTMMeanMaxPool, self).__init__()
        self.lstm1 = nn.LSTM(input_size=config["hidden_size"],
                             hidden_size=config["hidden_size"],
                             num_layers=config["num_layers"],
                             bias=True,
                             batch_first=True,
                             dropout=config["dropout_prob"],
                             bidirectional=False
                             )
        self.inv_lstm1 = nn.LSTM(input_size=config["hidden_size"],
                                 hidden_size=config["hidden_size"],
                                 num_layers=config["num_layers"],
                                 bias=True,
                                 batch_first=True,
                                 dropout=config["dropout_prob"],
                                 bidirectional=False
                                 )
        self.fc2 = nn.Linear(2 * 2 * config["hidden_size"], config["hidden_size"])
        self.fc2 = self.fc2.apply(self._init_weights)
        self.dropout2 = nn.Dropout(config["dropout_prob"])
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inv_out2, out2, mask_len):  # input:batch_size*seq_len*hidden_size  mask_len:batch_size*seq_len

        # self.lstm1.flatten_parameters()
        out3, _ = self.lstm1(out2)  # batch_size*seq_len*hidden_size
        out4_mean = (out3 * mask_len.unsqueeze(-1)).sum(dim=1) / mask_len.unsqueeze(-1).sum(dim=1)
        out4_max = out3 + (1 - mask_len.unsqueeze(-1)) * (-1e10)
        out4_max = out4_max.max(dim=1)[0].float()
        input5 = torch.cat((out4_max, out4_mean), dim=1)

        # self.inv_lstm1.flatten_parameters()
        inv_out3, _ = self.inv_lstm1(inv_out2)  # batch_size*seq_len*hidden_size
        invout4_mean = (inv_out3 * mask_len.unsqueeze(-1)).sum(dim=1) / mask_len.unsqueeze(-1).sum(dim=1)
        invout4_max = inv_out3 + (1 - mask_len.unsqueeze(-1)) * (-1e10)
        invout4_max = invout4_max.max(dim=1)[0].float()
        inv_input5 = torch.cat((invout4_max, invout4_mean), dim=1)

        input5 = torch.cat((input5, inv_input5), dim=-1)
        out5 = self.dropout2(torch.relu(self.fc2(input5)))
        return out5


@TorchModel.register("TransformerEncoder", "PyTorch")
class TransformerEncoder(TorchModel):
    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear_1 = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = torch.nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = torch.nn.LayerNorm(
            embed_dim)
        self.layernorm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x_in, padding_mask, attMask):
        attn_out, _ = self.attn(x_in, x_in, x_in, key_padding_mask=padding_mask, attn_mask=attMask)
        x = self.layernorm_1(x_in + attn_out)
        # x = x_in + attn_out
        ff_out = self.linear_2(torch.nn.functional.relu(self.linear_1(x)))
        x = self.layernorm_2(x + ff_out)
        # x = x + ff_out
        return x


@TorchModel.register("Beh_SelfAttnAttnPool", "PyTorch")
class Beh_SelfAttnAttnPool(TorchModel):
    def __init__(self, config):
        super(Beh_SelfAttnAttnPool, self).__init__()
        self.encoder_1 = TransformerEncoder(config["hidden_size"], num_heads=8, dropout=config["dropout_prob"],
                                            feedforward_dim=2 * config["hidden_size"])
        self.queryMatrix1 = nn.Parameter(torch.randn(config["hidden_size"], config["hidden_size"]))
        self.queryMatrix2 = nn.Parameter(torch.randn(config["hidden_size"], config["hidden_size"]))
        self.attvector = nn.Parameter(torch.zeros(config["hidden_size"], 1))
        self.fc2 = nn.Linear(2 * config["hidden_size"], config["hidden_size"])
        self.fc2 = self.fc2.apply(self._init_weights)
        self.dropout2 = nn.Dropout(config["dropout_prob"])
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inv_out2, out2, mask_len):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len
        bs, seq_len, _ = out2.size()
        attMASK = torch.from_numpy(np.triu(np.ones([seq_len, seq_len]), k=1)).to(out2.device)
        attMASK[torch.where(attMASK == 1)] = float("-inf")
        out2 = out2.permute(1, 0, 2)
        mask_len2 = (1 - mask_len).bool()
        x1 = self.encoder_1(out2, mask_len2, attMASK)
        # x2=self.encoder_2(x1,mask_len2)
        # out3=self.encoder_3(x2,mask_len2).permute(1,0,2)
        out3 = x1.permute(1, 0, 2)

        # mean max last Pooling
        out4_mean = (out3 * mask_len.unsqueeze(-1)).sum(dim=1) / mask_len.unsqueeze(-1).sum(dim=1)
        # out4_max = out3 + (1 - mask_len.unsqueeze(-1)) * (-1e10)
        # out4_max = out4_max.max(dim=1)[0].float()
        lastIDX = mask_len.sum(dim=-1) - 1
        out4_last = out3[(torch.arange(bs), lastIDX)]  # batch_size*hidden
        out4_max = out4_last - out4_mean
        attvectorMean = torch.matmul(out4_mean, self.queryMatrix1)
        attvectorMax = torch.matmul(out4_max, self.queryMatrix2)  # batch_size*hidden_size

        # atten-Pooling
        att_score = torch.matmul(out3, attvectorMean.unsqueeze(-1)).squeeze()  # batch_size * seq_len
        att_score = torch.tanh(att_score) + (1 - mask_len) * (-1e10)
        att_score = torch.exp(att_score) / torch.sum(torch.exp(att_score), dim=-1).unsqueeze(-1)
        input5 = torch.bmm(att_score.unsqueeze(1), out3).squeeze()  # batch_size*hidden

        att_score2 = torch.matmul(out3, attvectorMax.unsqueeze(-1)).squeeze()  # batch_size * seq_len
        att_score2 = torch.tanh(att_score2) + (1 - mask_len) * (-1e10)
        att_score2 = torch.exp(att_score2) / torch.sum(torch.exp(att_score2), dim=-1).unsqueeze(-1)
        input6 = torch.bmm(att_score2.unsqueeze(1), out3).squeeze()  # batch_size*hidden
        out5 = self.dropout2(torch.relu(self.fc2(torch.cat((input5, input6), dim=-1))))

        return out5


@TorchModel.register("CatEmbdConcatModel", "PyTorch")
class CatEmbdConcatModel(TorchModel):
    def __init__(self, config):
        super(CatEmbdConcatModel, self).__init__()
        self.Pre = Pre_CatEmbdConcatModel(config)
        self.input_size = config["embd_dim"] * (config["cat_num"] + config["time_num"]) + config["dense_num"]
        self.latentSpaces = config["latentSpaces"]
        layers = [Mid_Pre2Beh(config, self.input_size) for i in range(self.latentSpaces)]
        self.multiSpacesList = nn.ModuleList(layers)
        self.Beh = Beh_BiLSTMMeanMaxPool(config)
        self.fc3 = nn.Linear(config["hidden_size"] * config["latentSpaces"], config["label_dim"])
        self.fc3 = self.fc3.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_x):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len

        inv_cat_data, inv_dense_data, cat_data, dense_data, mask_len = input_x

        inv_input2, input2 = self.Pre(inv_cat_data, inv_dense_data, cat_data, dense_data, mask_len)
        # out2 = self.dropout1(torch.relu(self.fc1(input2)))
        # inv_out2 = self.dropout1(torch.relu(self.fc1(inv_input2)))
        out_list = []
        for layer in self.multiSpacesList:
            out2 = layer(input2)
            inv_out2 = layer(inv_input2)
            tmp = self.Beh(inv_out2, out2, mask_len)
            out_list.append(tmp)
        out5 = torch.cat(out_list, dim=-1)
        out6 = F.log_softmax(self.fc3(out5))
        return out6, out5


@TorchModel.register("CatEmbdConcatAttModel", "PyTorch")
class CatEmbdConcatAttModel(TorchModel):
    """
    self attention+meanMax Pooling
    """

    def __init__(self, config):
        super(CatEmbdConcatAttModel, self).__init__()
        self.Pre = Pre_CatEmbdConcatModel(config)
        self.input_size = config["embd_dim"] * (config["cat_num"] + config["time_num"]) + config["dense_num"]
        self.latentSpaces = config["latentSpaces"]
        layers = [Mid_Pre2Beh(config, self.input_size) for i in range(self.latentSpaces)]
        self.multiSpacesList = nn.ModuleList(layers)
        self.Beh = Beh_SelfAttnAttnPool(config)
        self.fc3 = nn.Linear(config["hidden_size"] * config["latentSpaces"], config["label_dim"])
        self.fc3 = self.fc3.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_x):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len
        inv_cat_data, inv_dense_data, cat_data, dense_data, mask_len = input_x
        inv_input2, input2 = self.Pre(inv_cat_data, inv_dense_data, cat_data, dense_data, mask_len)
        # out2 = self.dropout1(torch.relu(self.fc1(input2)))
        # inv_out2 = self.dropout1(torch.relu(self.fc1(inv_input2)))
        out_list = []
        for layer in self.multiSpacesList:
            out2 = layer(input2)
            inv_out2 = layer(inv_input2)
            tmp = self.Beh(inv_out2, out2, mask_len)
            out_list.append(tmp)
        out5 = torch.cat(out_list, dim=-1)
        out6 = F.log_softmax(self.fc3(out5))
        return out6, out5


@TorchModel.register("CatOnehotConcatModel", "PyTorch")
class CatOnehotConcatModel(TorchModel):
    def __init__(self, config):
        super(CatOnehotConcatModel, self).__init__()
        self.input_size = config["oneHotConcat_dim"]
        self.latentSpaces = config["latentSpaces"]
        layers = [Mid_Pre2Beh(config, self.input_size) for i in range(self.latentSpaces)]
        self.multiSpacesList = nn.ModuleList(layers)
        self.Beh = Beh_BiLSTMMeanMaxPool(config)
        self.fc3 = nn.Linear(config["hidden_size"] * config["latentSpaces"], config["label_dim"])
        self.fc3 = self.fc3.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_x):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len
        inv_data, data, mask_len = input_x
        out_list = []
        for layer in self.multiSpacesList:
            out2 = layer(data)
            inv_out2 = layer(inv_data)
            tmp = self.Beh(inv_out2, out2, mask_len)
            out_list.append(tmp)
        out5 = torch.cat(out_list, dim=-1)
        out6 = F.log_softmax(self.fc3(out5))
        return out6, out5


@TorchModel.register("CatOnehotConcatAttModel", "PyTorch")
class CatOnehotConcatAttModel(TorchModel):
    """
    self attention+ Attention Pooling
    """

    def __init__(self, config):
        super(CatOnehotConcatAttModel, self).__init__()
        self.input_size = config["oneHotConcat_dim"]
        self.latentSpaces = config["latentSpaces"]
        layers = [Mid_Pre2Beh(config, self.input_size) for i in range(self.latentSpaces)]
        self.multiSpacesList = nn.ModuleList(layers)
        self.Beh = Beh_SelfAttnAttnPool(config)
        self.fc3 = nn.Linear(config["hidden_size"] * config["latentSpaces"], config["label_dim"])
        self.fc3 = self.fc3.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_x):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len
        inv_data, data, mask_len = input_x
        out_list = []
        for layer in self.multiSpacesList:
            out2 = layer(data)
            inv_out2 = layer(inv_data)
            tmp = self.Beh(inv_out2, out2, mask_len)
            out_list.append(tmp)
        out5 = torch.cat(out_list, dim=-1)
        out6 = F.log_softmax(self.fc3(out5))
        return out6, out5


@TorchModel.register("AllEmbdSumModel", "PyTorch")
class AllEmbdSumModel(TorchModel):
    def __init__(self, config):
        super(AllEmbdSumModel, self).__init__()
        self.Pre = Pre_AllEmbdMultiHeadSumModel(config)
        self.input_size = config["embd_dim"] * config["num_heads"]
        self.latentSpaces = config["latentSpaces"]
        layers = [Mid_Pre2Beh(config, self.input_size) for i in range(self.latentSpaces)]
        self.multiSpacesList = nn.ModuleList(layers)
        self.Beh = Beh_BiLSTMMeanMaxPool(config)
        self.fc3 = nn.Linear(config["hidden_size"] * config["latentSpaces"], config["label_dim"])
        self.fc3 = self.fc3.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_x):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len
        inv_cat_data, inv_dense_data, cat_data, dense_data, inv_attmask, attmask, mask_len = input_x
        inv_input2, input2 = self.Pre(inv_cat_data, inv_dense_data, cat_data, dense_data, inv_attmask, attmask,
                                      mask_len)
        # out2 = self.dropout1(torch.relu(self.fc1(input2)))
        # inv_out2 = self.dropout1(torch.relu(self.fc1(inv_input2)))
        out_list = []
        for layer in self.multiSpacesList:
            out2 = layer(input2)
            inv_out2 = layer(inv_input2)
            tmp = self.Beh(inv_out2, out2, mask_len)
            out_list.append(tmp)
        out5 = torch.cat(out_list, dim=-1)
        out6 = F.log_softmax(self.fc3(out5))
        return out6, out5


@TorchModel.register("AllEmbdSumAttModel", "PyTorch")
class AllEmbdSumAttModel(TorchModel):
    """
    self attention+meanMax Pooling
    """

    def __init__(self, config):
        super(AllEmbdSumAttModel, self).__init__()
        self.Pre = Pre_AllEmbdMultiHeadSumModel(config)
        self.input_size = config["embd_dim"] * config["num_heads"]
        self.latentSpaces = config["latentSpaces"]
        layers = [Mid_Pre2Beh(config, self.input_size) for i in range(self.latentSpaces)]
        self.multiSpacesList = nn.ModuleList(layers)
        self.Beh = Beh_SelfAttnAttnPool(config)
        self.fc3 = nn.Linear(config["hidden_size"] * config["latentSpaces"], config["label_dim"])
        self.fc3 = self.fc3.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_x):  # input:batch_size*seq_len*input_dim  mask_len:batch_size*seq_len
        inv_cat_data, inv_dense_data, cat_data, dense_data, inv_attmask, attmask, mask_len = input_x
        inv_input2, input2 = self.Pre(inv_cat_data, inv_dense_data, cat_data, dense_data, inv_attmask, attmask,
                                      mask_len)
        # out2 = self.dropout1(torch.relu(self.fc1(input2)))
        # inv_out2 = self.dropout1(torch.relu(self.fc1(inv_input2)))
        out_list = []
        for layer in self.multiSpacesList:
            out2 = layer(input2)
            inv_out2 = layer(inv_input2)
            tmp = self.Beh(inv_out2, out2, mask_len)
            out_list.append(tmp)
        out5 = torch.cat(out_list, dim=-1)
        out6 = F.log_softmax(self.fc3(out5))
        return out6, out5


@TorchModel.register("FocalLossModel", "PyTorch")
class FocalLossModel(TorchModel):
    """
    logsoftmax
    """

    def __init__(self, class_num, alpha=torch.tensor([0.1, 0.9]), gamma=2, size_average=True):
        super(FocalLossModel, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = torch.exp(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        min_probs = probs.new(probs.size()).fill_(1e-16)
        max_probs = probs.new(probs.size()).fill_(1 - 1e-16)
        probs = torch.max(probs, min_probs)
        probs = torch.min(probs, max_probs)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
