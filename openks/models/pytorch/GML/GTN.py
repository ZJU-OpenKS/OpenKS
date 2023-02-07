import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class GTNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_neigh, device):
        super(GTNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neigh = n_neigh
        self.device = device

        # self.W1 = torch.nn.Linear(2*self.input_dim, self.output_dim).to(device)
        # self.W2 = torch.nn.Linear(self.input_dim, self.output_dim).to(device)
        self.k_linear = nn.Linear(self.input_dim, self.output_dim).to(device)
        self.q_linear = nn.Linear(self.input_dim, self.output_dim).to(device)
        self.v_linear = nn.Linear(self.input_dim, self.output_dim).to(device)
        self.norm = nn.LayerNorm(self.output_dim).to(device)

    # def forward(self, tgt_feature, neigh_features):
    #     # batch_size = tgt_feature.shape[0]
    #     h_mean = neigh_features.mean(dim=-2)
    #     h_pna = torch.square(neigh_features.sum(dim=-2)) - torch.square(neigh_features).sum(dim=-2)
    #     # h_pna = torch.zeros(batch_size, self.input_dim).to(self.device)
    #     # h_mean = torch.zeros(batch_size, self.input_dim).to(self.device)
    #     # for neigh_feature in neigh_features:
    #     #     h_pna -= neigh_feature * neigh_feature
    #     #     h_mean += neigh_feature # h_mean= n_1+n_2+...+n_m
    #
    #     # h_pna += h_mean * h_mean # h_pna= (n_1+n_2+...)* (n_1+n_2+...)- n_1^2-n_2^2-...
    #
    #     h_pna = h_pna / (2 * self.n_neigh)
    #     # h_mean = h_mean / self.n_neigh # batchsize * input_dim
    #
    #     h_neigh = torch.cat([h_pna, h_mean], dim=-1) # batchsize * 2input_dim
    #     h_neigh = self.W1(h_neigh)
    #     h_tgt = self.W2(tgt_feature)
    #
    #     if self.mode == 'sum':
    #         e = h_neigh + h_tgt
    #     elif self.mode == 'concat':
    #         e = torch.cat([h_neigh, h_tgt], dim=-1)
    #
    #     # e = F.normalize(e, p=2, dim=-1)
    #     e = torch.tanh(e)
    #     # return e
    #     return h_mean

    def forward(self, tgt_feature, neigh_feature):
        key = self.k_linear(tgt_feature.unsqueeze(dim=-2))
        qry = self.q_linear(neigh_feature)
        val = self.v_linear(neigh_feature)
        att = (qry * key).sum(dim=-1) / math.sqrt(self.output_dim)
        att = F.softmax(att, dim=-2)
        h = torch.sum(att.unsqueeze(dim=-1) * val, dim=-2)
        return h

# user and item PNA
class GTN(nn.Module):
    def __init__(self, input_dim, output_dim, n_neighs, device):
        super(GTN, self).__init__()
        self.output_dim = output_dim
        self.n_layer = len(n_neighs)
        self.GTNLayers = nn.ModuleList(
            [GTNLayer(input_dim, output_dim, n_neighs[l], device).to(device) for l in range(self.n_layer)])

        self.k_linear = nn.Linear(input_dim, output_dim).to(device)
        self.q_linear = nn.Linear(input_dim, output_dim).to(device)
        self.v_linear = nn.Linear(input_dim, output_dim).to(device)
        self.res = nn.Linear(input_dim, output_dim).to(device)
        self.norm = nn.LayerNorm(output_dim).to(device)

    def forward(self, tgt_feature, neigh_features):
        h_list = [tgt_feature]
        for l in range(self.n_layer):
            h_list.append(self.GTNLayers[l](tgt_feature, neigh_features[l]))
        h_list = torch.stack(h_list, dim=-2)
        qry = self.q_linear(h_list)
        key = self.k_linear(h_list)
        val = self.v_linear(h_list)
        att = torch.matmul(qry, key.transpose(-2, -1)) / math.sqrt(self.output_dim)
        att = F.softmax(att, dim=-1)
        output = torch.matmul(att, val)
        val_res = self.res(h_list)
        output += val_res
        output = self.norm(output)
        return output

        # h_list = []
        # for l in range(self.n_layer):
        #     h_list.append(self.PNALayers[l](tgt_feature, neigh_features[l]))
        # h_cross = []
        # for l1 in range(self.n_layer):
        #     for l2 in range(l1+1, self.n_layer):
        #         h_cross.append(h_list[l1] * h_list[l2])
        #
        # if self.cross_mode == 'sum':
        #     h = torch.sum(torch.stack(h_list, dim=1), dim=1) #+ torch.sum(torch.stack(h_cross, dim=1), dim=1)
        # elif self.cross_mode == 'concat':
        #     h = torch.cat(h_list + h_cross, dim=-1)
        #
        # h = F.normalize(h, p=2, dim=-1)
        # # h = torch.tanh(h)
        # return h