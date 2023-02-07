import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class GTNLayer(nn.Module):
    def __init__(self, cur_dim, n_head_layer, dropout, device):
        super(GTNLayer, self).__init__()
        self.cur_dim = cur_dim
        self.n_head_layer = n_head_layer
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.W = nn.ModuleList()
        for _ in range(self.n_head_layer):
            self.W.append(nn.Linear(self.cur_dim*2, 1).to(device))
        self.relu = nn.ReLU()
        self.proj = nn.Linear(cur_dim*n_head_layer, cur_dim)

    def forward(self, tgt_feature, neigh_feature):
        tgt_feature = torch.stack([tgt_feature]*neigh_feature.shape[-2], dim=-2)
        hs = []
        for l in range(self.n_head_layer):
            att = self.relu(self.W[l](torch.cat([tgt_feature, neigh_feature], dim=-1)))
            att = F.softmax(att, dim=-2)
            att = self.dropout(att)
            h = torch.sum(att * neigh_feature, dim=-2)
            hs.append(h)
        hs = self.proj(torch.cat(hs, dim=-1))
        # hs = torch.cat(hs, dim=-1)
        return hs
#

class CrossLayer(nn.Module):
    def __init__(self, cur_dim, n_head_cross, dropout, device):
        super(CrossLayer, self).__init__()
        self.cur_dim = cur_dim
        self.n_head_cross = n_head_cross
        self.dropout = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.k_linear = nn.Linear(cur_dim, cur_dim).to(device)
        self.q_linear = nn.Linear(cur_dim, cur_dim).to(device)
        self.v_linear = nn.Linear(cur_dim, cur_dim).to(device)
        self.o_linear = nn.Linear(cur_dim, cur_dim).to(device)
        # self.proj = nn.Linear(cur_dim, cur_dim).to(device)
        self.res = nn.Linear(cur_dim, cur_dim).to(device)

        # self.position_wise_feed_forward = nn.Sequential(
        #     nn.Linear(cur_dim, cur_dim),
        #     nn.ReLU(),
        #     nn.Linear(cur_dim, cur_dim),
        # )
        self.norm = nn.LayerNorm(cur_dim).to(device)
        # self.norm2 = nn.LayerNorm(cur_dim).to(device)

    def forward(self, h_feature):
        qry = self.q_linear(h_feature)
        key = self.k_linear(h_feature)
        val = self.v_linear(h_feature)
        qry = torch.cat(qry.chunk(self.n_head_cross, dim=-1), dim=0)
        key = torch.cat(key.chunk(self.n_head_cross, dim=-1), dim=0)
        val = torch.cat(val.chunk(self.n_head_cross, dim=-1), dim=0)
        att = torch.matmul(qry, key.transpose(-2, -1)) / math.sqrt(self.cur_dim//self.n_head_cross)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        output_att = torch.matmul(att, val)
        output_att = torch.cat(output_att.chunk(self.n_head_cross, dim=0), dim=-1)
        output_att = self.o_linear(output_att)
        # val_res = self.res(h_feature)
        output_att = self.norm(self.res(h_feature) + self.dropout(output_att))
        # # output_att = self.norm1(val_res + output_att)
        # output = self.position_wise_feed_forward(output_att)
        # output = output_att + self.dropout(self.norm2(output))
        # # # output = self.relu(output)
        # # # output = self.norm(output)
        # # # output = F.normalize(output, p=2, dim=-1)
        return output_att

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout, max_len=6):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + torch.autograd.Variable(self.pe[:x.size(0), :, :], requires_grad=False)
#         return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GTN(nn.Module):
    def __init__(self, cur_dim, n_neighs, n_head_layer, n_head_cross, n_cross, dropout, device):
        super(GTN, self).__init__()
        # self.dropout = nn.Dropout(dropout)
        # self.output_dim = output_dim
        self.cur_dim = cur_dim
        self.n_layer = len(n_neighs)
        self.n_cross = n_cross
        self.GTNLayers = nn.ModuleList(
            [GTNLayer(cur_dim, n_head_layer, dropout, device).to(device) for _ in range(self.n_layer)]).to(device)

        self.CrossLayers = nn.ModuleList(
            [CrossLayer(cur_dim, n_head_cross, dropout, device) for _ in range(self.n_cross)]).to(device)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=cur_dim, nhead=n_head_cross,
        #                                            dim_feedforward=cur_dim, dropout=dropout)
        # self.CrossLayers = nn.TransformerEncoder(encoder_layer, n_cross)

        self.pos_encoder = PositionalEncoding(cur_dim, dropout, self.n_layer+1).to(device)


    # def forward(self, tgt_feature, neigh_features):
    #     h_list = [tgt_feature]
    #     for l in range(self.n_layer):
    #         h_list.append(self.GTNLayers[l](tgt_feature, neigh_features[l]))
    #     h_list = torch.stack(h_list, dim=-2)
    #     h_feature = self.position(h_list)
    #     for l in range(self.n_cross):
    #         h_feature = self.CrossLayers[l](h_feature)
    #     return h_feature, h_list
    def forward(self, tgt_feature, neigh_features):
        h_list = [tgt_feature]
        for l in range(self.n_layer):
            h_list.append(self.GTNLayers[l](tgt_feature, neigh_features[l]))
        h_list = torch.stack(h_list, dim=-2)
        h_list = h_list.view(-1, self.n_layer+1, self.cur_dim) * math.sqrt(self.cur_dim)
        h_feature = self.pos_encoder(h_list)
        # h_feature = h_list
        # h_feature = torch.cat(self.CrossLayers(h_feature).chunk(self.n_layer+1, dim=0), dim=-1).squeeze(0)
        for l in range(self.n_cross):
            h_feature = self.CrossLayers[l](h_feature)
        return h_feature



# class GTNLayer(nn.Module):
#     def __init__(self, cur_dim, n_head_layer, dropout, device):
#         super(GTNLayer, self).__init__()
#         self.input_dim = cur_dim
#         self.output_dim = cur_dim
#         self.dropout = nn.Dropout(dropout)
#         self.device = device
#
#         # self.W1 = torch.nn.Linear(2*self.input_dim, self.output_dim).to(device)
#         # self.W2 = torch.nn.Linear(self.input_dim, self.output_dim).to(device)
#         self.k_linear = nn.Linear(self.input_dim, self.output_dim).to(device)
#         self.q_linear = nn.Linear(self.input_dim, self.output_dim).to(device)
#         self.v_linear = nn.Linear(self.input_dim, self.output_dim).to(device)
#         # self.norm = nn.LayerNorm(self.output_dim).to(device)
#
#     def forward(self, tgt_feature, neigh_feature):
#         key = self.k_linear(tgt_feature.unsqueeze(dim=-2))
#         qry = self.q_linear(neigh_feature)
#         val = self.v_linear(neigh_feature)
#         att = (qry * key).sum(dim=-1) / math.sqrt(self.output_dim)
#         att = F.softmax(att, dim=-2)
#         att = self.dropout(att)
#         h = torch.sum(att.unsqueeze(dim=-1) * val, dim=-2)
#         # h = self.norm(h)
#         return h

# class GTN(nn.Module):
#     def __init__(self, cur_dim, n_neighs, n_head_layer, n_head_cross, n_cross, dropout, device):
#         super(GTN, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.output_dim = cur_dim
#         self.n_layer = len(n_neighs)
#         self.GTNLayers = nn.ModuleList(
#             [GTNLayer(cur_dim, n_neighs[l], dropout, device).to(device) for l in range(self.n_layer)])
#
#         self.k_linear = nn.Linear(cur_dim, cur_dim).to(device)
#         self.q_linear = nn.Linear(cur_dim, cur_dim).to(device)
#         self.v_linear = nn.Linear(cur_dim, cur_dim).to(device)
#         self.res = nn.Linear(cur_dim, cur_dim).to(device)
#         self.relu = nn.ReLU(inplace=True)
#         self.norm = nn.LayerNorm(cur_dim).to(device)
#
#     def forward(self, tgt_feature, neigh_features):
#         h_list = [tgt_feature]
#         for l in range(self.n_layer):
#             h_list.append(self.GTNLayers[l](tgt_feature, neigh_features[l]))
#         h_list = torch.stack(h_list, dim=-2)
#         qry = self.q_linear(h_list)
#         key = self.k_linear(h_list)
#         val = self.v_linear(h_list)
#         att = torch.matmul(qry, key.transpose(-2, -1)) / math.sqrt(self.output_dim)
#         att = F.softmax(att, dim=-1)
#         att = self.dropout(att)
#         output = torch.matmul(att, val)
#         val_res = self.res(h_list)
#         output += val_res
#         # output = self.relu(output)
#         output = self.norm(output)
#         return output, h_list
