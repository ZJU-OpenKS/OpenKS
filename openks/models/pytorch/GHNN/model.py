import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn.conv import MessagePassing


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.encoder = Encoder(args.num_features, self.hidden_dim, args.num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), 
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc = nn.Linear(self.hidden_dim, args.num_classes)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index, batch)
        z = self.proj_head(x)
        p = self.fc(x)
        return z, p


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            # x = self.bns[i](x)
            xs.append(x)

        xpool = global_add_pool(x, batch)
        return xpool


class RW_NN(MessagePassing):
    def __init__(self, args):
        super(RW_NN, self).__init__()
        self.max_step = args.max_step
        self.hidden_graphs = args.hidden_graphs
        self.size_hidden_graphs = args.size_hidden_graphs
        self.normalize = args.normalize
        self.device = device
        self.adj_hidden = Parameter(torch.FloatTensor(args.hidden_graphs, (args.size_hidden_graphs*(args.size_hidden_graphs-1))//2))
        self.features_hidden = Parameter(torch.FloatTensor(args.hidden_graphs, args.size_hidden_graphs, args.hidden_dim))
        self.fc = torch.nn.Linear(args.num_features, args.hidden_dim)
        self.bn = nn.BatchNorm1d(args.hidden_graphs * args.max_step)
        self.fc1 = torch.nn.Linear(args.hidden_graphs * args.max_step, args.penultimate_dim)
        self.fc2 = torch.nn.Linear(args.penultimate_dim, args.num_classes)
        self.dropout = nn.Dropout(p=args.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.proj_head = nn.Sequential(nn.Linear(args.penultimate_dim, args.hidden_dim), 
                                       nn.ReLU(inplace=True),
                                       nn.Linear(args.hidden_dim, args.hidden_dim))
        self.init_weights()

    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
    #     self.init_emb()

    # def init_emb(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.fill_(0.0)
        
    def forward(self, data):    
        adj = data.edge_index
        features = data.x
        graph_indicator = data.batch
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)
        n_nodes = features.size(0)

        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.hidden_graphs)
        
        adj_hidden_norm = torch.zeros(self.hidden_graphs, self.size_hidden_graphs, self.size_hidden_graphs).to(self.device)
        idx = torch.triu_indices(self.size_hidden_graphs, self.size_hidden_graphs, 1)
        adj_hidden_norm[:,idx[0],idx[1]] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2)
        x = self.sigmoid(self.fc(features))
        z = self.features_hidden
        zx = torch.einsum("abc,dc->abd", (z, x))
        
        out = list()
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_hidden_graphs, device=self.device)
                eye = eye.repeat(self.hidden_graphs, 1, 1)              
                o = torch.einsum("abc,acd->abd", (eye, z))
                t = torch.einsum("abc,dc->abd", (o, x))
            else:
                # x = torch.spmm(adj, x)
                x = self.propagate(adj, x=x, size=None)
                z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
                t = torch.einsum("abc,dc->abd", (z, x))
            t = self.dropout(t)
            t = torch.mul(zx, t)
            t = torch.zeros(t.size(0), t.size(1), n_graphs, device=self.device).index_add_(2, graph_indicator, t)
            t = torch.sum(t, dim=1)
            t = torch.transpose(t, 0, 1)
            if self.normalize:
                t /= norm
            out.append(t)
            
        out = torch.cat(out, dim=1)
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        z = self.proj_head(out)
        out = self.fc2(out)
        return z, out



def contrastive_loss(x, x_aug, T=0.2):
    batch_size, _ = x.size()
    out = torch.cat([F.normalize(x), F.normalize(x_aug)], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / T)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(F.normalize(x) * F.normalize(x_aug), dim=-1) / T)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def sharpen(p, T):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p


def get_entropy(scores):
    scores = F.softmax(scores, dim=1)
    entropy = torch.sum(-scores * torch.log(scores + 1e-5))
    return entropy


def train_student(tau, data, teacher, student):
    data, data_aug = data
    data = data.to(device)
    data_aug = data_aug.to(device)

    teacher_feats, teacher_p = teacher(data.x, data.edge_index, data.batch)
    teacher_feats_aug, _ = teacher(data_aug.x, data_aug.edge_index, data_aug.batch)
    student_feats, student_p = student(data)
    student_feats_aug, _ = student(data_aug)
    
    entropy_teacher = get_entropy(teacher_p)
    entropy_student = get_entropy(student_p)
    weight_t = tau * torch.exp(-entropy_teacher)
    weight_s = tau * torch.exp(-entropy_student)

    loss_harmony = 0.5 * ((1 + weight_s) * contrastive_loss(teacher_feats, teacher_feats_aug) + \
                          (1 + weight_t) * contrastive_loss(student_feats, student_feats_aug))

    with torch.no_grad():
        teacher_p = F.softmax(teacher_p, dim=1)
        student_p = F.softmax(student_p, dim=1)
        target_t = sharpen(teacher_p, T=0.25)
        target_t[target_t < 1e-4] *= 0  # numerical stability
        target_s = sharpen(student_p, T=0.25)
        target_s[target_s < 1e-4] *= 0  # numerical stability

    # compute cross-entropy loss H(targets, queries)
    loss_cross = 0.5 * (torch.mean(torch.sum((-target_t) * torch.log(student_p + 1e-5), dim=1)) + \
                        torch.mean(torch.sum((-target_s) * torch.log(teacher_p + 1e-5), dim=1)))

    # loss_tc = 0.5 * ((1 / (1 + weight_t)) * max(0, abs(F.softmax(teacher_p, dim=1).mean() - F.softmax(student_p, dim=1).mean()) - 0.2) + \
    #                  (1 / (1 + weight_s)) * max(0, abs(F.softmax(student_p, dim=1).mean() - F.softmax(teacher_p, dim=1).mean()) - 0.2))
    loss = loss_harmony + loss_cross
    return loss