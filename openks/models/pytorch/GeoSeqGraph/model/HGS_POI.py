import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class HardAttn(nn.Module):
    def __init__(self, hidden_size):
        super(HardAttn, self).__init__()
        self.hidden_size = hidden_size
        self.K = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)

        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_normal_(w.weight)

    def forward(self, sess_embed, query, sections, seq_lens):
        v_i = torch.split(sess_embed, sections)
        v_i_pad = pad_sequence(v_i, batch_first=True, padding_value=0.)
        
        v_i_pad = self.K(v_i_pad)
        query = self.Q(query)
        seq_mask = sequence_mask(seq_lens)
        

        attn_weight = (v_i_pad * query.unsqueeze(1)).sum(-1)
        pad_val = (-2 ** 32 + 1) * torch.ones_like(attn_weight)
        attn_weight = torch.where(seq_mask, attn_weight, pad_val).softmax(-1)

        seq_feat = (v_i_pad * attn_weight.unsqueeze(-1)).sum(1)
        return self.V(seq_feat)


class Proj_head(nn.Module):
    def __init__(self, hid_dim):
        super(Proj_head, self).__init__()
        self.hid_dim = hid_dim
        self.lin_head = nn.Linear(hid_dim, hid_dim)
        self.BN = nn.BatchNorm1d(hid_dim)

    def forward(self, x):
        x = self.lin_head(x)
        return self.BN(x)

class EmbeddingLayer(nn.Module):
    def __init__(self, n_poi, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.n_poi = n_poi
        self.embeds = nn.Embedding(n_poi, embed_dim)
        nn.init.xavier_normal_(self.embeds.weight)
    def forward(self, idx):
        return self.embeds(idx)


class GeoGraph(nn.Module):
    def __init__(self, n_user, n_poi, gcn_num, embed_dim, dist_edges, dist_vec, device):
        super(GeoGraph, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.embed_dim = embed_dim
        self.gcn_num = gcn_num
        self.device = device

        self.dist_edges = dist_edges.to(device)
        loop_index = torch.arange(0, n_poi).unsqueeze(0).repeat(2, 1).to(device)
        self.dist_edges = torch.cat(
            (self.dist_edges, self.dist_edges[[1, 0]], loop_index), dim=-1
        )
        dist_vec = np.concatenate((dist_vec, dist_vec, np.zeros(self.n_poi)))
        self.dist_vec = torch.Tensor(dist_vec).to(device)

        self.attn = HardAttn(self.embed_dim).to(device)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

        self.gcn = []
        for _ in range(self.gcn_num):
            self.gcn.append(Geo_GCN(embed_dim, embed_dim, device).to(device))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
   
    def split_mean(self, section_feat, sections):
        section_embed = torch.split(section_feat, sections)
        mean_embeds = [torch.mean(embeddings, dim=0) for embeddings in section_embed]
        return torch.stack(mean_embeds) 

    def forward(self, data, poi_embeds):
        batch_idx = data.batch
        seq_lens = torch.bincount(batch_idx)
        sections = tuple(seq_lens.cpu().numpy())
        enc = poi_embeds.embeds.weight
        for i in range(self.gcn_num):
            enc = self.gcn[i](enc, self.dist_edges, self.dist_vec)
            enc = F.leaky_relu(enc)
            enc = F.normalize(enc, dim=-1)

        tar_embed = enc[data.poi]
        geo_feat = enc[data.x.squeeze()]
        aggr_feat = self.attn(geo_feat, tar_embed, sections, seq_lens)

        graph_enc = self.split_mean(enc[data.x.squeeze()], sections)
        pred_input = torch.cat((aggr_feat, tar_embed), dim=-1)

        pred_logits = self.predictor(pred_input)
        return self.proj_head(graph_enc), pred_logits


class SeqGraph(nn.Module):
    def __init__(self, n_user, n_poi, max_step, embed_dim, hid_graph_num, hid_graph_size, device):
        super(SeqGraph, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.embed_dim = embed_dim
        self.max_step = max_step
        self.encoder = []
        self.poi_embed = nn.Embedding(n_poi, embed_dim)
        nn.init.xavier_normal_(self.poi_embed.weight)

        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )
        self.rwnn = RW_NN(max_step, embed_dim, hid_graph_num, hid_graph_size, device)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)


    def split_mean(self, section_feat, sections):
        section_embed = torch.split(section_feat, sections)
        mean_embeds = [torch.mean(embeddings, dim=0) for embeddings in section_embed]
        return torch.stack(mean_embeds)


    def forward(self, data, poi_embeds):
        tar_poi = data.poi
        sess_feat = self.rwnn(data, poi_embeds)

        tar_embed = poi_embeds(tar_poi)
        pred_input = torch.cat((sess_feat, tar_embed), dim=-1)
        
        pred_logits = self.predictor(pred_input)
        return self.proj_head(sess_feat), pred_logits


class Geo_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(Geo_GCN, self).__init__()
        self.W = nn.Linear(in_channels, out_channels).to(device)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, edge_index, dist_vec):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        dist_weight = torch.exp(-(dist_vec ** 2))
        dist_adj = torch.sparse_coo_tensor(edge_index, dist_weight * norm)
        side_embed = torch.sparse.mm(dist_adj, x)

        return self.W(side_embed)


class RW_NN(MessagePassing):
    def __init__(self, max_step, hid_dim, hid_graph_num, hid_graph_size, device):
        super(RW_NN, self).__init__()
        self.max_step = max_step
        self.device = device
        self.hid_dim = hid_dim
        self.hid_graph_num = hid_graph_num
        self.hid_graph_size = hid_graph_size

        self.__init_weights(hid_dim, hid_graph_num, hid_graph_size)

    def __init_weights(self, hid_dim, hid_graph_num, hid_graph_size):
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.hid_adj = nn.Parameter(
                torch.empty(hid_graph_num, (hid_graph_size * (hid_graph_size - 1)) // 2)
            )
        self.hid_feat = nn.Parameter(torch.empty(hid_graph_num, hid_graph_size, hid_dim))

        self.bn = nn.BatchNorm1d(hid_graph_num * self.max_step)
        self.mlp = torch.nn.Linear(hid_graph_num * self.max_step, hid_dim)

        self.dropout = nn.Dropout()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.hid_adj)
        nn.init.xavier_normal_(self.hid_feat)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, data, poi_embeds):
        poi_feat, poi_adj = poi_embeds(data.x.squeeze()), data.edge_index
        graph_indicator = data.batch
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)

        adj_hidden_norm = torch.zeros(self.hid_graph_num, self.hid_graph_size, self.hid_graph_size).to(self.device)
        idx = torch.triu_indices(self.hid_graph_size, self.hid_graph_size, 1)
        adj_hidden_norm[:,idx[0],idx[1]] = self.relu(self.hid_adj)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2)
        x = self.sigmoid(self.fc(poi_feat))
        z = self.hid_feat
        zx = torch.einsum("abc,dc->abd", (z, x))

        out = []
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.hid_graph_size, device=self.device)
                eye = eye.repeat(self.hid_graph_num, 1, 1)              
                o = torch.einsum("abc,acd->abd", (eye, z))
                t = torch.einsum("abc,dc->abd", (o, x))
            else:
                x = self.propagate(poi_adj, x=x, size=None)
                z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
                t = torch.einsum("abc,dc->abd", (z, x))
            t = self.dropout(t)
            t = torch.mul(zx, t)
            t = torch.zeros(t.size(0), t.size(1), n_graphs, device=self.device).index_add_(2, graph_indicator, t)
            t = torch.sum(t, dim=1)
            t = torch.transpose(t, 0, 1)
            out.append(t)
        
        out = torch.cat(out, dim=1)
        out = self.relu(self.mlp(out))
        out = self.dropout(out)
        return out


def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)
    
    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len, )
    
    return (torch.arange(0, max_len, device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size,max_len)
    .lt(lengths.unsqueeze(1))).reshape(lengths_shape)
