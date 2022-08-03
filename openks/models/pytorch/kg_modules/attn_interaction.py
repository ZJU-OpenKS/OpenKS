from typing import Union, Optional, Callable

import torch
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch import nn
from ...model import TorchModel


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_encoder = torch.nn.Linear(4, emb_dim)
        # self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
            out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        else:
            out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=None))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(4, emb_dim)
        # self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if edge_attr is not None:
            # edge_embedding = self.bond_encoder(edge_attr)
            edge_embedding = self.edge_encoder(edge_attr)
        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        if edge_attr is not None:
            return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) \
                   + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)
        else:
            return self.propagate(edge_index, x=x, edge_attr=None, norm=norm) + F.relu(
                x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr is None:
            return norm.view(-1, 1) * F.relu(x_j)
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0.0, JK="last", residual=False, gnn_type='gin', feat_dim=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.fc = nn.Linear(feat_dim, emb_dim, bias=False)
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # self.atom_encoder = AtomEncoder(emb_dim)
        self.node_encoder = torch.nn.Embedding(feat_dim, emb_dim)  # uniform input node embedding

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr=None):
        # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        tmp = self.fc(x)

        h_list = [tmp]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation


# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, dim, hidden_size, projection_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class AttnPooling(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5,
                 min_score: Optional[float] = None, multiplier: float = 1.0,
                 nonlinearity: Callable = torch.tanh, **kwargs):
        super(AttnPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

    def forward(self, x, edge_index, query_vec, edge_attr=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = torch.einsum('ik,k->i', attn, query_vec)
        # score = self.gnn(attn, edge_index).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]


@TorchModel.register("AttInter", "PyTorch")
class AttInter(torch.nn.Module):
    def __init__(self, att_config):
        super(AttInter, self).__init__()

        self.num_layer = att_config.num_conv_layer
        self.feat_dim = att_config.feat_dim
        self.emb_dim = att_config.conv_emb_dim
        self.hid_dim = att_config.pred_hid_dim
        self.graph_pooling = att_config.graph_pooling
        self.device = att_config.device

        self.attn_pool = AttnPooling(att_config.in_channels, att_config.ratio)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.gnn_node = GNN_node(self.num_layer, self.emb_dim, JK=att_config.JK,
                                 drop_ratio=att_config.conv_drop_ratio, residual=False,
                                 gnn_type=att_config.gnn_type, feat_dim=self.feat_dim)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(torch.nn.Linear(self.emb_dim, 2 * self.emb_dim),
                                                                    torch.nn.BatchNorm1d(2 * self.emb_dim),
                                                                    torch.nn.ReLU(),
                                                                    torch.nn.Linear(2 * self.emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(self.emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.predictor = MLP(self.emb_dim+self.feat_dim, self.hid_dim, 2)

    def forward(self, emb_list):
        text_emb, demand_kg_emb, all_kg = emb_list

        x, edge_index, _, _, _, _ = self.attn_pool(all_kg.x, all_kg.edge_index, demand_kg_emb.flatten())

        sub_kg_node_embs = self.gnn_node(x, edge_index)
        sub_kg_emb = self.pool(sub_kg_node_embs, torch.zeros(sub_kg_node_embs.shape[0],
                                                             dtype=torch.long).to(self.device))

        vec_cat = torch.cat((sub_kg_emb, text_emb), dim=-1)
        logits = self.predictor(vec_cat)

        return logits






