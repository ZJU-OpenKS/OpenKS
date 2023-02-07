import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool


class GINEncoder(nn.Module):
    def __init__(self, opt):
        super(GINEncoder, self).__init__()
        dim = opt["hidden_dim"]
        self.num_gc_layers = opt["num_gc_layers"]

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(self.num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(opt["num_features"], dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

        self.projection = Sequential(Linear(dim, dim//2, bias=False),
                                     ReLU(),
                                     Linear(dim//2, dim, bias=False),
                                    )
                                    
    def forward(self, x, edge_index, batch):
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)

        xpool = global_add_pool(x, batch)
        return xpool



class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.hidden_dim = opt["hidden_dim"]
        self.num_class = opt["num_classes"]
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.linear2 = nn.Linear(self.hidden_dim//2, self.num_class)

        self.linear1.bias.data.fill_(0)
        self.linear2.bias.data.fill_(0)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, inputs):
        logits = self.linear1(inputs)
        out = self.linear2(logits)
        return out
