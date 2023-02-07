"""
A model for graph classification, written in pytorch.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU, init
from torch_geometric.nn import GINConv, global_add_pool


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class GINEncoder(nn.Module):
    """ A sequence model for graph representation. """

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

    def forward(self, x, edge_index, batch):
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)

        xpool = global_add_pool(x, batch)
        return xpool


class MLPEncoder(nn.Module):
    def __init__(self, opt):
        super(MLPEncoder, self).__init__()
        if opt["use_Nystroem"] == False:
            self.layer1 = nn.Sequential(   
                nn.Linear(opt["gk_features"], opt["hidden_dim"]),       
                # nn.BatchNorm1d(opt["hidden_dim"]), 
                nn.ReLU(),               
                nn.Dropout(opt["dropout"]),    
            )
        else:
            self.layer1 = nn.Sequential(   
                nn.Linear(opt["nystroem_dim"], opt["hidden_dim"]), 
                # nn.BatchNorm1d(opt["hidden_dim"]), 
                nn.ReLU(),               
                nn.Dropout(opt["dropout"]),  
            ) 
        self.layer2 = nn.Sequential(         
            nn.Linear(opt["hidden_dim"], opt["hidden_dim"]), 
            # nn.BatchNorm1d(opt["hidden_dim"]),
            nn.ReLU(),                 
            nn.Dropout(opt["dropout"]),   
        )
  
        for m in self.modules():
            weight_init(m)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class GINClassifier(nn.Module):
    def __init__(self, opt):
        super(GINClassifier, self).__init__()
        self.hidden_dim = opt["hidden_dim"]
        self.num_class = opt["num_class"]
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


class MemNNEncoder(nn.Module):
    def __init__(self, opt):
        super(MemNNEncoder, self).__init__()
        self.hops = opt["hops"]
        self.hidden_dim = opt["hidden_dim"]
        self.num_classes = opt["num_class"]

        self.dropout = nn.Dropout(opt["dropout"])
        self.leakyrelu = nn.LeakyReLU(0.1)
        # if opt["use_Nystroem"] == False:
        #     self.fc = Sequential(Linear(opt["gk_features"], self.hidden_dim), self.leakyrelu, \
        #                      Linear(self.hidden_dim, self.hidden_dim))
        # else:
        #     self.fc = Sequential(Linear(opt["nystroem_dim"], self.hidden_dim), self.leakyrelu, \
        #                      Linear(self.hidden_dim, self.hidden_dim))
        # if opt["use_Nystroem"] == False:
        #     self.fc = Linear(opt["gk_features"], self.hidden_dim)
        # else:
        #     self.fc = Linear(opt["node_attribute_dim"], self.hidden_dim)   
        self.fc = Linear(opt["node_attribute_dim"], self.hidden_dim)    
        self.A = nn.ModuleList([self.fc for _ in range(self.hops+1)])
        self.B = self.A[0] # query encoder

    def forward(self, x, q):
        # x (bs, memory_len, node_attribute_dim)
        # q (bs, q_sent_len)
        
        bs = x.size(0)
        memory_len = x.size(1)
        node_attribute_dim = x.size(2)

        x = x.view(bs*memory_len, -1) # (bs*memory_len, node_attribute_dim)

        u = self.dropout(self.B(q)) # (bs, hidden_dim)

        # Adjacent weight tying
        for k in range(self.hops):
            m = self.dropout(self.A[k](x))             # (bs*memory_len, hidden_dim)
            m = m.view(bs, memory_len, -1)             # (bs, memory_len, hidden_dim)

            c = self.dropout(self.A[k+1](x))           # (bs*memory_len, hidden_dim)
            c = c.view(bs, memory_len, -1)             # (bs, memory_len, hidden_dim)

            p = torch.bmm(m, u.unsqueeze(2)).squeeze(2) # (bs, memory_len)
            p = F.softmax(p, -1).unsqueeze(1)          # (bs, 1, memory_len)
            o = torch.bmm(p, c).squeeze(1)             # (bs, hidden_dim)
            u = o + u                                  # (bs, hidden_dim)
        return u


class MemNNClassifier(nn.Module):
    def __init__(self, opt):
        super(MemNNClassifier, self).__init__()
        self.hidden_dim = opt["hidden_dim"]
        self.num_class = opt["num_class"]
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.linear2 = nn.Linear(self.hidden_dim//2, self.num_class)

        self.linear1.bias.data.fill_(0)
        self.linear2.bias.data.fill_(0)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, inputs):   
        logits = self.linear1(inputs)
        W = torch.t(self.linear2.weight) # (hidden_dim, vocab_size)
        logits = torch.bmm(logits.unsqueeze(1), W.unsqueeze(0).repeat(logits.shape[0], 1, 1)).squeeze(1)
        return logits