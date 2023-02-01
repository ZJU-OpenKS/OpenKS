import torch, torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class CTR_Model_HAW(nn.Module):
    def __init__(self, n_user, n_item, embed_dim, hid, beta, entire_graph, K,
                 nConvs, max_len, device, max_time, min_time, tau, w):
        super(CTR_Model_HAW, self).__init__()
        self.device = device
        self.n_user = n_user
        self.n_item = n_item
        self.n_node = n_user + n_item + 1 # for padding
        self.nConvs = nConvs
        self.K = K
        self.embed_dim = embed_dim
        self.hid = hid
        self.beta = beta
        self.max_len = max_len
        self.max_time = max_time
        self.min_time = min_time
        self.tau = tau

        self.init_weight(entire_graph)
            

    def forward(self, uid, sid, sub_nodes, sub_edges, seqs, clk_time, tar_edges, tar_nodes):
        batch_size = uid.size(0)
        sub_graphs = []
        sid += self.n_user

        for nodes, edges in zip(sub_nodes, sub_edges):
            sub_graphs.append(Data(x=self.embed[nodes], edge_index=edges))
        for nodes, edges in zip(tar_nodes, tar_edges):
            sub_graphs.append(Data(x=self.embed[nodes], edge_index=edges))
        graph_batch = Batch().from_data_list(sub_graphs).to(self.device)

        seq_map, cur_idx = [], 0
        for i, seq in enumerate(seqs):
            seq_map.append(torch.arange(cur_idx, cur_idx + seq.size(0)))
            cur_idx += sub_nodes[i].size(0)
        seq_map = torch.cat(seq_map)

        tar_map = []
        for i, nodes in enumerate(tar_nodes):
            tar_map.append(cur_idx)
            cur_idx += len(tar_nodes[i])
        tar_map = torch.LongTensor(tar_map)


        # Disentagled Graph Convolution
        x = graph_batch.x
        resi_embed = x[seq_map]
        s_list = [torch.ones((self.K, graph_batch.num_edges)).to(self.device)]
        for i in range(self.nConvs):
            s_q = s_list[-1].softmax(dim=0)
            s_tmp = []
            for k in range(self.K):
                x[:, k, :] = self.DisConvs(
                    x=x[:, k, :],
                    edge_index=graph_batch.edge_index,
                    edge_weight=s_q[k]
                )

                z = x[graph_batch.edge_index[0], k, :] * torch.tanh(x[graph_batch.edge_index[1], k, :])
                s_tmp.append(s_list[-1][k] + z.sum(-1))

            s_list.append(torch.stack(s_tmp, dim=0))

        # Generate Contextual Representationon
        seq_embed = self.W1(x[seq_map]) + resi_embed
        x_out, cur_idx = [], 0
        for i, seq in enumerate(seqs):
            x_out.append(seq_embed[cur_idx: cur_idx + seq.size(0)])
            cur_idx += seq.size(0)

        a_v = self.W2(x[tar_map]) + self.embed[sid]
        return self.hawkes_enc(x_out, seqs, clk_time, uid, a_v, sid)

    def similarity(self, n1, n2):
        return torch.cosine_similarity(n1, n2, dim=-1)

    def hawkes_enc(self, seq_embed_list, seqs, clk_time, uid, a_v, sid):
        seq_idx = pad_sequence([seq[:, 0] for seq in seqs], batch_first=True, padding_value=0)
        a_u = self.embed[uid]

        # [bs, seq_len, K, hid_dim]
        seq_embed = pad_sequence(seq_embed_list, batch_first=True, padding_value=0)
        seqs_time = [t[: , 1] for t in seqs]
        seqs_time = pad_sequence(seqs_time, batch_first=True, padding_value=self.min_time).type_as(clk_time)

        # [bs, seq_len]
        seq_len = torch.LongTensor([seq.size(0) for seq in seqs])
        seq_mask = sequence_mask(seq_len).to(self.device)
        
        # [bs, K]
        mu_u_v = self.similarity(a_u, a_v)

        # [bs, seq_len, K]
        gamma_h_v = self.similarity(seq_embed, a_v.unsqueeze(1))
        
        # [bs, seq_len]
        j = torch.exp(-(clk_time.view(-1, 1) - seqs_time) / (self.max_time - self.min_time) * self.delta[uid].view(-1, 1))
        pad = torch.zeros_like(j)
        j = torch.where(seq_mask, j, pad)
        
        # Gumbel Max trick
        logits = self.similarity(seq_embed, a_u.unsqueeze(1))
        eps = 0.0001
        u_k = (eps - (1 - eps)) * torch.rand_like(logits) + (1 - eps)
        g_k = -torch.log(-torch.log(u_k))
        pi = ((logits + g_k) / self.tau[seq_idx].unsqueeze(-1)).softmax(-1)

        inc = j.unsqueeze(-1) * gamma_h_v * pi
        return (mu_u_v + inc.sum(1)).mean(-1)

    def init_weight(self, entire_graph):
        self.embed = nn.Parameter(
            torch.zeros((self.n_node, self.K, self.embed_dim))
        )
        nn.init.xavier_uniform_(self.embed)

        self.delta = nn.Parameter(torch.ones((self.n_user, )))
        self.tau = nn.Parameter(torch.ones(self.n_item, ))

        self.DisConvs = DisentangleConv()
        # self.S_mat = nn.Parameter(torch.randn(self.K, self.embed_dim, self.embed_dim))
        self.S = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.S.weight)

        self.W1 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.W1.weight)
        self.W2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.W2.weight)


class DisentangleConv(MessagePassing):
    def __init__(self, aggr='add'):
        super(DisentangleConv, self).__init__(aggr=aggr)
    
    def forward(self, x, edge_index, edge_weight):
        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, embed_dim]
        return norm.view(-1, 1) * x_j


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
