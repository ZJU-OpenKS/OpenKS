def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import math
import random
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from arguments import arg_parse
from aug import TUDataset_aug
from gin import Encoder
from evaluate_embedding import evaluate_embedding


# remove some invalid nodes, clean function only target at data_aug
def clean(node_num, data_aug):
    # node_num_aug, _ = data_aug.x.size()
    edge_idx = data_aug.edge_index.numpy()
    _, edge_num = edge_idx.shape
    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

    node_num_aug = len(idx_not_missing)
    data_aug.x = data_aug.x[idx_not_missing]

    data_aug.batch = data.batch[idx_not_missing]
    idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
    edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
    data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
    return data_aug


class CLERA(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers):
        super(CLERA, self).__init__()
        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), 
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

    def loss_cal(self, x, x_aug, T=0.2):
        batch_size, _ = x.size()
        out = torch.cat([F.normalize(x), F.normalize(x_aug)], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / T)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(F.normalize(x) * F.normalize(x_aug), dim=-1) / T)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = arg_parse()
    # setup_seed(args.seed)
    num_parts = True
    while num_parts:
        args.num_parts1 = random.randint(2, 4)
        args.num_parts2 = random.randint(2, 4)
        num_parts = args.num_parts1 >= args.num_parts2
    
    accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.DS)
    dataset = TUDataset_aug(path, name=args.DS, aug1=args.aug1, aug2=args.aug2, aug_ratio=args.aug_ratio, \
                            num_parts1=args.num_parts1, num_parts2=args.num_parts2).shuffle()
    dataset_eval = TUDataset(path, name=args.DS).shuffle()
    dataset_num_features = max(dataset_eval.num_features, 1)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)

    print('================')
    print('num_graphs: {}'.format(len(dataset)))
    print('lr: {}'.format(args.lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLERA(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_file = open("./train_loss.txt", mode="a", encoding="utf-8")
    for epoch in range(1, args.epochs+1):
        model.train()
        for data in dataloader:
            optimizer.zero_grad()
            data, data_aug1, data_aug2, subgraph_a, subgraph_b = data
            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch)

            if args.aug1 == 'dnodes' or args.aug1 == 'subgraph' or args.aug1 == 'random4':
                data_aug1 = clean(node_num, data_aug1).to(device)
            else: # 'diffusion', 'pedges', 'mask_nodes'
                data_aug1 = data_aug1.to(device)

            if args.aug2 == 'dnodes' or args.aug2 == 'subgraph' or args.aug2 == 'random4':
                data_aug2 = clean(node_num, data_aug2).to(device)
            else: # 'diffusion', 'pedges', 'mask_nodes'
                data_aug2 = data_aug2.to(device)

            x_aug1 = model(data_aug1.x, data_aug1.edge_index, data_aug1.batch)     
            x_aug2 = model(data_aug2.x, data_aug2.edge_index, data_aug2.batch)
            loss_gg = model.loss_cal(x_aug1, x_aug2)

            subgraph_a = [clean(node_num, subgraph).to(device) for subgraph in subgraph_a]
            subgraph_b = [clean(node_num, subgraph).to(device) for subgraph in subgraph_b]

            subgraph_a_rep = [model(subgraph.x, subgraph.edge_index, subgraph.batch) for subgraph in subgraph_a]
            subgraph_b_rep = [model(subgraph.x, subgraph.edge_index, subgraph.batch) for subgraph in subgraph_b]

            # calculate local
            if len(subgraph_a_rep) == 2 and len(subgraph_b_rep) == 3:
                sub_a1, sub_a2 = subgraph_a_rep[0], subgraph_a_rep[1]
                sub_b1, sub_b2, sub_b3 = subgraph_b_rep[0], subgraph_b_rep[1], subgraph_b_rep[2]

                if args.attention:
                    sub_a1_a2 = sub_a1 * sub_a2
                    query=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a1_a2.unsqueeze(2)), dim=2)
                    key=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a1_a2.unsqueeze(2)), dim=2)
                    value=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a1_a2.unsqueeze(2)), dim=2)
                    d_k = query.size(-1)
                    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
                    p_attn = F.softmax(scores, dim = -1)
                    local_data_a = torch.mean(torch.matmul(p_attn, value), dim=2)

                    sub_b1_b2 = sub_b1 * sub_b2
                    sub_b1_b3 = sub_b1 * sub_b3
                    sub_b2_b3 = sub_b2 * sub_b3
                    query=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b2_b3.unsqueeze(2)), dim=2)
                    key=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b2_b3.unsqueeze(2)), dim=2)
                    value=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b2_b3.unsqueeze(2)), dim=2)
                    d_k = query.size(-1)
                    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
                    p_attn = F.softmax(scores, dim = -1)
                    local_data_b = torch.mean(torch.matmul(p_attn, value), dim=2)
                else:
                    local_data_a, _ = torch.max(torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2)), dim=2), 2)
                    local_data_b, _ = torch.max(torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2)), dim=2), 2)


            elif len(subgraph_a_rep) == 2 and len(subgraph_b_rep) == 4:
                sub_a1, sub_a2 = subgraph_a_rep[0], subgraph_a_rep[1]
                sub_b1, sub_b2, sub_b3, sub_b4 = subgraph_b_rep[0], subgraph_b_rep[1], subgraph_b_rep[2], subgraph_b_rep[3]

                if args.attention:
                    sub_a1_a2 = sub_a1 * sub_a2
                    query=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a1_a2.unsqueeze(2)), dim=2)
                    key=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a1_a2.unsqueeze(2)), dim=2)
                    value=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a1_a2.unsqueeze(2)), dim=2)
                    d_k = query.size(-1)
                    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
                    p_attn = F.softmax(scores, dim = -1)
                    local_data_a = torch.mean(torch.matmul(p_attn, value), dim=2)

                    sub_b1_b2 = sub_b1 * sub_b2
                    sub_b1_b3 = sub_b1 * sub_b3
                    sub_b1_b4 = sub_b1 * sub_b4
                    sub_b2_b3 = sub_b2 * sub_b3
                    sub_b2_b4 = sub_b2 * sub_b4
                    sub_b3_b4 = sub_b3 * sub_b4
                    sub_b1_b2_b3 = sub_b1 * sub_b2 * sub_b3
                    sub_b1_b2_b4 = sub_b1 * sub_b2 * sub_b4
                    sub_b1_b3_b4 = sub_b1 * sub_b3 * sub_b4
                    sub_b2_b3_b4 = sub_b2 * sub_b3 * sub_b4
                    sub_b1_b2_b3_b4 = sub_b1 * sub_b2 * sub_b3 * sub_b4
                    query=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b1_b4.unsqueeze(2), sub_b2_b3.unsqueeze(2), sub_b2_b4.unsqueeze(2), sub_b3_b4.unsqueeze(2), \
                        sub_b1_b2_b3.unsqueeze(2), sub_b1_b2_b4.unsqueeze(2), sub_b1_b3_b4.unsqueeze(2), sub_b2_b3_b4.unsqueeze(2), sub_b1_b2_b3_b4.unsqueeze(2)), dim=2)
                    key=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b1_b4.unsqueeze(2), sub_b2_b3.unsqueeze(2), sub_b2_b4.unsqueeze(2), sub_b3_b4.unsqueeze(2), \
                        sub_b1_b2_b3.unsqueeze(2), sub_b1_b2_b4.unsqueeze(2), sub_b1_b3_b4.unsqueeze(2), sub_b2_b3_b4.unsqueeze(2), sub_b1_b2_b3_b4.unsqueeze(2)), dim=2)
                    value=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b1_b4.unsqueeze(2), sub_b2_b3.unsqueeze(2), sub_b2_b4.unsqueeze(2), sub_b3_b4.unsqueeze(2), \
                        sub_b1_b2_b3.unsqueeze(2), sub_b1_b2_b4.unsqueeze(2), sub_b1_b3_b4.unsqueeze(2), sub_b2_b3_b4.unsqueeze(2), sub_b1_b2_b3_b4.unsqueeze(2)), dim=2)
                    d_k = query.size(-1)
                    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
                    p_attn = F.softmax(scores, dim = -1)
                    local_data_b = torch.mean(torch.matmul(p_attn, value), dim=2)
                else:
                    local_data_a, _ = torch.max(torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2)), dim=2), 2)
                    local_data_b, _ = torch.max(torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), sub_b4.unsqueeze(2)), dim=2), 2)

            elif len(subgraph_a_rep) == 3 and len(subgraph_b_rep) == 4:
                sub_a1, sub_a2, sub_a3 = subgraph_a_rep[0], subgraph_a_rep[1], subgraph_a_rep[2]
                sub_b1, sub_b2, sub_b3, sub_b4 = subgraph_b_rep[0], subgraph_b_rep[1], subgraph_b_rep[2], subgraph_b_rep[3]

                if args.attention:
                    sub_a1_a2 = sub_a1 * sub_a2
                    sub_a1_a3 = sub_a1 * sub_a3
                    sub_a2_a3 = sub_a2 * sub_a3
                    query=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a3.unsqueeze(2), \
                        sub_a1_a2.unsqueeze(2), sub_a1_a3.unsqueeze(2), sub_a2_a3.unsqueeze(2)), dim=2)
                    key=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a3.unsqueeze(2), \
                        sub_a1_a2.unsqueeze(2), sub_a1_a3.unsqueeze(2), sub_a2_a3.unsqueeze(2)), dim=2)
                    value=torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a3.unsqueeze(2), \
                        sub_a1_a2.unsqueeze(2), sub_a1_a3.unsqueeze(2), sub_a2_a3.unsqueeze(2)), dim=2)
                    d_k = query.size(-1)
                    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
                    p_attn = F.softmax(scores, dim = -1)
                    local_data_a = torch.mean(torch.matmul(p_attn, value), dim=2)

                    sub_b1_b2 = sub_b1 * sub_b2
                    sub_b1_b3 = sub_b1 * sub_b3
                    sub_b1_b4 = sub_b1 * sub_b4
                    sub_b2_b3 = sub_b2 * sub_b3
                    sub_b2_b4 = sub_b2 * sub_b4
                    sub_b3_b4 = sub_b3 * sub_b4
                    sub_b1_b2_b3 = sub_b1 * sub_b2 * sub_b3
                    sub_b1_b2_b4 = sub_b1 * sub_b2 * sub_b4
                    sub_b1_b3_b4 = sub_b1 * sub_b3 * sub_b4
                    sub_b2_b3_b4 = sub_b2 * sub_b3 * sub_b4
                    sub_b1_b2_b3_b4 = sub_b1 * sub_b2 * sub_b3 * sub_b4
                    query=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b1_b4.unsqueeze(2), sub_b2_b3.unsqueeze(2), sub_b2_b4.unsqueeze(2), sub_b3_b4.unsqueeze(2), \
                        sub_b1_b2_b3.unsqueeze(2), sub_b1_b2_b4.unsqueeze(2), sub_b1_b3_b4.unsqueeze(2), sub_b2_b3_b4.unsqueeze(2), sub_b1_b2_b3_b4.unsqueeze(2)), dim=2)
                    key=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b1_b4.unsqueeze(2), sub_b2_b3.unsqueeze(2), sub_b2_b4.unsqueeze(2), sub_b3_b4.unsqueeze(2), \
                        sub_b1_b2_b3.unsqueeze(2), sub_b1_b2_b4.unsqueeze(2), sub_b1_b3_b4.unsqueeze(2), sub_b2_b3_b4.unsqueeze(2), sub_b1_b2_b3_b4.unsqueeze(2)), dim=2)
                    value=torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), \
                        sub_b1_b2.unsqueeze(2), sub_b1_b3.unsqueeze(2), sub_b1_b4.unsqueeze(2), sub_b2_b3.unsqueeze(2), sub_b2_b4.unsqueeze(2), sub_b3_b4.unsqueeze(2), \
                        sub_b1_b2_b3.unsqueeze(2), sub_b1_b2_b4.unsqueeze(2), sub_b1_b3_b4.unsqueeze(2), sub_b2_b3_b4.unsqueeze(2), sub_b1_b2_b3_b4.unsqueeze(2)), dim=2)
                    d_k = query.size(-1)
                    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
                    p_attn = F.softmax(scores, dim = -1)
                    local_data_b = torch.mean(torch.matmul(p_attn, value), dim=2)
                else:
                    local_data_a, _ = torch.max(torch.cat((sub_a1.unsqueeze(2), sub_a2.unsqueeze(2), sub_a3.unsqueeze(2)), dim=2), 2)
                    local_data_b, _ = torch.max(torch.cat((sub_b1.unsqueeze(2), sub_b2.unsqueeze(2), sub_b3.unsqueeze(2), sub_b4.unsqueeze(2)), dim=2), 2)
            else:
                print('Cluster Partition Error')

            # Local-Global
            loss_lg = model.loss_cal(x, local_data_a)
            # Local-Local
            loss_ll = model.loss_cal(local_data_a, local_data_b)

            loss = loss_gg + loss_ll + loss_lg
            loss.backward()
            optimizer.step()
        print('===== Epoch {}, Loss {} ====='.format(epoch, loss.item()))
        loss_file.write(str(loss.item()) + "\n")

        if epoch % args.log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            res = evaluate_embedding(emb, y)
            # accuracies['logreg'].append(res[0])
            accuracies['svc'].append(res)
            # accuracies['linearsvc'].append(res[2])
            # accuracies['randomforest'].append(res[3])
            print(accuracies)
    loss_file.write("\n")
    loss_file.close()

    result_file = open("./result_table.txt", mode="a", encoding="utf-8")
    result_file.write("%s"%args.DS+ "\t" + "%s"%args.num_parts1 + "\t" + "%s"%args.num_parts2 +  "\n")
    for name in ['logreg', 'svc', 'linearsvc', 'randomforest']:
        result_file.write(name + "\t" + str(accuracies[name]) + "\n")
    result_file.write("\n")
    result_file.close()




