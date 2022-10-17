#!/usr/bin/env python
# coding: utf-8

# @Time    : 2022/5/20 
# @Author  : Tongdunkeji Fuguohui
# @FileName: homo_graph_link_predictor.py
### Dependencies
# dgl == 0.8.1

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
import tqdm
import sklearn.metrics
from sklearn.metrics import roc_auc_score , f1_score
import warnings
warnings.filterwarnings("ignore")

"""
本算法主要实现同构图中链接预测，链接预测就是预测图中给定节点间是否存在边。
采用负采样，利用三层GraphSAGE模型来表示节点，后接三层全连接层来预测目标。

"""

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, aggregator_type, dropout, activation):
        super().__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator_type, dropout, activation))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type, dropout, activation))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type, dropout))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 2),
            nn.Softmax(dim=1) 
            )

    def predict(self, h_src, h_dst):
        
        return self.predictor(h_src * h_dst)[:,1]


    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        pos_score = self.predict(h[pos_src], h[pos_dst])
        neg_score = self.predict(h[neg_src], h[neg_dst])
        return pos_score, neg_score

def split_graph_edge(graph, non_train_ratio=0.2):
    '''
    将graph划分为训练集、验证集、测试集。
    '''
    edges = graph.edges('eid')
    e = graph.num_edges()
    
    edges = np.random.permutation(edges)
    split1 = int((1 - non_train_ratio) * e)
    split2 = int((1 - non_train_ratio / 2) * e)

    edges_train = edges[:split1]
    edges_val = edges[split1:split2]
    edges_test = edges[split2:]

    graph_train = dgl.edge_subgraph(graph, edges_train, relabel_nodes=False) 
    graph_val = dgl.edge_subgraph(graph, edges_val, relabel_nodes=False) 
    graph_test = dgl.edge_subgraph(graph, edges_test, relabel_nodes=False) 
    
    graph_split={'train':graph_train,'val':graph_val,'test':graph_test}
    
    sampler = dgl.dataloading.NeighborSampler([15, 10, 5], prefetch_node_feats=['feat'])

    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5))

    train_dataloader = dgl.dataloading.DataLoader(
            graph_split['train'], graph_split['train'].edges('eid'), sampler,
            device=device, batch_size=256, shuffle=True,
            drop_last=False, num_workers=0, use_uva= False)

    val_dataloader = dgl.dataloading.DataLoader(
            graph_split['val'], graph_split['val'].edges('eid'), sampler,
            device=device, batch_size=256, shuffle=True,
            drop_last=False, num_workers=0, use_uva= False)

    test_dataloader = dgl.dataloading.DataLoader(
            graph_split['test'], graph_split['test'].edges('eid'), sampler,
            device=device, batch_size=256, shuffle=True,
            drop_last=False, num_workers=0, use_uva= False)

    dataloader_split={'train':train_dataloader,'val':val_dataloader,'test':test_dataloader}
    
    return graph_split , dataloader_split  

def homo_graph_link_predictor(graph,epoches,non_train_ratio,device,n_hidden, aggregator_type, dropout, activation,learning_rate):
    
    # split dataset into train, validate, test
    graph_split , dataloader_split = split_graph_edge(graph, non_train_ratio)

    # create model
    model = GraphSAGE(graph.ndata['feat'].shape[1], n_hidden,aggregator_type, dropout, activation).to(device)
    
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    durations = []
    
    # training loop
    for epoch in range(epoches):
        model.train()
        t0 = time.time()
        train_preds = []
        train_labels = []
        val_preds = []
        val_labels = []
        test_preds = []
        test_labels = []

        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader_split['train']):
            x = blocks[0].srcdata['feat']
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
    #         print(pos_score)
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            score = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            train_preds.append(score)
            train_labels.append(labels)
    #         loss = F.binary_cross_entropy_with_logits(score, labels)
            loss = F.binary_cross_entropy(score, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if (it + 1) % 10 == 0:
                mem = torch.cuda.max_memory_allocated() / 1000000
                print('epoch:',epoch, 'Loss:', loss.item(), 'GPU Mem:', mem, 'MB')
                if (it + 1) == 50:
                    tt = time.time()
                    #print(tt - t0)
                    durations.append(tt - t0)
                    
        if epoch % 10 == 0:
            model.eval()

            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader_split['val']):
                x = blocks[0].srcdata['feat']
                pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                score = torch.cat([pos_score, neg_score])
                labels = torch.cat([pos_label, neg_label])
                val_preds.append(score)
                val_labels.append(labels)

            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader_split['test']):
                x = blocks[0].srcdata['feat']
                pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                score = torch.cat([pos_score, neg_score])
                labels = torch.cat([pos_label, neg_label])
                test_preds.append(score)
                test_labels.append(labels)

            train_preds = torch.cat(train_preds).cpu().detach().numpy()
            train_labels = torch.cat(train_labels).cpu().detach().numpy()
            val_preds = torch.cat(val_preds).cpu().detach().numpy()
            val_labels = torch.cat(val_labels).cpu().detach().numpy()
            test_preds = torch.cat(test_preds).cpu().detach().numpy()
            test_labels = torch.cat(test_labels).cpu().detach().numpy()

            train_auc = roc_auc_score(train_labels , train_preds)
            val_auc = roc_auc_score(val_labels , val_preds)
            test_auc = roc_auc_score(test_labels , test_preds)

            model.train()

            print('epoch:',epoch , 'Loss:', round(loss.item(),4) , 'train_auc:', round(train_auc,4), 'val_auc:', round(val_auc,4),'test_auc:', round(test_auc,4))
    print('durations mean:',np.mean(durations[4:]), 'durations std:',np.std(durations[4:]))

    return model

if __name__ == "__main__":
    device='cpu'
    # 非训练集占比
    non_train_ratio=0.2
    epoches=11
    n_hidden =256 
    aggregator_type = 'mean'
    dropout = 0.5
    activation = F.relu
    learning_rate =0.01

    src = np.array(range(100000))

    dst = np.random.permutation(src)
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    graph = dgl.graph((u, v))
    
    graph.ndata['feat'] = torch.randn(graph.number_of_nodes(), 10)

    graph = dgl.remove_self_loop(graph)
    graph = graph.to(device)


    model = homo_graph_link_predictor(graph,epoches,non_train_ratio,device,n_hidden, aggregator_type, dropout, activation,learning_rate)
