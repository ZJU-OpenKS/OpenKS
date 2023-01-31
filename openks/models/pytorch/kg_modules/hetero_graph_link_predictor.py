#!/usr/bin/env python
# coding: utf-8

# @Time    : 2022/5/20 
# @Author  : Tongdunkeji Fuguohui
# @FileName: hetero_graph_link_predictor.py
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
本算法主要实现异构图中链接预测，链接预测就是预测图中给定节点间是否存在边。
采用负采样，利用2层RGCN模型来表示节点，后接2层全连接层来预测目标。

"""
class RGCN_link_predictor(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.hidden_feat = hidden_feat
        self.layers = nn.ModuleList()
        self.layers.append( dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, hidden_feat, norm='right',activation=F.relu)
                for rel in rel_names
            }))
        self.layers.append( dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                for rel in rel_names
            }))
        self.predictor = nn.Sequential(
            nn.Linear(out_feat, out_feat),
            nn.ReLU(),
            nn.Linear(out_feat, 2),
            nn.Softmax(dim=1) 
            )

    
    def predict(self, edge_subgraph , h):
        with edge_subgraph.local_scope():
            edge_score = dict()
            scores=[]
            for canonical_etype in edge_subgraph.canonical_etypes:
                src_type, etype, dst_type =canonical_etype
                src , dst = edge_subgraph.edges(etype=etype)
                score = self.predictor(h[src_type][src] * h[dst_type][dst])[:,1]

                edge_score[etype] = score
                scores.append(score)
            
        return edge_score, scores


    def forward(self, positive_graph, negative_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

        pos_edge_score, pos_score = self.predict(positive_graph,h)
        neg_edge_score, neg_score = self.predict(negative_graph,h)
        
        return pos_edge_score, pos_score, neg_edge_score,neg_score
    
def split_hetero_graph_edge(graph,reverse_etypes, non_train_ratio=0.2):
    '''
    将graph划分为训练集、验证集、测试集。
    '''
    
    edges_train_dict = {}
    edges_val_dict = {}
    edges_test_dict = {}
    for etype in graph.etypes:
        edges = graph.edges('eid',etype = etype)
        e = len(edges)

        edges = np.random.permutation(edges)
        split1 = int((1 - non_train_ratio) * e)
        split2 = int((1 - non_train_ratio / 2) * e)

        edges_train = edges[:split1]
        edges_val = edges[split1:split2]
        edges_test = edges[split2:]

        edges_train_dict.update({etype : edges_train})
        edges_val_dict.update({etype : edges_val})
        edges_test_dict.update({etype : edges_test})

    graph_train = dgl.edge_subgraph(graph, edges_train_dict, relabel_nodes=False ,store_ids = True) 
    graph_val = dgl.edge_subgraph(graph, edges_val_dict, relabel_nodes=False,store_ids = True) 
    graph_test = dgl.edge_subgraph(graph, edges_test_dict, relabel_nodes=False,store_ids = True) 
    
    graph_split={'train':graph_train,'val':graph_val,'test':graph_test}
    
    sampler = dgl.dataloading.NeighborSampler([10, 5])

    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        exclude='reverse_types',
        reverse_etypes=reverse_etypes)
    
    train_eid_dict = {etype: graph_train.edges(etype=etype, form='eid')for etype in graph_train.etypes}
    val_eid_dict = {etype: graph_val.edges(etype=etype, form='eid')for etype in graph_val.etypes}
    test_eid_dict = {etype: graph_test.edges(etype=etype, form='eid')for etype in graph_test.etypes}
    
    train_dataloader = dgl.dataloading.DataLoader(
            graph_train, train_eid_dict, sampler,
            device=device, batch_size=1024, shuffle=True,
            drop_last=False, num_workers=0, use_uva= False)

    val_dataloader = dgl.dataloading.DataLoader(
            graph_val, val_eid_dict, sampler,
            device=device, batch_size=1024, shuffle=True,
            drop_last=False, num_workers=0, use_uva= False)

    test_dataloader = dgl.dataloading.DataLoader(
            graph_test, test_eid_dict, sampler,
            device=device, batch_size=1024, shuffle=True,
            drop_last=False, num_workers=0, use_uva= False)

    dataloader_split={'train':train_dataloader,'val':val_dataloader,'test':test_dataloader}
    
    return graph_split , dataloader_split  

def hetero_graph_link_predictor(graph, epoches, non_train_ratio, device, n_hidden, n_out, learning_rate, reverse_etypes):
    
    etypes = graph.etypes
    
    in_features = graph.ndata['feat'][graph.ntypes[0]].shape[1]
    
    model = RGCN_link_predictor(in_features, n_hidden, n_out, etypes)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    graph_split , dataloader_split = split_hetero_graph_edge(graph, reverse_etypes, non_train_ratio)
    
    # training loop
    for epoch in range(epoches):
        
        train_preds = []
        train_labels = []
        val_preds = []
        val_labels = []
        test_preds = []
        test_labels = []

        for it, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader_split['train']):
        
            input_features = blocks[0].srcdata['feat']
            
            pos_edge_score, pos_score, neg_edge_score,neg_score = model(positive_graph, negative_graph, blocks, input_features)
            
            pos_score = torch.cat(pos_score)
            neg_score = torch.cat(neg_score)
            
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            score = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            train_preds.append(score)
            train_labels.append(labels)

            loss = F.binary_cross_entropy(score, labels)         
            #print('pos_score',pos_score)
            opt.zero_grad()
            loss.backward()

            opt.step()

            if (it + 1) % 10 == 0:
                mem = torch.cuda.max_memory_allocated() / 1000000
                print('epoch:',epoch, 'Loss:', loss.item(), 'GPU Mem:', mem, 'MB')

        if epoch % 10 == 0 or True:
            model.eval()

            for it, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader_split['val']):
                x = blocks[0].srcdata['feat']

                pos_edge_score, pos_score, neg_edge_score,neg_score = model(positive_graph, negative_graph, blocks, x)
                pos_score = torch.cat(pos_score)
                neg_score = torch.cat(neg_score)
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                score = torch.cat([pos_score, neg_score])
                labels = torch.cat([pos_label, neg_label])
                val_preds.append(score)
                val_labels.append(labels)

            for it, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader_split['test']):
                x = blocks[0].srcdata['feat']
                pos_edge_score, pos_score, neg_edge_score,neg_score = model(positive_graph, negative_graph, blocks, x)
                pos_score = torch.cat(pos_score)
                neg_score = torch.cat(neg_score)
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

    return model

    
    
    
if __name__ == "__main__":
    
    device = 'cpu'
    n_hidden = 128
    n_out = 64
    epoches = 131
    learning_rate = 0.01
    non_train_ratio = 0.2

    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 10

    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)

    graph = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src),
        ('user', 'click', 'item'): (click_src, click_dst),
        ('item', 'clicked-by', 'user'): (click_dst, click_src),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
        ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})

    graph.nodes['user'].data['feat'] = torch.randn(n_users, n_hetero_features)
    graph.nodes['item'].data['feat'] = torch.randn(n_items, n_hetero_features)
    
    reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click','follow':'followed-by','followed-by':'follow','dislike':'disliked-by','disliked-by':'dislike'}
    
    model = hetero_graph_link_predictor(graph, epoches, non_train_ratio, device, n_hidden, n_out, learning_rate, reverse_etypes)
    
