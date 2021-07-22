# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University.
# All Rights Reserved.

import logging
import torch
import torch.nn as nn
import numpy as np
from ...model import TorchModel
from torch.nn import functional as F

@TorchModel.register("DyE", "PyTorch")
class DyE(TorchModel):
    def __init__(self, **kwargs):
        super(DyE, self).__init__()
        self.num_entity = kwargs['num_entity']
        self.num_relation = kwargs['num_relation']
        self.hidden_size = kwargs['hidden_size']
        self.margin = kwargs['margin']
        self.norm = 1

        uniform_range = 6 / np.sqrt(self.hidden_size)
        self.updater_inp = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.entity_updater = nn.GRUCell(self.hidden_size, self.hidden_size)
        # self.entities_emb = nn.Embedding(self.num_entity, self.hidden_size)
        # self.relations_emb = nn.Embedding(self.num_relation, self.hidden_size)
        # self.entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # self.relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        self.initial_entities_emb = nn.Parameter(F.normalize(torch.rand(self.hidden_size).cuda(), dim=0))
        self.entities_emb = self.initial_entities_emb.repeat(self.num_entity, 1)
        self.relations_emb = nn.Embedding(self.num_relation, self.hidden_size)
        self.relations_emb.weight.data.uniform_(-uniform_range, uniform_range)

    def _algorithm(self, pos_triples, neg_triples):
        """ graph embedding similarity algorithm method """

        pos_scores = []
        neg_scores = []
        print(pos_triples)

        for pos_triple, neg_triple in zip(pos_triples, neg_triples):
            pos_head = pos_triple[0].view(-1)
            pos_relation = pos_triple[1].view(-1)
            pos_tail = pos_triple[2].view(-1)

            pos_score = self.entities_emb[pos_head] + self.relations_emb(pos_relation) - self.entities_emb[pos_tail]
            pos_score = pos_score.norm(p=self.norm, dim=1)
            pos_scores.append(pos_score)

            neg_head = neg_triple[0].view(-1)
            neg_relation = neg_triple[1].view(-1)
            neg_tail = neg_triple[2].view(-1)
            neg_score = self.entities_emb[neg_head] + self.relations_emb(neg_relation) - self.entities_emb[neg_tail]
            neg_score = neg_score.norm(p=self.norm, dim=1)
            neg_scores.append(neg_score)

            head_inp = self.updater_inp(
                torch.cat((self.entities_emb[pos_tail], self.relations_emb(pos_relation)), dim=1))
            tail_inp = self.updater_inp(
                torch.cat((self.entities_emb[pos_head], self.relations_emb(pos_relation)), dim=1))
            head_out = self.entity_updater(head_inp, self.entities_emb[pos_head])
            tail_out = self.entity_updater(tail_inp, self.entities_emb[pos_tail])
            self.entities_emb[pos_head] = head_out
            self.entities_emb[pos_tail] = tail_out

        return torch.cat(pos_scores, dim=0), torch.cat(neg_scores, dim=0)

    # heads = triples[:, 0]
    # relations = triples[:, 1]
    # tails = triples[:, 2]
    # score = self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)
    # score = score.norm(p=self.norm, dim=1)
    # print('score shape:', score.size())
    # print('heads shape:', self.entities_emb(heads).size())
    # return score

    def loss(self, positive_score, negative_score):
        """graph embedding loss function"""
        target = torch.tensor([-1], dtype=torch.long, device=positive_score.device)
        loss_func = nn.MarginRankingLoss(margin=self.margin, reduction='none')
        return loss_func(positive_score, negative_score, target)

    def forward(self, pos_triples, neg_triples):
        """entry for calling model.train(), combining similarity and loss calculation"""
        # positive_score = self._algorithm(pos_triples)
        # negative_score = self._algorithm(neg_triples)
        positive_score, negative_score = self._algorithm(pos_triples, neg_triples)
        print('positive_score shape:', positive_score.size())
        print('negative_score shape:', negative_score.size())
        return self.loss(positive_score, negative_score), positive_score, negative_score

    def predict(self, triples):
        """dissimilar score calculation for triples"""
        # return self._algorithm(triples)
        scores = []
        for triple in triples:
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            score = self.entities_emb[head] + self.relations_emb(relation) - self.entities_emb[tail]
            score = score.norm(p=self.norm, dim=1)
            scores.append(score)
            head_inp = self.updater_inp(
                torch.cat((self.entities_emb[tail], self.relations_emb(relation)), dim=1))
            tail_inp = self.updater_inp(
                torch.cat((self.entities_emb[head], self.relations_emb(relation)), dim=1))
            self.entities_emb[head] = self.entity_updater(head_inp, self.entities_emb[head])
            self.entities_emb[tail] = self.entity_updater(tail_inp, self.entities_emb[tail])

        return torch.cat(scores, dim=0)
