# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import torch
import torch.nn as nn
import numpy as np
from ...model import TorchModel


@TorchModel.register("TransE-dist", "PyTorch")
class TransE(TorchModel):
    def __init__(self, **kwargs):
        super(TransE, self).__init__()
        self.num_entity = kwargs['num_entity']
        self.num_relation = kwargs['num_relation']
        self.hidden_size = kwargs['hidden_size']
        self.margin = kwargs['margin']
        self.norm = 1

        uniform_range = 6 / np.sqrt(self.hidden_size)
        self.entities_emb = nn.Embedding(self.num_entity, self.hidden_size)
        self.relations_emb = nn.Embedding(self.num_relation, self.hidden_size)
        self.entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        self.relations_emb.weight.data.uniform_(-uniform_range, uniform_range)

    def _algorithm(self, triples):
        """ graph embedding similarity algorithm method """
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]
        score = self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)
        score = score.norm(p=self.norm, dim=1)
        return score

    def loss(self, positive_score, negative_score):
        """graph embedding loss function"""
        target = torch.tensor([-1], dtype=torch.long)
        loss_func = nn.MarginRankingLoss(margin=self.margin, reduction='none')
        return loss_func(positive_score, negative_score, target)

    def forward(self, pos_triples, neg_triples):
        """entry for calling model.train(), combining similarity and loss calculation"""
        positive_score = self._algorithm(pos_triples)
        negative_score = self._algorithm(neg_triples)
        return self.loss(positive_score, negative_score), positive_score, negative_score

    def predict(self, triples):
        """dissimilar score calculation for triples"""
        return self._algorithm(triples)

