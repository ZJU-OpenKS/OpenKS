# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import torch
import torch.nn as nn
import numpy as np
from ...model import TorchModel


@TorchModel.register("TransE", "PyTorch")
class TransE(TorchModel):
	def __init__(self, **kwargs):
		super(TransE, self).__init__()
		self.num_entity = kwargs['num_entity']
		self.num_relation = kwargs['num_relation']
		self.hidden_size = kwargs['hidden_size']
		self.margin = kwargs['margin']
		self.norm = 1
		if 'epsilon' in kwargs:
			self.epsilon = kwargs['epsilon']
		else:
			self.epsilon = 2.0
		if 'gamma' in kwargs:
			self.gamma = kwargs['gamma']
		else:
			self.gamma = 24.0
		if 'double_entity_embedding' in kwargs and 'double_relation_embedding' in kwargs:
			self.double_entity_embedding = kwargs['double_entity_embedding']
			self.double_relation_embedding = kwargs['double_relation_embedding']
		else:
			self.double_entity_embedding = False
			self.double_relation_embedding = False

		self.uniform_range = (self.epsilon + self.gamma) / self.hidden_size

		self.entity_dim = self.hidden_size * 2 if self.double_entity_embedding else self.hidden_size
		self.relation_dim = self.hidden_size * 2 if self.double_relation_embedding else self.hidden_size

		self.entity_embedding = nn.Parameter(torch.zeros(self.num_entity, self.entity_dim))
		nn.init.uniform_(
			tensor=self.entity_embedding,
			a=-self.uniform_range,
			b=self.uniform_range
		)

		self.relation_embedding = nn.Parameter(torch.zeros(self.num_relation, self.relation_dim))
		nn.init.uniform_(
			tensor=self.relation_embedding,
			a=-self.uniform_range,
			b=self.uniform_range
		)
		'''
		self.entity_embedding = nn.Embedding(self.num_entity, self.hidden_size)
		self.relation_embedding = nn.Embedding(self.num_relation, self.hidden_size)
		self.entity_embedding.weight.data.uniform_(-uniform_range, uniform_range)
		self.relation_embedding.weight.data.uniform_(-uniform_range, uniform_range)
		'''

	def forward(self, head, relation, tail, mode):
		if mode == 'head-batch':
			score = head + (relation - tail)
		else:
			score = (head + relation) - tail

		score = self.gamma - torch.norm(score, p=1, dim=2)
		return score

	'''
	def _algorithm(self, triples):
		""" graph embedding similarity algorithm method """
		heads = triples[:, 0]
		relations = triples[:, 1]
		tails = triples[:, 2]
		score = self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)
		score = score.norm(p=self.norm, dim=1)
		# print('score shape:', score.size())
		# print('heads shape:', self.entities_emb(heads).size())
		return score

	def loss(self, positive_score, negative_score):
		"""graph embedding loss function"""
		target = torch.tensor([-1], dtype=torch.long, device=positive_score.device)
		loss_func = nn.MarginRankingLoss(margin=self.margin, reduction='none')
		return loss_func(positive_score, negative_score, target)

	def forward(self, pos_triples, neg_triples):
		"""entry for calling model.train(), combining similarity and loss calculation"""
		positive_score = self._algorithm(pos_triples)
		negative_score = self._algorithm(neg_triples)
		# print('negative_score shape:', negative_score.size())
		return self.loss(positive_score, negative_score), positive_score, negative_score

	def predict(self, triples):
		"""dissimilar score calculation for triples"""
		return self._algorithm(triples)
	'''
