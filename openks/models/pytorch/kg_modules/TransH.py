# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...model import TorchModel


@TorchModel.register("TransH", "PyTorch")
class TransH(TorchModel):
	def __init__(self, **kwargs):
		super(TransH, self).__init__()
		self.num_entity = kwargs['num_entity']
		self.num_relation = kwargs['num_relation']
		self.hidden_size = kwargs['hidden_size']
		self.margin = kwargs['margin']
		# self.norm = 1
		self.epsilon = kwargs['epsilon']
		self.gamma = kwargs['gamma']

		self.uniform_range = (self.epsilon + self.gamma) / self.hidden_size

		self.entity_dim = self.hidden_size * 2 if kwargs['double_entity_embedding'] else self.hidden_size
		self.relation_dim = self.hidden_size * 2 if kwargs['double_relation_embedding'] else self.hidden_size

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

		self.norm_vector = nn.Parameter(torch.zeros(self.num_relation, self.relation_dim))
		nn.init.uniform_(
			tensor=self.norm_vector,
			a=-self.uniform_range,
			b=self.uniform_range
		)
		'''
		uniform_range = 6 / np.sqrt(self.hidden_size)
		self.entities_emb = nn.Embedding(self.num_entity, self.hidden_size)
		self.relations_emb = nn.Embedding(self.num_relation, self.hidden_size)
		self.norm_vector = nn.Embedding(self.num_relation, self.hidden_size)
		self.entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
		self.relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
		self.norm_vector.weight.data.uniform_(-uniform_range, uniform_range)
		'''

	def forward(self, head, relation, tail, r_norm, mode):
		h = self._transfer(head, r_norm)
		t = self._transfer(tail, r_norm)
		if mode == 'head-batch':
			score = h + (relation - t)
		else:
			score = (h + relation) - t

		score = self.gamma - torch.norm(score, p=1, dim=2)
		return score

	def _transfer(self, e, norm):
		norm = F.normalize(norm, p=2, dim=-1)
		assert e.shape[0] == norm.shape[0]
		'''
		if e.shape[0] != norm.shape[0]:
			e = e.view(-1, norm.shape[0], e.shape[-1])
			norm = norm.view(-1, norm.shape[0], norm.shape[-1])
			e = e - torch.sum(e * norm, -1, True) * norm
			return e.view(-1, e.shape[-1])
		else:
		'''
		return e - torch.sum(e * norm, -1, True) * norm

	'''
	def _algorithm(self, triples):
		""" graph embedding similarity algorithm method """
		heads = triples[:, 0]
		relations = triples[:, 1]
		tails = triples[:, 2]
		r_norm = self.norm_vector(relations)
		h = self._transfer(self.entities_emb(heads), r_norm)
		t = self._transfer(self.entities_emb(tails), r_norm)
		score = h + self.relations_emb(relations) - t
		score = score.norm(p=self.norm, dim=1)
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
		return self.loss(positive_score, negative_score), positive_score, negative_score

	def predict(self, triples):
		"""dissimilar score calculation for triples"""
		return self._algorithm(triples)
	'''
