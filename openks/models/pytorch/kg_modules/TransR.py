# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...model import TorchModel


@TorchModel.register("TransR", "PyTorch")
class TransR(TorchModel):
	def __init__(self, **kwargs):
		super(TransR, self).__init__()
		self.num_entity = kwargs['num_entity']
		self.num_relation = kwargs['num_relation']
		# self.entity_size = kwargs['hidden_size']
		# self.relation_size = kwargs['hidden_size']
		self.hidden_size = kwargs['hidden_size']
		self.margin = kwargs['margin']
		self.norm = 1
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

		# self.entities_emb = nn.Embedding(self.num_entity, self.entity_size)
		# self.relations_emb = nn.Embedding(self.num_relation, self.relation_size)
		# self.transfer_matrix = nn.Embedding(self.num_relation, self.entity_size * self.relation_size)
		self.transfer_matrix = nn.Parameter(torch.zeros(self.num_relation, self.entity_dim * self.relation_dim))
		# nn.init.xavier_uniform_(self.entities_emb.weight.data)
		# nn.init.xavier_uniform_(self.relations_emb.weight.data)
		
		identity = torch.zeros(self.entity_dim, self.relation_dim)
		for i in range(min(self.entity_dim, self.relation_dim)):
			identity[i][i] = 1
		identity = identity.view(self.relation_dim * self.entity_dim)
		for i in range(self.num_relation):
			self.transfer_matrix.data[i] = identity

	def forward(self, head, relation, tail, r_transfer, mode):
		r_transfer = r_transfer.view(-1, self.entity_dim, self.relation_dim)
		h = self._transfer(head, r_transfer)
		t = self._transfer(tail, r_transfer)
		if mode == 'head-batch':
			score = h + (relation - t)
		else:
			score = (h + relation) - t

		score = self.gamma - torch.norm(score, p=1, dim=2)
		return score

	'''
	def _algorithm(self, triples):
		""" graph embedding similarity algorithm method """
		heads = triples[:, 0]
		relations = triples[:, 1]
		tails = triples[:, 2]
		r_transfer = self.transfer_matrix(relations)
		h = self._transfer(self.entities_emb(heads), r_transfer)
		t = self._transfer(self.entities_emb(tails), r_transfer)
		score = h + self.relations_emb(relations) - t
		score = score.norm(p=self.norm, dim=1)
		return score
	'''

	def _transfer(self, e, r_transfer):
		assert e.shape[0] == r_transfer.shape[0] and e.ndim == 3
		return torch.matmul(e, r_transfer)
		'''
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], self.entity_size).permute(1, 0, 2)
			e = torch.matmul(e, r_transfer).permute(1, 0, 2)
		else:
			e = e.view(-1, 1, self.entity_size)
			e = torch.matmul(e, r_transfer)
		return e.view(-1, self.relation_size)
		'''

	'''
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
