# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import torch
import torch.nn as nn
import numpy as np
from ...model import TorchModel


@TorchModel.register("RotatE", "PyTorch")
class RotatE(TorchModel):
	def __init__(self, **kwargs):
		super(RotatE, self).__init__()
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

		'''
		self.gamma = nn.Parameter(
			torch.Tensor([kwargs['gamma']]), 
			requires_grad=False
		)
		
		self.embedding_range = nn.Parameter(
			torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_size]), 
			requires_grad=False
		)
		
		uniform_range = 6 / np.sqrt(self.hidden_size)
		self.entities_emb = nn.Embedding(self.num_entity, self.hidden_size*2)
		self.relations_emb = nn.Embedding(self.num_relation, self.hidden_size)
		self.entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
		self.relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
		'''

	def forward(self, head, relation, tail, mode):
		pi = 3.14159265358979323846

		re_head, im_head = torch.chunk(head, 2, dim=2)
		re_tail, im_tail = torch.chunk(tail, 2, dim=2)

		# Make phases of relations uniformly distributed in [-pi, pi]

		phase_relation = relation / (self.uniform_range / pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)

		if mode == 'head-batch':
			re_score = re_relation * re_tail + im_relation * im_tail
			im_score = re_relation * im_tail - im_relation * re_tail
			re_score = re_score - re_head
			im_score = im_score - im_head
		else:
			re_score = re_head * re_relation - im_head * im_relation
			im_score = re_head * im_relation + im_head * re_relation
			re_score = re_score - re_tail
			im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim=0)
		score = score.norm(dim=0)

		score = self.gamma - score.sum(dim=2)
		return score

	'''
	def _RotatE(self, head, relation, tail):
		pi = 3.14159265358979323846

		head = head.unsqueeze(1)
		relation = relation.unsqueeze(1)
		tail = tail.unsqueeze(1)

		re_head, im_head = torch.chunk(head, 2, dim=2)
		re_tail, im_tail = torch.chunk(tail, 2, dim=2)

		#Make phases of relations uniformly distributed in [-pi, pi]

		phase_relation = relation/(self.embedding_range.item()/pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)

		re_score = re_head * re_relation - im_head * im_relation
		im_score = re_head * im_relation + im_head * re_relation
		re_score = re_score - re_tail
		im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(p=self.norm, dim = 0)

		score = self.gamma.item() - score.sum(dim = 2)
		return score

	def _algorithm(self, triples):
		""" graph embedding similarity algorithm method """
		heads = triples[:, 0]
		relations = triples[:, 1]
		tails = triples[:, 2]
		score = self._RotatE(self.entities_emb(heads) , self.relations_emb(relations), self.entities_emb(tails))
		score = score.squeeze(1)
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