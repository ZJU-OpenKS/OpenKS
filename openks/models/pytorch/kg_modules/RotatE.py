# Copyright (c) 2021 OpenKS Authors, Dlib Lab, Peking University. 
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
