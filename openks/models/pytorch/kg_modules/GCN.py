# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import torch
from torch import nn
from torch.nn import functional as F
from ...model import TorchModel


class GCNLayer(nn.Module):
	def __init__(self, input_dim, output_dim, num_features_nonzero,
				 dropout=0.,
				 bias=False,
				 activation = F.relu,
				 featureless=False):
		super(GCNLayer, self).__init__()


		self.dropout = dropout
		self.bias = bias
		self.activation = activation
		self.featureless = featureless
		self.num_features_nonzero = num_features_nonzero

		self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
		self.bias = None
		if bias:
			self.bias = nn.Parameter(torch.zeros(output_dim))


	def forward(self, inputs):
		x, support = inputs
		x = F.dropout(x, self.dropout)

		# convolve
		if not self.featureless:
			xw = torch.mm(x, self.weight)
		else:
			xw = self.weight
		out = torch.sparse.mm(support, xw)
		if self.bias is not None:
			out += self.bias
		return self.activation(out), support


@TorchModel.register("GCN", "PyTorch")
class GCN(TorchModel):
	def __init__(self, **kwargs):
		super(GCN, self).__init__()

		self.input_dim = kwargs['input_dim']
		self.output_dim = kwargs['output_dim']
		self.num_features_nonzero = kwargs['num_features_nonzero']
		self.hidden_size = kwargs['hidden_size']
		self.dropout = dropout
		print('input dim:', self.input_dim)
		print('output dim:', self.output_dim)
		print('num_features_nonzero:', self.num_features_nonzero)

		self.layers = nn.Sequential(
			GCNLayer(self.input_dim, self.hidden_size, self.num_features_nonzero,
				activation=F.relu,
				dropout=self.dropout,
				is_sparse_inputs=True),
			GCNLayer(self.hidden_size, self.output_dim, self.num_features_nonzero,
				activation=F.relu,
				dropout=self.dropout,
				is_sparse_inputs=False))

	def forward(self, inputs):
		x, support = inputs
		x = self.layers((x, support))
		return x

	def loss(self):
		layer = self.layers.children()
		layer = next(iter(layer))
		loss = None
		for p in layer.parameters():
			if loss is None:
				loss = p.pow(2).sum()
			else:
				loss += p.pow(2).sum()
		return loss
