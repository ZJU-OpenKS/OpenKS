# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import torch
import torch.nn as nn
from ...model import TorchModel

logger = logging.getLogger(__name__)

@TorchModel.register("question-embedding", "PyTorch")
class QuestionEmbedding(TorchModel):
	def __init__(self, **kwargs):
		super(QuestionEmbedding, self).__init__()
		self.config = kwargs
		target_size = self.config['label']
		self.embed = nn.Embedding(self.config['words_num'], self.config['words_dim'])
		if self.config['train_embed'] == False:
			self.embed.weight.requires_grad = False
		if self.config['qa_mode'] == 'LSTM':
			self.lstm = nn.LSTM(input_size=self.config['words_num'], 
				hidden_size=self.config['hidden_size'],
				num_layers=self.config['num_layer'],
				dropout=self.config['rnn_dropout'],
				bidirectional=True)
		elif self.config['qa_mode'] == 'GRU':
			self.gru = nn.GRU(input_size=self.config['words_num'],
				hidden_size=self.config['hidden_size'],
				num_layers=self.config['num_layer'],
				dropout=self.config['rnn_dropout'],
				bidirectional=True)
		self.dropout = nn.Dropout(p=self.config['rnn_fc_dropout'])
		self.nonlinear = nn.Tanh()
		self.hidden2tag = nn.Sequential(
			nn.Linear(self.config['hidden_size'] * 2, self.config['hidden_size'] * 2),
			nn.BatchNorm1d(self.config['hidden_size'] * 2),
			self.nonlinear,
			self.dropout,
			nn.Linear(self.config['hidden_size'] * 2, target_size)
		)

	def loss(self, scores, batch):
		loss_func = nn.MSELoss()
		return loss_func(scores, batch)

	def forward(self, x):
		text = x.text
		embed = x.embed
		x = self.embed(text)
		num_word, batch_size, words_dim = x.size()
		if self.config['entity_detection_mode'] == 'LSTM':
			outputs, (ht, ct) = self.lstm(x)
		elif self.config['entity_detection_mode'] == 'GRU':
			outputs, ht = self.gru(x)
		else:
			print("Wrong Entity Prediction Mode")
			exit(1)
		outputs = outputs.view(-1, outputs.size(2))
		tags = self.hidden2tag(outputs).view(num_word, batch_size, -1)
		scores = nn.functional.normalize(torch.mean(tags, dim=0), dim=1)
		return self.loss(scores, x.embed), scores
