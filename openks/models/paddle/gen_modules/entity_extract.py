# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from ...model import PaddleModel

@PaddleModel.register("entity-extract", "Paddle")
class EntityExtract(PaddleModel):

	def __init__(self, **kwargs):
		self.word_dict_len = kwargs['word_dict_len']
		self.label_dict_len = kwargs['label_dict_len']
		self.word_dim = kwargs['word_dim']
		self.hidden_lr = kwargs['hidden_lr']
		self.hidden_size = kwargs['hidden_size']
		self.depth = kwargs['depth']
		self.mix_hidden_lr = kwargs['mix_hidden_lr']
		self.word = fluid.data(name='word_data', shape=[None, 1], dtype='int64', lod_level=1)
		self.target = fluid.layers.data(name='target', shape=[1], dtype='int64', lod_level=1)
		fluid.default_startup_program().random_seed = 90
		fluid.default_main_program().random_seed = 90
		self.loss, self.feature_out = self.forward(self.word)
	

	def forward(self, word):
		word_input = [word]
		emb_layers = [
			fluid.embedding(
				size=[self.word_dict_len, self.word_dim], 
				input=x, 
				param_attr=fluid.ParamAttr(name='emb', learning_rate=self.hidden_lr, trainable=True))
			for x in word_input
		]

		hidden_0_layers = [fluid.layers.fc(input=emb, size=self.hidden_size, act='tanh') for emb in emb_layers]

		hidden_0 = fluid.layers.sums(input=hidden_0_layers)

		lstm_0 = fluid.layers.dynamic_lstm(
			input=hidden_0,
			size=self.hidden_size,
			candidate_activation='relu',
			gate_activation='sigmoid',
			cell_activation='sigmoid')

		# stack L-LSTM and R-LSTM with direct edges
		input_tmp = [hidden_0, lstm_0]

		for i in range(1, self.depth):
			mix_hidden = fluid.layers.sums(input=[
				fluid.layers.fc(input=input_tmp[0], size=self.hidden_size, act='tanh'),
				fluid.layers.fc(input=input_tmp[1], size=self.hidden_size, act='tanh')
			])

			lstm = fluid.layers.dynamic_lstm(
				input=mix_hidden,
				size=self.hidden_size,
				candidate_activation='relu',
				gate_activation='sigmoid',
				cell_activation='sigmoid',
				is_reverse=((i % 2) == 1))

			input_tmp = [mix_hidden, lstm]

		feature_out = fluid.layers.sums(input=[
			fluid.layers.fc(input=input_tmp[0], size=self.label_dict_len, act='tanh'),
			fluid.layers.fc(input=input_tmp[1], size=self.label_dict_len, act='tanh')
		])

		
		crf_cost = fluid.layers.linear_chain_crf(
			input=feature_out,
			label=self.target,
			param_attr=fluid.ParamAttr(name='crfw', learning_rate=self.mix_hidden_lr))

		avg_cost = fluid.layers.mean(crf_cost)
		self.backward(avg_cost)
		return avg_cost, feature_out


	def backward(self, loss):
		sgd_optimizer = fluid.optimizer.SGD(
			learning_rate=fluid.layers.exponential_decay(
				learning_rate=0.01,
				decay_steps=100000,
				decay_rate=0.5,
				staircase=True))

		return sgd_optimizer.minimize(loss)
