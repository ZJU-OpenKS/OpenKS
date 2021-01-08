# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layer_helper import LayerHelper
from ...model import PaddleModel

logger = logging.getLogger(__name__)

@PaddleModel.register("GCN", "Paddle")
class GCN(PaddleModel):
	def __init__(self, **kwargs):
		# parameters setup
		self.hidden_dim = kwargs['hidden_dim']
		self.learning_rate = kwargs['lr']
		self.act_func = kwargs['act_func']
		self.graph = kwargs['graph']
		self.forward()

	@staticmethod
	def _algorithm(graph, feature, hidden_dim, act_func, name):
		# gcn layer implement
		def send_src_copy(src_feat, dst_feat, edge_feat):
			return src_feat["h"]
		size = len(feature)
		if size > self.hidden_dim:
			feature = fluid.layers.fc(feature, size=hidden_dim, bias_attr=False, param_attr=fluid.ParamAttr(name=name))
		msg = graph.send(send_src_copy, nfeat_list=[("h", feature)])
		if size > self.hidden_dim:
			output = graph.recv(msg, "sum")
		else:
			output = graph.recv(msg, "sum")
			output = fluid.layers.fc(output, size=hidden_size, bias_attr=False, param_attr=fluid.ParamAttr(name=name))
		bias = fluid.layers.create_parameter(shape=[hidden_size], dtype='float32', is_bias=True, name=name + '_bias')
		output = fluid.layers.elementwise_add(output, bias, act=act_func)
		return output


	def forward(self):
		self.startup_program = fluid.Program()
		self.train_program = fluid.Program()
		with fluid.program_guard(self.train_program, self.startup_program):
			# forward structure
			self.train_fetch_vars = self.train_forward()
			loss = self.train_fetch_vars[0]
			self.backward(loss, opt=self.opt)

		self.test_program = train_program.clone(for_test=True)

	def train_forward(self):
		# network structure
		output = self._algorithm(self.graph, self.graph.triples, self.hidden_dim, self.act_func, "gcn_layer_1")
		output = fluid.layers.dropout(output, 0.5, dropout_implementation='upscale_in_train')
		output = self._algorithm(self.graph, output, len(self.graph.num_class()), None, "gcn_layer_2")
		node_index = fluid.layers.data("node_index", shape=[None, 1], dtype="int64", append_batch_size=False)
		node_label = fluid.layers.data("node_label", shape=[None, 1], dtype="int64", append_batch_size=False)
		pred = fluid.layers.gather(output, node_index)
		loss, pred = fluid.layers.softmax_with_cross_entropy(logits=pred, label=node_label, return_softmax=True)
		acc = fluid.layers.accuracy(input=pred, label=node_label, k=1)
		loss = fluid.layers.mean(loss)
		return [loss]

	def backward(self, loss, opt):
		optimizer_available = {
			"adam": fluid.optimizer.Adam,
			"sgd": fluid.optimizer.SGD,
			"momentum": fluid.optimizer.Momentum
		}
		if opt in optimizer_available:
			opt_func = optimizer_available[opt]
		else:
			opt_func = None
		if opt_func is None:
			raise ValueError("You should chose the optimizer in %s" % optimizer_available.keys())
		else:
			optimizer = opt_func(learning_rate=self.learning_rate, regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0005))
			return optimizer.minimize(loss)