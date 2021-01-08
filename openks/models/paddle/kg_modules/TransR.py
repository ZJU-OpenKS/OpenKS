# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layer_helper import LayerHelper
from ...model import PaddleModel

@PaddleModel.register("TransR", "Paddle")
class TransR(PaddleModel):
	def __init__(self, **kwargs):
		self.num_entity = kwargs['num_entity']
		self.num_relation = kwargs['num_relation']
		self.hidden_size = kwargs['hidden_size']
		self.margin = kwargs['margin']
		self.learning_rate = kwargs['lr']
		self.opt = kwargs['opt']
		self.dist = kwargs['dist']
		self._ent_shape = [self.num_entity, self.hidden_size]
		self._rel_shape = [self.num_relation, self.hidden_size]
		self.forward()

	@staticmethod
	def lookup_table(input_var, embedding_table, dtype='float32'):
		is_sparse = False
		is_distributed = False
		helper = LayerHelper('embedding', **locals())
		remote_prefetch = is_sparse and (not is_distributed)
		if remote_prefetch:
			assert is_sparse is True and is_distributed is False
		tmp = helper.create_variable_for_type_inference(dtype)
		padding_idx = -1
		helper.append_op(
			type='lookup_table',
			inputs={'Ids': input_var, 'W': embedding_table},
			outputs={'Out': tmp},
			attrs={
				'is_sparse': is_sparse,
				'is_distributed': is_distributed,
				'remote_prefetch': remote_prefetch,
				'padding_idx': padding_idx
			})
		return tmp

	def create_share_variables(self):
		""" Share variables for train and test programs. """
		entity_embedding = layers.create_parameter(
			shape=self._ent_shape, 
			dtype="float32", 
			name='ent_emb', 
			default_initializer=fluid.initializer.Xavier())
		relation_embedding = layers.create_parameter(
			shape=self._rel_shape, 
			dtype="float32", 
			name='rel_emb', 
			default_initializer=fluid.initializer.Xavier())
		init_values = np.tile(np.identity(self.hidden_size, dtype="float32").reshape(-1),(self.num_relation, 1))
		transfer_matrix = layers.create_parameter(
			shape=[self.num_relation, self.hidden_size * self.hidden_size],
            dtype="float32",
            name="transfer_matrix",
            default_initializer=fluid.initializer.NumpyArrayInitializer(init_values))
		return entity_embedding, relation_embedding, transfer_matrix

	@staticmethod
	def _algorithm(head, rel, tail):
		""" graph embedding similarity algorithm method """
		head = layers.l2_normalize(head, axis=-1)
		rel = layers.l2_normalize(rel, axis=-1)
		tail = layers.l2_normalize(tail, axis=-1)
		score = head + rel - tail
		return score

	@staticmethod
	def matmul_with_expend_dims(x, y):
		"""matmul_with_expend_dims"""
		x = layers.unsqueeze(x, axes=[1])
		res = layers.matmul(x, y)
		return layers.squeeze(res, axes=[1])

	def forward(self):
		self.startup_program = fluid.Program()
		self.train_program = fluid.Program()
		self.test_program = fluid.Program()
		with fluid.program_guard(self.train_program, self.startup_program):
			self.train_pos_input = layers.data("pos_triple", dtype="int64", shape=[None, 3, 1], append_batch_size=False)
			self.train_neg_input = layers.data("neg_triple", dtype="int64", shape=[None, 3, 1], append_batch_size=False)
			self.train_feed_list = ["pos_triple", "neg_triple"]
			self.train_feed_vars = [self.train_pos_input, self.train_neg_input]
			self.train_fetch_vars = self.train_forward()
			loss = self.train_fetch_vars[0]
			self.backward(loss, opt=self.opt, dist=self.dist)

		with fluid.program_guard(self.test_program, self.startup_program):
			self.test_input = layers.data("test_triple", dtype="int64", shape=[3], append_batch_size=False)
			self.test_feed_list = ["test_triple"]
			self.test_fetch_vars = self.test_forward()

	def train_forward(self):
		entity_embedding, relation_embedding, transfer_matrix = self.create_share_variables()
		pos_head = self.lookup_table(self.train_pos_input[:, 0], entity_embedding)
		pos_tail = self.lookup_table(self.train_pos_input[:, 2], entity_embedding)
		pos_rel = self.lookup_table(self.train_pos_input[:, 1], relation_embedding)
		neg_head = self.lookup_table(self.train_neg_input[:, 0], entity_embedding)
		neg_tail = self.lookup_table(self.train_neg_input[:, 2], entity_embedding)
		neg_rel = self.lookup_table(self.train_neg_input[:, 1], relation_embedding)

		rel_matrix = layers.reshape(
			self.lookup_table(self.train_pos_input[:, 1], transfer_matrix),
			[-1, self.hidden_size, self.hidden_size])
		pos_head_trans = self.matmul_with_expend_dims(pos_head, rel_matrix)
		pos_tail_trans = self.matmul_with_expend_dims(pos_tail, rel_matrix)

		rel_matrix_neg = layers.reshape(
			self.lookup_table(self.train_neg_input[:, 1], transfer_matrix),
			[-1, self.hidden_size, self.hidden_size])
		neg_head_trans = self.matmul_with_expend_dims(neg_head, rel_matrix_neg)
		neg_tail_trans = self.matmul_with_expend_dims(neg_tail, rel_matrix_neg)

		pos_score = self._algorithm(pos_head_trans, pos_rel, pos_tail_trans)
		neg_score = self._algorithm(neg_head_trans, neg_rel, neg_tail_trans)
		pos = layers.reduce_sum(layers.abs(pos_score), -1, keep_dim=False)
		neg = layers.reduce_sum(layers.abs(neg_score), -1, keep_dim=False)
		neg = layers.reshape(neg, shape=[-1, 1], inplace=True)
		loss = layers.reduce_mean(layers.relu(pos - neg + self.margin))
		return [loss]

	def test_forward(self):
		entity_embedding, relation_embedding, transfer_matrix = self.create_share_variables()

		rel_matrix = layers.reshape(
			self.lookup_table(self.test_input[1], transfer_matrix),
			[self.hidden_size, self.hidden_size])
		entity_embedding_trans = layers.matmul(entity_embedding, rel_matrix, False, False)
		rel_vec = self.lookup_table(self.test_input[1], relation_embedding)
		entity_embedding_trans = layers.l2_normalize(entity_embedding_trans, axis=-1)
		rel_vec = layers.l2_normalize(rel_vec, axis=-1)
		head_vec = self.lookup_table(self.test_input[0], entity_embedding_trans)
		tail_vec = self.lookup_table(self.test_input[2], entity_embedding_trans)
		id_replace_head = layers.reduce_sum(layers.abs(entity_embedding_trans + rel_vec - tail_vec), dim=1)
		id_replace_tail = layers.reduce_sum(layers.abs(entity_embedding_trans - rel_vec - head_vec), dim=1)
		return [id_replace_head, id_replace_tail]

	def backward(self, loss, opt, dist=None):
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
			optimizer = opt_func(learning_rate=self.learning_rate)
			return optimizer.minimize(loss)


