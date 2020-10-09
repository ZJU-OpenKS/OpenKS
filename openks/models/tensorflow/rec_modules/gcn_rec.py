'''
reference to: https://github.com/xiangwang1223/neural_graph_collaborative_filtering
'''
import logging
import tensorflow as tf
import numpy as np
from ...model import TFModel

logger = logging.getLogger(__name__)

@TFModel.register("GCNRec", "TensorFlow")
class GCNRec(object):
	def __init__(self, **kwargs):

		self.n_fold = 100
		self.lr = kwargs['lr']
		self.emb_dim = kwargs['embed_size']
		self.batch_size = kwargs['batch_size']
		self.weight_size = eval(kwargs['layer_size'])
		self.n_layers = len(self.weight_size)
		self.regs = eval(kwargs['regs'])
		self.decay = self.regs[0]
		self.n_users = kwargs['n_users']
		self.n_items = kwargs['n_items']
		self.norm_adj = kwargs['adj']

		self.users = tf.placeholder(tf.int32, shape=(None,))
		self.pos_items = tf.placeholder(tf.int32, shape=(None,))
		self.neg_items = tf.placeholder(tf.int32, shape=(None,))
		self.node_dropout = tf.placeholder(tf.float32, shape=[None])
		self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

		self.weights = self._init_weights()
		self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()
		self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
		self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
		self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

		self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings)
		self.loss = self.mf_loss + self.emb_loss + self.reg_loss
		self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

	def _init_weights(self):
		all_weights = dict()

		initializer = tf.contrib.layers.xavier_initializer()

		# xavier initialization embeddings
		all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
		all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')

		self.weight_size_list = [self.emb_dim] + self.weight_size

		for k in range(self.n_layers):
			all_weights['W_gc_%d' %k] = tf.Variable(initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
			all_weights['b_gc_%d' %k] = tf.Variable(initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

			all_weights['W_bi_%d' % k] = tf.Variable(initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
			all_weights['b_bi_%d' % k] = tf.Variable(initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

			all_weights['W_mlp_%d' % k] = tf.Variable(initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
			all_weights['b_mlp_%d' % k] = tf.Variable(initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

		return all_weights

	def _split_A_hat(self, X):
		A_fold_hat = []
		fold_len = (self.n_users + self.n_items) // self.n_fold
		for i_fold in range(self.n_fold):
			start = i_fold * fold_len
			if i_fold == self.n_fold -1:
				end = self.n_users + self.n_items
			else:
				end = (i_fold + 1) * fold_len
			A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
		return A_fold_hat

	def _create_gcn_embed(self):
		A_fold_hat = self._split_A_hat(self.norm_adj)
		embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
		all_embeddings = [embeddings]

		for k in range(0, self.n_layers):
			temp_embed = []
			for f in range(self.n_fold):
				temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

			embeddings = tf.concat(temp_embed, 0)
			embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
			embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

			all_embeddings += [embeddings]

		all_embeddings = tf.concat(all_embeddings, 1)
		u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
		return u_g_embeddings, i_g_embeddings

	def create_bpr_loss(self, users, pos_items, neg_items):
		pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
		neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

		regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
		regularizer = regularizer/self.batch_size
		
		maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
		mf_loss = tf.negative(tf.reduce_mean(maxi))
		emb_loss = self.decay * regularizer
		reg_loss = tf.constant(0.0, tf.float32, [1])
		return mf_loss, emb_loss, reg_loss

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo().astype(np.float32)
		indices = np.mat([coo.row, coo.col]).transpose()
		return tf.SparseTensor(indices, coo.data, coo.shape)

