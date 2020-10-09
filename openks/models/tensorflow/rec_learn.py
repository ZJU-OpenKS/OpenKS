'''
reference to: https://github.com/xiangwang1223/neural_graph_collaborative_filtering
'''
import numpy as np
import scipy.sparse as sp
import random as rd
import tensorflow as tf
import multiprocessing
from ..model import RecModel

@RecModel.register("recommendation", "TensorFlow")
class RecTF(RecModel):

	def __init__(self, name='tf-default', dataset=None, model=None, args=None):
		self.name = name
		self.dataset = dataset
		self.args = args
		self.model = model
		self.train_set = self.dataset.bodies[0]
		self.valid_set = self.dataset.bodies[1]

	def data_counter(self):
		self.n_items, self.n_users = 0, 0
		self.n_train, self.n_test = 0, 0
		for l in self.train_set:
			if len(l) > 0:
				l = l.strip('\n').split(' ')
				items = [int(i) for i in l[1:]]
				uid = int(l[0])
				self.exist_users.append(uid)
				self.n_items = max(self.n_items, max(items))
				self.n_users = max(self.n_users, uid)
				self.n_train += len(items)
		for l in self.valid_set:
			if len(l) > 0:
				l = l.strip('\n')
				try:
					items = [int(i) for i in l.split(' ')[1:]]
				except Exception:
					continue
				self.n_items = max(self.n_items, max(items))
				self.n_test += len(items)

		self.n_items += 1
		self.n_users += 1
		return self.n_items, self.n_users, self.n_train, self.n_test

	def data_generator(self):
		self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

		self.train_items, self.test_set = {}, {}

		for l in self.train_set:
			if len(l) == 0: 
				break
			l = l.strip('\n')
			items = [int(i) for i in l.split(' ')]
			uid, train_items = items[0], items[1:]

			for i in train_items:
				self.R[uid, i] = 1.
				# self.R[uid][i] = 1

			self.train_items[uid] = train_items

		for l in self.valid_set:
			if len(l) == 0: break
			l = l.strip('\n')
			try:
				items = [int(i) for i in l.split(' ')]
			except Exception:
				continue

			uid, test_items = items[0], items[1:]
			self.test_set[uid] = test_items

		return self.train_items, self.test_set

	def sample(self, batch_size):
		if batch_size <= self.n_users:
			users = rd.sample(self.exist_users, batch_size)
		else:
			users = [rd.choice(self.exist_users) for _ in range(batch_size)]

		def sample_pos_items_for_u(u, num):
			pos_items = self.train_items[u]
			n_pos_items = len(pos_items)
			pos_batch = []
			while True:
				if len(pos_batch) == num: break
				pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
				pos_i_id = pos_items[pos_id]

				if pos_i_id not in pos_batch:
					pos_batch.append(pos_i_id)
			return pos_batch

		def sample_neg_items_for_u(u, num):
			neg_items = []
			while True:
				if len(neg_items) == num: break
				neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
				if neg_id not in self.train_items[u] and neg_id not in neg_items:
					neg_items.append(neg_id)
			return neg_items

		def sample_neg_items_for_u_from_pools(u, num):
			neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
			return rd.sample(neg_items, num)

		pos_items, neg_items = [], []
		for u in users:
			pos_items += sample_pos_items_for_u(u, 1)
			neg_items += sample_neg_items_for_u(u, 1)

		return users, pos_items, neg_items

	def get_adj_mat(self):
		try:
			t1 = time()
			adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
			norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
			mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
			print('already load adj matrix', adj_mat.shape, time() - t1)

		except Exception:
			adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
			sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
			sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
			sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
		return adj_mat, norm_adj_mat, mean_adj_mat

	def create_adj_mat(self):
		t1 = time()
		adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
		adj_mat = adj_mat.tolil()
		R = self.R.tolil()

		adj_mat[:self.n_users, self.n_users:] = R
		adj_mat[self.n_users:, :self.n_users] = R.T
		adj_mat = adj_mat.todok()
		print('already create adjacency matrix', adj_mat.shape, time() - t1)

		t2 = time()

		def normalized_adj_single(adj):
			rowsum = np.array(adj.sum(1))

			d_inv = np.power(rowsum, -1).flatten()
			d_inv[np.isinf(d_inv)] = 0.
			d_mat_inv = sp.diags(d_inv)

			norm_adj = d_mat_inv.dot(adj)
			# norm_adj = adj.dot(d_mat_inv)
			print('generate single-normalized adjacency matrix.')
			return norm_adj.tocoo()

		def check_adj_if_equal(adj):
			dense_A = np.array(adj.todense())
			degree = np.sum(dense_A, axis=1, keepdims=False)

			temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
			print('check normalized adjacency matrix whether equal to this laplacian matrix.')
			return temp

		norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		mean_adj_mat = normalized_adj_single(adj_mat)

		print('already normalize adjacency matrix', time() - t2)
		return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

	def save_model(self):
		return NotImplemented


	def load_model(self):
		return NotImplemented

	def evaluate(sess, model, users_to_test, batch_size, drop_flag=False, batch_test_flag=False):
		result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
				  'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

		pool = multiprocessing.Pool(cores)

		u_batch_size = batch_size * 2
		i_batch_size = batch_size

		test_users = users_to_test
		n_test_users = len(test_users)
		n_user_batchs = n_test_users // u_batch_size + 1

		count = 0

		for u_batch_id in range(n_user_batchs):
			start = u_batch_id * u_batch_size
			end = (u_batch_id + 1) * u_batch_size

			user_batch = test_users[start: end]

			if batch_test_flag:

				n_item_batchs = self.n_items // i_batch_size + 1
				rate_batch = np.zeros(shape=(len(user_batch), self.n_items))

				i_count = 0
				for i_batch_id in range(n_item_batchs):
					i_start = i_batch_id * i_batch_size
					i_end = min((i_batch_id + 1) * i_batch_size, self.n_items)

					item_batch = range(i_start, i_end)

					if drop_flag == False:
						i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
																	model.pos_items: item_batch})
					else:
						i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
																	model.pos_items: item_batch,
																	model.node_dropout: [0.]*len(eval(self.args['layer_size'])),
																	model.mess_dropout: [0.]*len(eval(self.args['layer_size']))})
					rate_batch[:, i_start: i_end] = i_rate_batch
					i_count += i_rate_batch.shape[1]

				assert i_count == self.n_items

			else:
				item_batch = range(self.n_items)

				if drop_flag == False:
					rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
																  model.pos_items: item_batch})
				else:
					rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
																  model.pos_items: item_batch,
																  model.node_dropout: [0.] * len(eval(self.args['layer_size'])),
																  model.mess_dropout: [0.] * len(eval(self.args['layer_size']))})

			user_batch_rating_uid = zip(rate_batch, user_batch)
			batch_result = pool.map(test_one_user, user_batch_rating_uid)
			count += len(batch_result)

			for re in batch_result:
				result['precision'] += re['precision']/n_test_users
				result['recall'] += re['recall']/n_test_users
				result['ndcg'] += re['ndcg']/n_test_users
				result['hit_ratio'] += re['hit_ratio']/n_test_users
				result['auc'] += re['auc']/n_test_users

		assert count == n_test_users
		pool.close()
		return result

	
	def run(self):
		n_users, n_items, n_train, n_test = self.data_counter()
		train_items, test_set = self.data_generator()
		plain_adj, norm_adj, mean_adj = self.get_adj_mat()
		model = self.model(
			lr=self.args['lr'],
			embed_size=self.args['embed_size'],
			batch_size=self.args['batch_size'],
			layer_size=self.args['layer_size'],
			regs=self.args['regs'],
			n_users=n_users,
			n_items=n_items,
			adj=plain_adj)

		for epoch in range(self.args['epoch']):
			t1 = time()
			loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
			n_batch = n_train // self.args['batch_size'] + 1
			for idx in range(n_batch):
				users, pos_items, neg_items = self.sample(self.args['batch_size'])
				_, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run(
					[model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
					feed_dict={model.users: users, 
							model.pos_items: pos_items,
							model.node_dropout: eval(self.args['node_dropout']), 
							model.mess_dropout: eval(self.args['mess_dropout']),
							model.neg_items: neg_items})
				loss += batch_loss
				mf_loss += batch_mf_loss
				emb_loss += batch_emb_loss
				reg_loss += batch_reg_loss

			if np.isnan(loss) == True:
				print('ERROR: loss is nan.')
				sys.exit()

			if (epoch + 1) % 10 != 0:
				perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, reg_loss)
				print(perf_str)
				continue

			t2 = time()
			users_to_test = list(test_set.keys())
			ret = self.evaluate(sess, model, users_to_test, self.args['batch_size'], drop_flag=True)

			t3 = time()

			perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
					'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
					(epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
					ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
					ret['ndcg'][0], ret['ndcg'][-1])
			print(perf_str)
