# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

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

	def get_auc(item_score, user_pos_test):
		item_score = sorted(item_score.items(), key=lambda kv: kv[1])
		item_score.reverse()
		item_sort = [x[0] for x in item_score]
		posterior = [x[1] for x in item_score]

		r = []
		for i in item_sort:
			if i in user_pos_test:
				r.append(1)
			else:
				r.append(0)
		auc = metrics.auc(ground_truth=r, prediction=posterior)
		return auc

	def ranklist_by_sorted(user_pos_test, test_items, rating, ranks):
		item_score = {}
		for i in test_items:
			item_score[i] = rating[i]

		K_max = max(ranks)
		K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

		r = []
		for i in K_max_item_score:
			if i in user_pos_test:
				r.append(1)
			else:
				r.append(0)
		auc = get_auc(item_score, user_pos_test)
		return r, auc

	def get_performance(user_pos_test, r, auc, ranks):
		precision, recall, ndcg, hit_ratio = [], [], [], []

		for K in ranks:
			precision.append(metrics.precision_at_k(r, K))
			recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
			ndcg.append(metrics.ndcg_at_k(r, K))
			hit_ratio.append(metrics.hit_at_k(r, K))

		return {'recall': np.array(recall), 'precision': np.array(precision),
				'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

	def test_one_user(x):
		# user u's ratings for user u
		rating = x[0]
		#uid
		u = x[1]
		#user u's items in the training set
		try:
			training_items = data_generator.train_items[u]
		except Exception:
			training_items = []
		#user u's items in the test set
		user_pos_test = data_generator.test_set[u]

		all_items = set(range(self.n_items))

		test_items = list(all_items - set(training_items))

		r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, self.args['ranks'])

		return get_performance(user_pos_test, r, auc, self.args['ranks'])

	def evaluate(sess, model, users_to_test, batch_size, drop_flag=False, batch_test_flag=False):
		result = {'precision': np.zeros(len(self.args['ranks'])), 'recall': np.zeros(len(self.args['ranks'])), 
					'ndcg': np.zeros(len(self.args['ranks'])), 'hit_ratio': np.zeros(len(self.args['ranks'])), 'auc': 0.}

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

	def early_stopping(self, log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
		assert expected_order in ['acc', 'dec']
		if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
			stopping_step = 0
			best_value = log_value
		else:
			stopping_step += 1

		if stopping_step >= flag_step:
			print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
			should_stop = True
		else:
			should_stop = False
		return best_value, stopping_step, should_stop

	
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

		config = tf.ConfigProto()
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())
		best_res = 0.


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
			res = self.evaluate(sess, model, users_to_test, self.args['batch_size'], drop_flag=True)

			t3 = time()

			perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
					'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
					(epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, res['recall'][0], res['recall'][-1],
					res['precision'][0], res['precision'][-1], res['hit_ratio'][0], res['hit_ratio'][-1],
					res['ndcg'][0], res['ndcg'][-1])
			print(perf_str)

			best_res, stopping_step, should_stop = self.early_stopping(
				res['recall'][0], best_res, stopping_step, expected_order='acc', flag_step=5)

			if should_stop == True:
				break

			if res['recall'][0] == best_res and args.save_flag == 1:
				save_saver.save(sess, self.args['model_dir'] + '/weights', global_step=epoch)
				print('save the weights in path: ', self.args['model_dir'])
