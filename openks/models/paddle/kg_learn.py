# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import argparse
import os
import paddle.fluid as fluid
import numpy as np
from sklearn.model_selection import train_test_split
from ..model import KGLearnModel
from ...distributed.openks_distributed import KSDistributedFactory
from ...distributed.openks_distributed.base import RoleMaker
from ...distributed.openks_strategy.cpu import CPUStrategy, SyncModeConfig

logger = logging.getLogger(__name__)


@KGLearnModel.register("KGLearn", "Paddle")
class KGLearnPaddle(KGLearnModel):
	def __init__(self, name='paddle-default', graph=None, model=None, args=None):
		self.name = name
		self.graph = graph
		self.args = args
		self.model = model

	def triples_reader(self, ratio=0.01):
		"""read from triple data files to id triples"""
		rel2id = self.graph.relation_to_id()
		train_triples, test_triples = train_test_split(self.graph.triples, test_size=ratio)
		train_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in train_triples]
		test_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in test_triples]
		return np.array(train_triples), np.array(test_triples), np.array(test_triples)

	def triples_generator(self, train_triples, batch_size):
		train_triple_positive_batches = []
		n = len(train_triples)
		rand_idx = np.random.permutation(n)
		n_triple = len(rand_idx)
		start = 0
		while start < n_triple:
			end = min(start + batch_size, n_triple)
			train_triple_positive = train_triples[rand_idx[start:end]]
			start = end
			train_triple_positive_batches.append(train_triple_positive)

		def triple_constructor(train_triple_positive):
			""" training triples generator """
			size = len(train_triple_positive)  # neg_times = 1
			train_triple_negative = train_triple_positive.repeat(1, axis=0)
			replace_head_probability = 0.5 * np.ones(size)
			replace_entity_id = np.random.randint(size, size=size)
			random_num = np.random.random(size=size)
			index_t = (random_num < replace_head_probability) * 1
			train_triple_negative[:, 0] = train_triple_negative[:, 0] + (replace_entity_id - train_triple_negative[:, 0]) * index_t
			train_triple_negative[:, 2] = replace_entity_id + (train_triple_negative[:, 2] - replace_entity_id) * index_t
			train_triple_positive = np.expand_dims(train_triple_positive, axis=2)
			train_triple_negative = np.expand_dims(train_triple_negative, axis=2)
			return train_triple_positive, train_triple_negative

		def triple_loader():
			for batch_data in train_triple_positive_batches:
				yield triple_constructor(batch_data)
		
		return triple_loader

	def evaluate(self, exe, program, test_triples, test_feed_list, fetch_list):
		all_rank = []
		count = 0
		for triple in test_triples:
			data = np.array(triple)
			data = data.reshape((-1))
			feed_dict = {}
			for k, v in zip(test_feed_list, [data]):
				feed_dict[k] = v
			tail_score, head_score = exe.run(program=program, fetch_list=fetch_list, feed=feed_dict)

			head, relation, tail = feed_dict["test_triple"][0], feed_dict["test_triple"][1], feed_dict["test_triple"][2]
			head_order = np.argsort(head_score)
			tail_order = np.argsort(tail_score)
			head_rank_raw = 1
			tail_rank_raw = 1
			for candidate in head_order:
				if candidate == head:
					break
				else:
					head_rank_raw += 1
			for candidate in tail_order:
				if candidate == tail:
					break
				else:
					tail_rank_raw += 1
			all_rank.extend([head_rank_raw, tail_rank_raw])
			if count % 500 == 0:
				print("=================")
				print((np.array(all_rank) <= 1).sum(), (np.array(all_rank) <= 3).sum(), (np.array(all_rank) <= 10).sum())
				print("=================")
			count += 1
		raw_rank = np.array(all_rank)
		return (raw_rank <= 1).mean(), (raw_rank <= 3).mean(), (raw_rank <= 10).mean(), (1 / raw_rank).mean()

	def run(self, dist=False):
		program = None
		dist_algorithm = None

		train_triples, valid_triples, test_triples = self.triples_reader(ratio=0.01)

		device = fluid.cuda_places() if self.args['gpu'] else fluid.cpu_places()

		if dist:
			dist_algorithm = KSDistributedFactory.instantiation(flag=0)
			role = RoleMaker.PaddleCloudRoleMaker()
			dist_algorithm.init(role)

		model = self.model(
			num_entity=self.graph.get_entity_num(),
			num_relation=self.graph.get_relation_num(),
			hidden_size=self.args['hidden_size'],
			margin=self.args['margin'],
			lr=self.args['learning_rate'],
			opt=self.args['optimizer'],
			dist=dist_algorithm)

		if dist:
			if dist_algorithm.is_server():
				dist_algorithm.init_server()
				dist_algorithm.run_server()
			elif dist_algorithm.is_worker():
				dist_algorithm.init_worker()
				program = dist_algorithm.main_program
		else:
			program = fluid.CompiledProgram(model.train_program).with_data_parallel(loss_name=model.train_fetch_vars[0].name)

		train_loader = fluid.io.DataLoader.from_generator(feed_list=model.train_feed_vars, capacity=20, iterable=True)
		train_loader.set_batch_generator(self.triples_generator(train_triples, batch_size=self.args['batch_size']), places=device)

		exe = fluid.Executor(device[0])
		exe.run(model.startup_program)
		exe.run(fluid.default_startup_program())

		best_score = 0.0
		for epoch in range(1, self.args['epoch'] + 1):
			print("Starting epoch: ", epoch)
			loss = 0
			# train in a batch
			for batch_feed_dict in train_loader():
				batch_fetch = exe.run(program, fetch_list=model.train_fetch_vars, feed=batch_feed_dict)
				loss += batch_fetch[0]
			print("Loss: " + str(loss))

			# evaluation periodically
			if epoch % self.args['eval_freq'] == 0:
				print("Starting validation...")
				_, _, hits_at_10, _ = self.evaluate(exe, model.test_program, valid_triples, model.test_feed_list, model.test_fetch_vars)
				score = hits_at_10
				print("HIT@10: " + str(score))
				if score > best_score:
					best_score = score
					fluid.io.save_params(exe, dirname=self.args['model_dir'], main_program=model.train_program)
		if dist:
			dist_algorithm.stop_worker()

		# load saved model and test
		fluid.io.load_params(exe, dirname=self.args['model_dir'], main_program=model.train_program)
		scores = self.evaluate(exe, program, test_triples, model.test_feed_list, model.test_fetch_vars)
		print("Test scores: ", scores)
