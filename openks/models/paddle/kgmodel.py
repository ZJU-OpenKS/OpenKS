import logging
import argparse
import os
import paddle.fluid as fluid
import numpy as np
from ..model import KGModelBase

logger = logging.getLogger(__name__)

def model_selector(model_name):
	if model_name == 'TransE':
		from .TransE import TransE
		return TransE
	else:
		logger.warn("No model named " + model_name + ". Please define it first!")

@KGModelBase.register("paddle-kgmodel")
class KGModel(KGModelBase):
	def __init__(self, name='paddle-default', graph=None, model='', args=None):
		self.name = name
		self.graph = graph
		self.args = self.parse_args(args)
		self.model = model_selector(model)
	def parse_args(self, args):
		""" parameter settings """
		parser = argparse.ArgumentParser(
			description='Training and Testing Knowledge Graph Embedding Models',
			usage='train.py [<args>] [-h | --help]'
		)
		parser.add_argument('--data_path', default='openks/data/wn18rr', type=str)
		parser.add_argument('--margin', default=1.0, type=float)
		parser.add_argument('--gpu', action='store_true')
		parser.add_argument('--batch_size', default=1024, type=int)
		parser.add_argument('--hidden_dim', default=50, type=int)
		parser.add_argument('--lr', default=0.01, type=float)
		parser.add_argument('--epochs', default=100, type=int)
		parser.add_argument('--valid_freq', default=10, type=int)
		parser.add_argument('--model_path', default='./', type=str)
		parser.add_argument('--opt', default='sgd', type=str)
		return parser.parse_args(args)

	def triples_reader(self, file_path, entity2id, relation2id):
		"""read from triple data files to id triples"""
		triples = []
		with open(file_path) as fin:
			for line in fin:
				h, r, t = line.strip().split('\t')
				triples.append((entity2id[h], relation2id[r], entity2id[t]))
		return np.array(triples)

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

	def run(self):
		device = fluid.cuda_places() if self.args.gpu else fluid.cpu_places()

		with open(os.path.join(self.args.data_path, 'entities.dict'), "r") as f:
			entity2id = dict()
			for line in f.readlines():
				eid, entity = line.strip().split('\t')
				entity2id[entity] = int(eid)
		with open(os.path.join(self.args.data_path, 'relations.dict'), "r") as f:
			relation2id = dict()
			for line in f.readlines():
				rid, relation = line.strip().split('\t')
				relation2id[relation] = int(rid)

		train_triples = self.triples_reader(os.path.join(self.args.data_path, 'train.txt'), entity2id, relation2id)
		valid_triples = self.triples_reader(os.path.join(self.args.data_path, 'valid.txt'), entity2id, relation2id)
		test_triples = self.triples_reader(os.path.join(self.args.data_path, 'test.txt'), entity2id, relation2id)

		model = self.model(
			num_entity=len(entity2id),
			num_relation=len(relation2id),
			hidden_dim=self.args.hidden_dim,
			margin=self.args.margin,
			lr=self.args.lr,
			opt=self.args.opt)

		train_loader = fluid.io.DataLoader.from_generator(feed_list=model.train_feed_vars, capacity=20, iterable=True)
		train_loader.set_batch_generator(self.triples_generator(train_triples, batch_size=self.args.batch_size), places=device)

		exe = fluid.Executor(device[0])
		exe.run(model.startup_program)
		exe.run(fluid.default_startup_program())
		program = fluid.CompiledProgram(model.train_program).with_data_parallel(loss_name=model.train_fetch_vars[0].name)

		best_score = 0.0

		for epoch in range(1, self.args.epochs + 1):
			print("Starting epoch: ", epoch)
			loss = 0
			# train in a batch
			for batch_feed_dict in train_loader():
				batch_fetch = exe.run(program, fetch_list=model.train_fetch_vars, feed=batch_feed_dict)
				loss += batch_fetch[0]
			print("Loss: " + str(loss))

			# evaluation periodically
			if epoch % self.args.valid_freq == 0:
				print("Starting validation...")
				_, _, hits_at_10, _ = self.evaluate(exe, model.test_program, valid_triples, model.test_feed_list, model.test_fetch_vars)
				score = hits_at_10
				print("HIT@10: " + str(score))
				if score > best_score:
					best_score = score
					fluid.io.save_params(exe, dirname=self.args.model_path, main_program=model.train_program)

		# load saved model and test
		fluid.io.load_params(exe, dirname=self.args.model_path, main_program=model.train_program)
		scores = self.evaluate(exe, program, test_triples, model.test_feed_list, model.test_fetch_vars)
		print("Test scores: ", scores)
