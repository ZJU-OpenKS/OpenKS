import logging
import argparse
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.optim import optimizer
import numpy as np
from ..model import KGModelBase, TorchDataset

logger = logging.getLogger(__name__)


class DataSet(TorchDataset):
	def __init__(self, triples):
		self.triples = triples

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, index):
		head, relation, tail = self.triples[index]
		return head, relation, tail


@KGModelBase.register("KGLearn", "PyTorch")
class KGModel(KGModelBase):
	def __init__(self, name='pytorch-default', graph=None, model=None, args=None):
		self.name = name
		self.graph = graph
		self.args = self.parse_args(args)
		self.model = model

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
		parser.add_argument('--model_path', default='./model.tar', type=str)
		parser.add_argument('--opt', default='sgd', type=str)
		return parser.parse_args(args)

	def triples_reader(self, file_path, entity2id, relation2id):
		"""read from triple data files to id triples"""
		triples = []
		with open(file_path) as fin:
			for line in fin:
				h, r, t = line.strip().split('\t')
				triples.append((entity2id[h], relation2id[r], entity2id[t]))
		return triples

	def triples_generator(self, train_triples, device):
		heads, relations, tails = train_triples
		heads_pos, relations_pos, tails_pos = (heads.to(device), relations.to(device), tails.to(device))
		positive_triples = torch.stack((heads_pos, relations_pos, tails_pos), dim=1)
		head_or_tail = torch.randint(high=2, size=heads.size(), device=device)
		random_entities = torch.randint(high=len(train_triples), size=heads.size(), device=device)
		broken_heads = torch.where(head_or_tail==1, random_entities, heads)
		broken_tails = torch.where(head_or_tail==0, random_entities, tails)
		negative_triples = torch.stack((broken_heads, relations, broken_tails), dim=1)
		return positive_triples, negative_triples

	def hit_at_k(self, predictions, ground_truth_idx, device, k=10):
		"""how many true entities in top k similar ones"""
		assert predictions.size(0) == ground_truth_idx.size(0)
		zero_tensor = torch.tensor([0], device=device)
		one_tensor = torch.tensor([1], device=device)
		_, indices = predictions.topk(k=k, largest=False)
		return torch.where(indices == ground_truth_idx, one_tensor, zero_tensor).sum().item()


	def mrr(self, predictions, ground_truth_idx):
		"""Mean Reciprocal Rank"""
		assert predictions.size(0) == ground_truth_idx.size(0)
		indices = predictions.argsort()
		return (1.0 / (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0)).sum().item()

	def evaluate(self, model, data_generator, num_entity, device):
		"""predicting validation and test set and show performance metrics"""
		hits_at_1 = 0.0
		hits_at_3 = 0.0
		hits_at_10 = 0.0
		mrr_value = 0.0
		examples_count = 0.0
		entity_ids = torch.arange(end=num_entity, device=device).unsqueeze(0)
		count = 0
		for head, relation, tail in data_generator:
			current_batch_size = head.size()[0]
			head, relation, tail = head.to(device), relation.to(device), tail.to(device)
			all_entities = entity_ids.repeat(current_batch_size, 1)
			heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
			relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
			tails = tail.reshape(-1, 1).repeat(1, all_entities.size()[1])

			triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
			tails_predictions = model.predict(triplets).reshape(current_batch_size, -1)
			triplets = torch.stack((all_entities, relations, tails), dim=2).reshape(-1, 3)
			heads_predictions = model.predict(triplets).reshape(current_batch_size, -1)

			predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
			ground_truth_entity_id = torch.cat((head.reshape(-1, 1), tail.reshape(-1, 1)))

			hits_at_1 += self.hit_at_k(predictions, ground_truth_entity_id, device=device, k=1)
			hits_at_3 += self.hit_at_k(predictions, ground_truth_entity_id, device=device, k=3)
			hits_at_10 += self.hit_at_k(predictions, ground_truth_entity_id, device=device, k=10)
			mrr_value += self.mrr(predictions, ground_truth_entity_id)
			examples_count += predictions.size()[0]

			if count % 500 == 0:
				print("=================")
				print(hits_at_1, hits_at_3, hits_at_10)
				print("=================")
			count += 1

		hits_at_1_score = hits_at_1 / examples_count
		hits_at_3_score = hits_at_3 / examples_count
		hits_at_10_score = hits_at_10 / examples_count
		mrr_score = mrr_value / examples_count
		return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score

	def load_model(self, model_path, model, opt):
		"""load model from local model file"""
		checkpoint = torch.load(model_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		opt.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch'] + 1
		best_score = checkpoint['best_score']
		return start_epoch, best_score

	def save_model(self, model, optim, epoch, best_score, model_path):
		"""save model to local file"""
		torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optim.state_dict(),
			'epoch': epoch,
			'best_score': best_score
		}, model_path)

	def run(self):
		device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
		# read triples
		with open(os.path.join(self.args.data_path, 'entities.dict')) as f:
			entity2id = dict()
			for line in f:
				eid, entity = line.strip().split('\t')
				entity2id[entity] = int(eid)

		with open(os.path.join(self.args.data_path, 'relations.dict')) as f:
			relation2id = dict()
			for line in f:
				rid, relation = line.strip().split('\t')
				relation2id[relation] = int(rid)

		train_triples = self.triples_reader(os.path.join(self.args.data_path, 'train.txt'), entity2id, relation2id)
		valid_triples = self.triples_reader(os.path.join(self.args.data_path, 'valid.txt'), entity2id, relation2id)
		test_triples = self.triples_reader(os.path.join(self.args.data_path, 'test.txt'), entity2id, relation2id)

		# set PyTorch sample iterators
		train_set = DataSet(train_triples)
		train_generator = data.DataLoader(train_set, batch_size=self.args.batch_size)
		valid_set = DataSet(valid_triples)
		valid_generator = data.DataLoader(valid_set, batch_size=1)
		test_set = DataSet(test_triples)
		test_generator = data.DataLoader(test_set, batch_size=1)

		# initialize model
		model = self.model(
			num_entity=len(entity2id),
			num_relation=len(relation2id),
			hidden_dim=self.args.hidden_dim,
			margin=self.args.margin
		)
		model = model.to(device)

		# initialize optimizer
		optimizer_available = {
			"adam": optim.Adam,
			"sgd": optim.SGD,
		}
		opt = optimizer_available[self.args.opt](model.parameters(), lr=self.args.lr)

		start_epoch = 1
		best_score = 0.0

		# train iteratively
		for epoch in range(start_epoch, self.args.epochs + 1):
			print("Starting epoch: ", epoch)
			model.train()
			# train in a batch
			for train_triples in train_generator:
				positive_triples, negative_triples = self.triples_generator(train_triples, device)
				opt.zero_grad()
				loss, pos_score, neg_score = model(positive_triples, negative_triples)
				loss.mean().backward()
				opt.step()

			# evaluation periodically
			if epoch % self.args.valid_freq == 0:
				print("Starting validation...")
				model.eval()
				_, _, hits_at_10, _ = self.evaluate(model=model, data_generator=valid_generator, num_entity=len(entity2id), device=device)
				score = hits_at_10
				print("HIT@10: " + str(score))
				if score > best_score:
					best_score = score
					self.save_model(model, opt, epoch, best_score, self.args.model_path)

		# load saved model and test
		self.load_model(self.args.model_path, model, opt)
		best_model = model.to(device)
		best_model.eval()
		scores = self.evaluate(model=best_model, data_generator=test_generator, num_entity=len(entity2id), device=device)
		print("Test scores: ", scores)

