# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import json
import argparse
import torch
import torch.nn as nn
import torch.utils.data.dataset as Dataset
from sklearn.model_selection import train_test_split
from ..model import KELearnModel
#from .ke_modules.nero_modules import read, nero_run

@KELearnModel.register("KELearn", "PyTorch")
class KELearnTorch(KELearnModel):
	def __init__(self, name='pytorch-default', dataset=None, model=None, args=None):
		self.name = name
		self.dataset = dataset
		self.args = args
		self.model = model

	def parse_args(self, args):
		""" parameter settings """
		parser = argparse.ArgumentParser(
			description='Training and Testing Knowledge Graph Embedding Models',
			usage='train.py [<args>] [-h | --help]'
		)
		parser.add_argument('--gpu', action='store_true')
		parser.add_argument('--batch_size', default=1024, type=int)
		parser.add_argument('--hidden_dim', default=50, type=int)
		parser.add_argument('--lr', default=0.01, type=float)
		parser.add_argument('--epochs', default=100, type=int)
		parser.add_argument('--valid_freq', default=10, type=int)
		parser.add_argument('--model_path', default='./model.tar', type=str)
		parser.add_argument('--opt', default='sgd', type=str)
		parser.add_argument('--words_dim', type=int, default=300)
		parser.add_argument('--num_layer', type=int, default=2)
		parser.add_argument('--dropout', type=float, default=0.3)
		parser.add_argument('--weight_decay',type=float, default=0)
		parser.add_argument('--valid_freq', default=10, type=int)
		return parser.parse_args(args)

	def data_reader(self, ratio=0.01):
		train_set, test_set = train_test_split(self.dataset, test_size=ratio)
		return train_set, test_set, test_set

	def get_span():
		return NotImplemented

	def evaluation(self, gold, pred, index2tag, type):
		right = 0
		predicted = 0
		total_en = 0
		for i in range(len(gold)):
			gold_batch = gold[i]
			pred_batch = pred[i]

			for j in range(len(gold_batch)):
				gold_label = gold_batch[j]
				pred_label = pred_batch[j]
				gold_span = get_span(gold_label, index2tag, type)
				pred_span = get_span(pred_label, index2tag, type)
				total_en += len(gold_span)
				predicted += len(pred_span)
				for item in pred_span:
					if item in gold_span:
						right += 1
		if predicted == 0:
			precision = 0
		else:
			precision = right / predicted
		if total_en == 0:
			recall = 0
		else:
			recall = right / total_en
		if precision + recall == 0:
			f1 = 0
		else:
			f1 = 2 * precision * recall / (precision + recall)
		return precision, recall, f1

	def save_model(self, model, model_path):
		torch.save(model, model_path)

	def run_nero_model(self):
		# read data
		# train
		unlabeled_data, test_data, pattern = self.dataset.bodies[0], self.dataset.bodies[1], self.dataset.bodies[2]
		
		patterns = json.loads(pattern[0])
		print(self.args)
		self.args.patterns = patterns
		data = read(self.args, unlabeled_data, test_data)
		word2idx_dict, word_emb, train_data, dev_data, test_data = data
		
		model = self.model(self.args, word_emb, word2idx_dict)
		nero_run(self.args, data, model)
	
	def run(self, run_type=None):
		if run_type == 'run_nero_model':
			self.run_nero_model()
		else:
			self.default_run()

	def default_run(self):
		
		device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

		train_set, valid_set, test_set = self.triples_reader(ratio=0.01)
		train_set = DataSet(train_set)
		train_generator = data.DataLoader(train_set, batch_size=self.args.batch_size)
		valid_set = DataSet(valid_set)
		valid_generator = data.DataLoader(valid_set, batch_size=1)
		test_set = DataSet(test_set)
		test_generator = data.DataLoader(test_set, batch_size=1)

		model = self.model(
			words_dim=self.args.words_dim,
			num_layer=self.args.num_layer,
			dropout=self.args.dropout,
			hidden_dim=self.args.hidden_dim
		)
		model = model.to(device)

		optimizer = torch.optim.Adam(parameter, lr=self.args.lr, weight_decay=self.args.weight_decay)

		index2tag = None
		start_epoch = 1
		best_score = 0.0

		# train iteratively
		for epoch in range(start_epoch, self.args.epochs + 1):
			print("Starting epoch: ", epoch)
			model.train()

			for batch_idx, batch in enumerate(train_set):
				optimizer.zero_grad()
				loss, scores = model(batch)
				loss.backward()
				optimizer.step()

			if epoch % self.args.valid_freq == 0:
				print("Starting validation...")
				model.eval()
				gold_list = []
				pred_list = []

				for dev_batch_idx, dev_batch in enumerate(valid_set):
					answer = model(dev_batch)
					index_tag = np.transpose(torch.max(answer, 1)[1].view(dev_batch.ed.size()).cpu().data.numpy())
					gold_list.append(np.transpose(dev_batch.ed.cpu().data.numpy()))
					pred_list.append(index_tag)

				P, R, F = evaluation(gold_list, pred_list, index2tag, type=False)
				print("{} Recall: {:10.6f}% Precision: {:10.6f}% F1 Score: {:10.6f}%".format("Dev", 100. * R, 100. * P, 100. * F))

				if R > best_score:
					best_dev_R = R
					self.save_model(model, self.args.model_path)
