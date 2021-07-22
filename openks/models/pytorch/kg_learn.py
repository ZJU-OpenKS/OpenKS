# Copyright (c) 2021 OpenKS Authors, DCD Research Lab and Dlib Lab, Zhejiang University and Peking University. 
# All Rights Reserved.

import logging
import argparse
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
import numpy as np
from sklearn.model_selection import train_test_split
from ..model import KGLearnModel, TorchDataset
from .kg_modules import NCESoftmaxLossNS

from .dataloader import TrainDataset, TestDataset
from .dataloader import BidirectionalOneShotIterator
import json

import pdb

logger = logging.getLogger(__name__)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


class DataSet(TorchDataset):
	def __init__(self, triples):
		self.triples = triples

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, index):
		head, relation, tail = self.triples[index]
		return head, relation, tail

@KGLearnModel.register("KGLearn_Dy", "PyTorch")
class KGLearn_DyTorch(KGLearnModel):
    def __init__(self, name='pytorch-default', graph=None, model=None, args=None):
        self.name = name
        self.graph = graph
        self.args = args
        self.model = model

    def dy_train_test_split(self, triples, test_size):
        triples_filted = []
        for t in triples:
            if len(t[1]) > 1:
                triples_filted.append(t)
        triples_filted.sort(key=lambda x: x[1][1])
        n_train = int(len(triples) * (1-test_size))
        train_triples = triples[:n_train]
        test_triples = triples[n_train:]
        return train_triples, test_triples

    def triples_reader(self, ratio=0.01):
        """read from triple data files to id triples"""
        rel2id = self.graph.relation_to_id()
        train_valid_triples, test_triples = self.dy_train_test_split(self.graph.triples, test_size=ratio)
        train_triples, valid_triples = self.dy_train_test_split(train_valid_triples, test_size=ratio)
        train_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in train_triples]
        valid_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in valid_triples]
        test_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in test_triples]
        return train_triples, valid_triples, test_triples

    def triples_generator(self, train_triples, device):
        heads, relations, tails = train_triples
        heads_pos, relations_pos, tails_pos = (heads.to(device), relations.to(device), tails.to(device))
        positive_triples = torch.stack((heads_pos, relations_pos, tails_pos), dim=1)
        head_or_tail = torch.randint(high=2, size=heads.size(), device=device)
        random_entities = torch.randint(high=self.graph.get_entity_num(), size=heads.size(), device=device)
        # random_entities = torch.randint(high=len(train_triples), size=heads.size(), device=device)
        broken_heads = torch.where(head_or_tail==1, random_entities, heads_pos)
        broken_tails = torch.where(head_or_tail==0, random_entities, tails_pos)
        negative_triples = torch.stack((broken_heads, relations_pos, broken_tails), dim=1)
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

            # pdb.set_trace()

            # predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
            predictions = torch.cat((heads_predictions, tails_predictions), dim=0)
            ground_truth_entity_id = torch.cat((head.reshape(-1, 1), tail.reshape(-1, 1)))

            hits_at_1 += self.hit_at_k(predictions, ground_truth_entity_id, device=device, k=1)
            hits_at_3 += self.hit_at_k(predictions, ground_truth_entity_id, device=device, k=3)
            hits_at_10 += self.hit_at_k(predictions, ground_truth_entity_id, device=device, k=10)
            mrr_value += self.mrr(predictions, ground_truth_entity_id)
            examples_count += predictions.size()[0]

            count += 1
            if count % 500 == 0:
                print("=================")
                print(hits_at_1, hits_at_3, hits_at_10)
                print("=================")

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

    def run(self, dist=False):
        device = torch.device('cuda') if self.args['gpu'] else torch.device('cpu')

        train_triples, valid_triples, test_triples = self.triples_reader(ratio=0.01)
        # set PyTorch sample iterators
        train_set = DataSet(train_triples)
        train_generator = data.DataLoader(train_set, batch_size=self.args['batch_size'], shuffle=False)
        valid_set = DataSet(valid_triples)
        valid_generator = data.DataLoader(valid_set, batch_size=1, shuffle=False)
        test_set = DataSet(test_triples)
        test_generator = data.DataLoader(test_set, batch_size=1, shuffle=False)

        train_part_triples = train_triples[:1000]
        train_part_generator = data.DataLoader(train_part_triples, batch_size=1, shuffle=False)

        # initialize model
        # model = self.model(
        # 	num_entity=self.graph.get_entity_num(),
        # 	num_relation=self.graph.get_relation_num(),
        # 	hidden_size=self.args['hidden_size'],
        # 	margin=self.args['margin']
        # )
        model = self.model(
            num_entity=self.graph.get_entity_num(),
            num_relation=self.graph.get_relation_num(),
            # hidden_size=self.args['hidden_size'],
            # margin=self.args['margin']
            **self.args
        )
        model = model.to(device)

        # initialize optimizer
        optimizer_available = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
        }
        opt = optimizer_available[self.args['optimizer']](model.parameters(), lr=self.args['learning_rate'])

        start_epoch = 1
        best_score = 0.0

        # train iteratively
        for epoch in range(start_epoch, self.args['epoch'] + 1):
            print("Starting epoch: ", epoch)
            run_loss = 0
            model.train()
            # train in a batch
            for train_triples in train_generator:
                positive_triples, negative_triples = self.triples_generator(train_triples, device)
                opt.zero_grad()
                loss, pos_score, neg_score = model(positive_triples, negative_triples)
                run_loss += loss.mean().item()
                loss.mean().backward()
                opt.step()
                model.entities_emb.detach_()
            print("Loss: " + str(run_loss))

            # evaluation periodically
            if epoch % self.args['eval_freq'] == 0:
                print("Starting validation...")
                model.eval()
                print("train set...")
                _, _, hits_at_10, _ = self.evaluate(model=model, data_generator=train_part_generator, num_entity=self.graph.get_entity_num(), device=device)
                print("valid set...")
                _, _, hits_at_10, _ = self.evaluate(model=model, data_generator=valid_generator, num_entity=self.graph.get_entity_num(), device=device)
                score = hits_at_10
                print("HIT@10: " + str(score))
                if score > best_score:
                    best_score = score
                    self.save_model(model, opt, epoch, best_score, self.args['model_dir'])

        # load saved model and test
        self.load_model(self.args['model_dir'], model, opt)
        best_model = model.to(device)
        best_model.eval()
        scores = self.evaluate(model=best_model, data_generator=test_generator, num_entity=self.graph.get_entity_num(), device=device)
        print("Test scores: ", scores)


@KGLearnModel.register("KGLearn_GCN", "PyTorch")
class KGLearn_GCNTorch(KGLearnModel):
	def __init__(self, name='pytorch-default', graph=None, model=None, args=None):
		self.name = name
		self.graph = graph
		self.args = args
		self.model = model
		self.train_loader = args['train_loader']

	def triples_reader(self, ratio=0.01):
		"""read from triple data files to id triples"""
		pass

	def triples_generator(self, train_triples, device):
		pass

	def hit_at_k(self, predictions, ground_truth_idx, device, k=10):
		"""how many true entities in top k similar ones"""
		pass


	def mrr(self, predictions, ground_truth_idx):
		"""Mean Reciprocal Rank"""
		pass

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

	def run(self, dist=False):
		device = torch.device('cuda') if self.args['gpu'] else torch.device('cpu')

		model = self.model(num_layers=self.args['num_layer'], gnn_model='gat')
		model = model.to(device)

		# initialize optimizer
		optimizer_available = {
			"adam": optim.Adam,
			"sgd": optim.SGD,
		}
		opt = optimizer_available[self.args['optimizer']](model.parameters(), lr=self.args['learning_rate'])

		start_epoch = 1
		best_score = 0.0
		criterion = NCESoftmaxLossNS()
		criterion = criterion.to(device)
		# train iteratively
		for epoch in range(start_epoch, self.args['epoch'] + 1):
			print("Starting epoch: ", epoch)
			run_loss = 0
			model.train()
			# train in a batch
			for idx, batch in enumerate(self.train_loader):
				graph_q, graph_k = batch
				graph_q.to(device)
				graph_k.to(device)
				bsz = graph_q.batch_size
				feat_q = model(graph_q)
				feat_k = model(graph_k)

				out = torch.matmul(feat_k, feat_q.t()) / self.args["nce_t"]
				prob = out[range(graph_q.batch_size), range(graph_q.batch_size)].mean()
				opt.zero_grad()

				loss = criterion(out)
				loss.backward()
				run_loss += loss.mean().item()
				opt.step()

			print("Loss: " + str(run_loss))

