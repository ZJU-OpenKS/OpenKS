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

@KGLearnModel.register("KGLearn", "PyTorch")
class KGLearnTorch(KGLearnModel):
	def __init__(self, name='pytorch-default', graph=None, model=None, args=None):
		self.name = name
		self.graph = graph
		self.args = args
		self.model = model

	def triples_reader(self, ratio=0.05):
		"""read from triple data files to id triples"""
		rel2id = self.graph.relation_to_id()
		train_valid_triples, test_triples = train_test_split(self.graph.triples, test_size=ratio, random_state=self.args['random_seed'])
		train_triples, valid_triples = train_test_split(train_valid_triples, test_size=ratio, random_state=self.args['random_seed'])
		train_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in train_triples]
		valid_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in valid_triples]
		test_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in test_triples]
		return train_triples, valid_triples, test_triples

	def triples_reader_v2(self):
		"""read from triple data files to id triples"""
		rel2id = self.graph.relation_to_id()
		train_triples = self.graph.triples
		train_triples = [(triple[0][0], rel2id[triple[0][1]], triple[0][2]) for triple in train_triples]

		with open(os.path.join(self.args['data_dir'], 'entities')) as fin:
			entity2id = dict()
			for line in fin:
				eid, _, entity = line.strip().split('\t')
				entity2id[entity] = int(eid)

		valid_triples = read_triple(os.path.join(self.args['data_dir'], 'valid.txt'), entity2id, rel2id)
		logging.info('#valid: %d' % len(valid_triples))
		test_triples = read_triple(os.path.join(self.args['data_dir'], 'test.txt'), entity2id, rel2id)
		logging.info('#test: %d' % len(test_triples))

		return train_triples, valid_triples, test_triples

	def load_model(self, model, opt):
		"""load model from local model file"""
		# checkpoint = torch.load(model_path)
		checkpoint = torch.load(os.path.join(self.args['save_path'], 'checkpoint'))
		model.load_state_dict(checkpoint['model_state_dict'])
		opt.load_state_dict(checkpoint['optimizer_state_dict'])
		init_step = checkpoint['step'] + 1
		current_learning_rate = checkpoint['current_learning_rate']
		warm_up_steps = checkpoint['warm_up_steps']
		best_score = checkpoint['best_score']
		logging.info('load best-valid-score model at step %d: %f' % (init_step-1, best_score))
		return init_step, current_learning_rate, warm_up_steps, best_score

	def model_to_onnx(self, model, mode='single'):
		input_head = torch.index_select(model.entity_embedding, 0, torch.tensor([0]))
		input_rel = torch.index_select(model.relation_embedding, 0, torch.tensor([0]))
		input_tail = torch.index_select(model.entity_embedding, 0, torch.tensor([1]))
		input_head = input_head[None, :]
		input_rel = input_rel[None, :]
		input_tail = input_tail[None, :]
		input = (input_head, input_rel, input_tail, mode)
		input_names = ['head_input', 'relation_input', 'tail_input']
		output_names = ['score']
		torch.onnx.export(model, input, self.args['market_path'], verbose=True, input_names=input_names, output_names=output_names)

	def save_model(self, model, optimizer, save_variable_list):
		'''
	    Save the parameters of the model and the optimizer,
	    as well as some other variables such as step and learning_rate
	    '''
		with open(os.path.join(self.args['save_path'], 'config.json'), 'w') as fjson:
			json.dump(self.args, fjson)

		torch.save({
			**save_variable_list,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()},
			os.path.join(self.args['save_path'], 'checkpoint')
		)

		entity_embedding = model.entity_embedding.detach().cpu().numpy()
		np.save(
			os.path.join(self.args['save_path'], 'entity_embedding'),
			entity_embedding
		)

		relation_embedding = model.relation_embedding.detach().cpu().numpy()
		np.save(
			os.path.join(self.args['save_path'], 'relation_embedding'),
			relation_embedding
		)

	def set_logger(self):
		'''
        Write logs to checkpoint and console
        '''
		log_file = os.path.join(self.args['save_path'] or self.args['init_checkpoint'], 'train.log')

		logging.basicConfig(
			format='%(asctime)s %(levelname)-8s %(message)s',
			level=logging.INFO,
			datefmt='%Y-%m-%d %H:%M:%S',
			filename=log_file,
			filemode='w'
		)
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
		console.setFormatter(formatter)
		logging.getLogger('').addHandler(console)

	def run(self, dist=False):
		self.set_logger()
		device = torch.device('cuda') if self.args['gpu'] else torch.device('cpu')

		if self.args['random_split']:
			train_triples, valid_triples, test_triples = self.triples_reader(ratio=self.args['split_ratio'])
		else:
			train_triples, valid_triples, test_triples = self.triples_reader_v2()
		all_true_triples = train_triples + valid_triples + test_triples

		nentity = self.graph.get_entity_num()
		nrelation = self.graph.get_relation_num()

		self.args['nentity'] = nentity
		self.args['nrelation'] = nrelation

		torch.manual_seed(self.args['random_seed'])
		model = self.model(
			num_entity=nentity,
			num_relation=nrelation,
			**self.args
		)

		logging.info('Model Parameter Configuration:')
		for name, param in model.named_parameters():
			logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

		model = model.to(device)

		train_dataloader_head = data.DataLoader(
			TrainDataset(train_triples, nentity, nrelation, self.args['negative_sample_size'], 'head-batch'),
			batch_size=self.args['batch_size'],
			shuffle=True,
			num_workers=max(1, self.args['cpu_num'] // 2),
			collate_fn=TrainDataset.collate_fn
		)

		train_dataloader_tail = data.DataLoader(
			TrainDataset(train_triples, nentity, nrelation, self.args['negative_sample_size'], 'tail-batch'),
			batch_size=self.args['batch_size'],
			shuffle=True,
			num_workers=max(1, self.args['cpu_num'] // 2),
			collate_fn=TrainDataset.collate_fn
		)

		train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

		
		current_learning_rate = self.args['learning_rate']

		# initialize optimizer
		optimizer_available = {
			"adam": optim.Adam,
			"sgd": optim.SGD,
		}
		opt = optimizer_available[self.args['optimizer']](model.parameters(), lr=current_learning_rate)

		if self.args['warm_up_steps'] is None:
			warm_up_steps = self.args['epoch'] // 2
		else:
			warm_up_steps = self.args['warm_up_steps']

		# start_epoch = 1
		init_step = 1
		step = init_step
		best_score = 0.0

		# train iteratively
		# for epoch in range(start_epoch, self.args['epoch'] + 1):
		logging.info('learning_rate = %d' % current_learning_rate)
		training_logs = []
		for step in range(init_step, self.args['epoch']+1):
			log = self.train_step(model, opt, train_iterator, self.args)
			training_logs.append(log)
			if step >= warm_up_steps:
				current_learning_rate = current_learning_rate / 10
				logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
				opt = optimizer_available[self.args['optimizer']](model.parameters(), lr=current_learning_rate)
				warm_up_steps = warm_up_steps * 3

			if step % self.args['log_steps'] == 0:
				metrics = {}
				for metric in training_logs[0].keys():
					metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
				log_metrics('Training average', step, metrics)
				training_logs = []

			if step % self.args['eval_freq'] == 0:
				logging.info('Evaluating on Valid Dataset...')
				metrics = self.test_step(model, valid_triples, all_true_triples, self.args)
				log_metrics('Valid', step, metrics)
				score = metrics['HITS@10']
				if score > best_score:
					best_score = score
					save_variable_list = {
						'step': step,
						'current_learning_rate': current_learning_rate,
						'warm_up_steps': warm_up_steps,
						'best_score': best_score
					}
					# self.save_model(model, opt, save_variable_list)
					self.model_to_onnx(model)

		# load saved model and test
		# self.load_model(model, opt)
		model = model.to(device)

		if self.args['do_valid']:
			logging.info('Evaluating on Valid Dataset...')
			metrics = self.test_step(model, valid_triples, all_true_triples, self.args)
			log_metrics('Valid', step, metrics)

		if self.args['do_test']:
			logging.info('Evaluating on Test Dataset...')
			metrics = self.test_step(model, test_triples, all_true_triples, self.args)
			log_metrics('Test', step, metrics)

		if self.args['evaluate_train']:
			logging.info('Evaluating on Training Dataset...')
			metrics = self.test_step(model, train_triples, all_true_triples, self.args)
			log_metrics('Test', step, metrics)


	def train_step(self, model, optimizer, train_iterator, args):
		'''
        A single train step. Apply back-propation and return the loss
        '''

		model.train()

		optimizer.zero_grad()

		positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

		if args['gpu']:
			positive_sample = positive_sample.cuda()
			negative_sample = negative_sample.cuda()
			subsampling_weight = subsampling_weight.cuda()

		negative_score = self.forward(model, (positive_sample, negative_sample), mode=mode)

		if args['negative_adversarial_sampling']:
			# In self-adversarial sampling, we do not apply back-propagation on the sampling weight
			negative_score = (F.softmax(negative_score * args['adversarial_temperature'], dim=1).detach()
							  * F.logsigmoid(-negative_score)).sum(dim=1)
		else:
			negative_score = F.logsigmoid(-negative_score).mean(dim=1)

		positive_score = self.forward(model, positive_sample)

		positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

		if args['uni_weight']:
			positive_sample_loss = - positive_score.mean()
			negative_sample_loss = - negative_score.mean()
		else:
			positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
			negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

		loss = (positive_sample_loss + negative_sample_loss) / 2

		if args['regularization'] != 0.0:
			# Use L3 regularization for ComplEx and DistMult
			regularization = args.regularization * (
					model.entity_embedding.norm(p=3) ** 3 +
					model.relation_embedding.norm(p=3).norm(p=3) ** 3
			)
			loss = loss + regularization
			regularization_log = {'regularization': regularization.item()}
		else:
			regularization_log = {}

		loss.backward()

		optimizer.step()

		log = {
			**regularization_log,
			'positive_sample_loss': positive_sample_loss.item(),
			'negative_sample_loss': negative_sample_loss.item(),
			'loss': loss.item()
		}

		return log

	def forward(self, model, sample, mode='single'):
		'''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

		if mode == 'single':
			batch_size, negative_sample_size = sample.size(0), 1

			head = torch.index_select(
				model.entity_embedding,
				dim=0,
				index=sample[:, 0]
			).unsqueeze(1)

			relation = torch.index_select(
				model.relation_embedding,
				dim=0,
				index=sample[:, 1]
			).unsqueeze(1)

			if self.args['model_name'] == 'TransH':
				norm_r = torch.index_select(
					model.norm_vector,
					dim=0,
					index=sample[:, 1]
				).unsqueeze(1)
			elif self.args['model_name'] == 'TransR':
				r_transfer = torch.index_select(
					model.transfer_matrix,
					dim=0,
					index=sample[:, 1]
				).unsqueeze(1)

			tail = torch.index_select(
				model.entity_embedding,
				dim=0,
				index=sample[:, 2]
			).unsqueeze(1)

		elif mode == 'head-batch':
			tail_part, head_part = sample
			batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

			head = torch.index_select(
				model.entity_embedding,
				dim=0,
				index=head_part.view(-1)
			).view(batch_size, negative_sample_size, -1)

			relation = torch.index_select(
				model.relation_embedding,
				dim=0,
				index=tail_part[:, 1]
			).unsqueeze(1)

			if self.args['model_name'] == 'TransH':
				norm_r = torch.index_select(
					model.norm_vector,
					dim=0,
					index=tail_part[:, 1]
				).unsqueeze(1)
			elif self.args['model_name'] == 'TransR':
				r_transfer = torch.index_select(
					model.transfer_matrix,
					dim=0,
					index=tail_part[:, 1]
				).unsqueeze(1)

			tail = torch.index_select(
				model.entity_embedding,
				dim=0,
				index=tail_part[:, 2]
			).unsqueeze(1)

		elif mode == 'tail-batch':
			head_part, tail_part = sample
			batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

			head = torch.index_select(
				model.entity_embedding,
				dim=0,
				index=head_part[:, 0]
			).unsqueeze(1)

			relation = torch.index_select(
				model.relation_embedding,
				dim=0,
				index=head_part[:, 1]
			).unsqueeze(1)

			if self.args['model_name'] == 'TransH':
				norm_r = torch.index_select(
					model.norm_vector,
					dim=0,
					index=head_part[:, 1]
				).unsqueeze(1)
			elif self.args['model_name'] == 'TransR':
				r_transfer = torch.index_select(
					model.transfer_matrix,
					dim=0,
					index=head_part[:, 1]
				).unsqueeze(1)

			tail = torch.index_select(
				model.entity_embedding,
				dim=0,
				index=tail_part.view(-1)
			).view(batch_size, negative_sample_size, -1)

		else:
			raise ValueError('mode %s not supported' % mode)

		if self.args['model_name'] == 'TransH':
			score = model(head, relation, tail, norm_r, mode)
		elif self.args['model_name'] == 'TransR':
			score = model(head, relation, tail, r_transfer, mode)
		else:
			score = model(head, relation, tail, mode)

		return score

	def test_step(self, model, test_triples, all_true_triples, args):
		'''
        Evaluate the model on test or valid datasets
        '''

		model.eval()

		# Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
		# Prepare dataloader for evaluation
		test_dataloader_head = data.DataLoader(
			TestDataset(
				test_triples,
				all_true_triples,
				args['nentity'],
				args['nrelation'],
				'head-batch'
			),
			batch_size=args['test_batch_size'],
			num_workers=max(1, args['cpu_num'] // 2),
			collate_fn=TestDataset.collate_fn
		)

		test_dataloader_tail = data.DataLoader(
			TestDataset(
				test_triples,
				all_true_triples,
				args['nentity'],
				args['nrelation'],
				'tail-batch'
			),
			batch_size=args['test_batch_size'],
			num_workers=max(1, args['cpu_num'] // 2),
			collate_fn=TestDataset.collate_fn
		)

		test_dataset_list = [test_dataloader_head, test_dataloader_tail]

		logs = []

		step = 0
		total_steps = sum([len(dataset) for dataset in test_dataset_list])

		with torch.no_grad():
			for test_dataset in test_dataset_list:
				for positive_sample, negative_sample, filter_bias, mode in test_dataset:
					if args['gpu']:
						positive_sample = positive_sample.cuda()
						negative_sample = negative_sample.cuda()
						filter_bias = filter_bias.cuda()

					batch_size = positive_sample.size(0)

					score = self.forward(model, (positive_sample, negative_sample), mode)
					score += filter_bias

					# Explicitly sort all the entities to ensure that there is no test exposure bias
					argsort = torch.argsort(score, dim=1, descending=True)

					if mode == 'head-batch':
						positive_arg = positive_sample[:, 0]
					elif mode == 'tail-batch':
						positive_arg = positive_sample[:, 2]
					else:
						raise ValueError('mode %s not supported' % mode)

					for i in range(batch_size):
						# Notice that argsort is not ranking
						ranking = (argsort[i, :] == positive_arg[i]).nonzero()
						assert ranking.size(0) == 1

						# ranking + 1 is the true ranking used in evaluation metrics
						ranking = 1 + ranking.item()
						logs.append({
							'MRR': 1.0 / ranking,
							'MR': float(ranking),
							'HITS@1': 1.0 if ranking <= 1 else 0.0,
							'HITS@3': 1.0 if ranking <= 3 else 0.0,
							'HITS@10': 1.0 if ranking <= 10 else 0.0,
						})

					if step % args['test_log_steps'] == 0:
						logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

					step += 1

		metrics = {}
		for metric in logs[0].keys():
			metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

		return metrics

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

