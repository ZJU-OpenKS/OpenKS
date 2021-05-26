# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ..model import KGLearnModel, TorchDataset
import json
import pickle
import pdb

logger = logging.getLogger(__name__)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def read_embdding(file_path):
    '''
    Read embeddings
    '''
    with open(file_path) as fin:
       embeddings = pickle.load(fin)
    return embeddings

def read_NodeClassifier_data(file_path):
    '''
    Read data
    '''
    data = []
    with open(file_path) as fin:
        for line in fin:
            node_id, label = line.strip().split('\t')
            data.append((node_id, label))
    return data

def read_NodeMatching_data(file_path):
    '''
    Read data
    '''
    data = []
    with open(file_path) as fin:
        for line in fin:
            node1_id, node2_id, label = line.strip().split('\t')
            data.append((node1_id, node2_id, label))
    return data

def read_LinkPreiction_data(file_path):
    '''
    Read data
    '''
    data = []
    with open(file_path) as fin:
        for line in fin:
            node1_id, node2_id, label = line.strip().split('\t')
            data.append((node1_id, node2_id, label))
    return data

def read_RelationPrediction_data(file_path):
    '''
    Read data
    '''
    data = []
    with open(file_path) as fin:
        for line in fin:
            head_id, tail_id, label = line.strip().split('\t')
            data.append((head_id, tail_id, label))
    return data

class DataSet(TorchDataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index]

@KGLearnModel.register("TaskLearn", "PyTorch")
class TaskLearn_DyTorch(KGLearnModel):
	def __init__(self, name='pytorch-default', model=None, node_embedding=None, relation_embedding=None, args=None):
		self.name = name
		self.args = args
		self.model = model
		self.node_embedding = node_embedding
		self.relation_embedding = relation_embedding

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
		self.node_embedding = np.load(os.path.join(self.args['save_path'], 'node_embedding'))
		self.relation_embedding = np.load(os.path.join(self.args['save_path'], 'relation_embedding'))
		logging.info('load best-valid-score model at step %d: %f' % (init_step-1, best_score))
		return init_step, current_learning_rate, warm_up_steps, best_score

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

		node_embedding = self.node_embedding.detach().cpu().numpy()
		np.save(
			os.path.join(self.args['save_path'], 'node_embedding'),
			node_embedding
		)

		relation_embedding = self.relation_embedding.detach().cpu().numpy()
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

		torch.manual_seed(self.args['random_seed'])
		if self.args['model_name'] == 'NodeClassifier':
			model = self.model(self.embedding, self.args['class_num'])
			train_data = read_NodeClassifier_data(os.path.join(self.args['data_dir'], 'train.txt'))
			valid_data = read_NodeClassifier_data(os.path.join(self.args['data_dir'], 'valid.txt'))
			test_data = read_NodeClassifier_data(os.path.join(self.args['data_dir'], 'test.txt'))
		elif self.args['model_name'] == 'NodeMatching':
			model = self.model(self.embedding)
			train_data = read_NodeMatching_data(os.path.join(self.args['data_dir'], 'train.txt'))
			valid_data = read_NodeMatching_data(os.path.join(self.args['data_dir'], 'valid.txt'))
			test_data = read_NodeMatching_data(os.path.join(self.args['data_dir'], 'test.txt'))
		elif self.args['model_name'] == 'LinkPrediction':
			model = self.model(self.embedding, self.relation_embedding, self.args['entity_num'])
			train_data = read_LinkPreiction_data(os.path.join(self.args['data_dir'], 'train.txt'))
			valid_data = read_LinkPreiction_data(os.path.join(self.args['data_dir'], 'valid.txt'))
			test_data = read_LinkPreiction_data(os.path.join(self.args['data_dir'], 'test.txt'))
		elif self.args['model_name'] == 'RelationPrediction':
			model = self.model(self.embedding, self.relation_embedding)
			train_data = read_RelationPrediction_data(os.path.join(self.args['data_dir'], 'train.txt'))
			valid_data = read_RelationPrediction_data(os.path.join(self.args['data_dir'], 'valid.txt'))
			test_data = read_RelationPrediction_data(os.path.join(self.args['data_dir'], 'test.txt'))

		logging.info('Model Parameter Configuration:')
		for name, param in model.named_parameters():
			logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

		model = model.to(device)

		train_dataloader = DataLoader(
			DataSet(train_data),
			batch_size=self.args['batch_size'],
			shuffle=True,
			num_workers=max(1, self.args['cpu_num'] // 2),
		)

		valid_dataloader = DataLoader(
			DataSet(valid_data),
			batch_size=self.args['batch_size'],
			shuffle=True,
			num_workers=max(1, self.args['cpu_num'] // 2),
		)

		test_dataloader = DataLoader(
			DataSet(test_data),
			batch_size=self.args['batch_size'],
			shuffle=False,
			num_workers=max(1, self.args['cpu_num'] // 2),
		)

		current_learning_rate = self.args['learning_rate']

		# initialize optimizer
		optimizer_available = {
			"adam": optim.Adam,
			"sgd": optim.SGD,
		}
		opt = optimizer_available[self.args['optimizer']](model.parameters(), lr=current_learning_rate)

		if self.args['warm_up_steps'] is None:
			warm_up_steps = self.args['max_steps'] // 2
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
		for step in range(init_step, self.args['max_steps'] + 1):
			log = self.train_step(model, opt, train_dataloader, device)
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
				metrics = self.eval_step(model, valid_dataloader, self.args)
				log_metrics('Valid', step, metrics)
				score = metrics['acc']
				if score > best_score:
					best_score = score
					save_variable_list = {
						'step': step,
						'current_learning_rate': current_learning_rate,
						'warm_up_steps': warm_up_steps,
						'best_score': best_score
					}
					self.save_model(model, opt, save_variable_list)

		# load saved model and test
		self.load_model(model, opt)
		model = model.to(device)

		if self.args['do_valid']:
			logging.info('Evaluating on Valid Dataset...')
			metrics = self.eval_step(model, valid_dataloader, self.args)
			log_metrics('Valid', step, metrics)

		if self.args['do_test']:
			logging.info('Evaluating on Test Dataset...')
			metrics = self.eval_step(model, test_dataloader, self.args)
			log_metrics('Test', step, metrics)

		if self.args['evaluate_train']:
			logging.info('Evaluating on Training Dataset...')
			metrics = self.eval_step(model, train_dataloader, self.args)
			log_metrics('Test', step, metrics)

	def train_step(self, model, optimizer, dataloader, device):
		'''
        A single train step. Apply back-propation and return the loss
        '''

		model.train()
		optimizer.zero_grad()

		for step, batch_data in enumerate(dataloader):
			if self.args['model_name'] == 'NodeClassifier':
				node_id = batch_data[0].to(device)
				label = batch_data[1].to(device)
				out = self.model(node_id)
				criterion = nn.CrossEntropyLoss().to(device)
				loss = criterion(out, label)
			elif self.args['model_name'] == 'NodeMatching':
				node1_id = batch_data[0].to(device)
				node2_id = batch_data[1].to(device)
				label = batch_data[2].to(device)
				out = nn.Sigmoid(self.model(node1_id, node2_id))
				criterion = nn.BCELoss().to(device)
				loss = criterion(out, label)
			elif self.args['model_name'] == 'LinkPrediction':
				head_id = batch_data[0].to(device)
				relation_id = batch_data[1].to(device)
				tail_id = batch_data[2].to(device)
				out = self.model(head_id, relation_id, tail_id)
				criterion = nn.BCELoss().to(device)
				loss = criterion(out, label)
			elif self.args['model_name'] == 'RelationPrediction':
				head_id = batch_data[0].to(device)
				tail_id = batch_data[1].to(device)
				out = self.model(head_id, tail_id)
				criterion = nn.BCELoss().to(device)
				loss = criterion(out, label)
		# if args['regularization'] != 0.0:
		# 	# Use L2 regularization
		# 	regularization = args.regularization * (
		# 			model.node_embedding.norm() + model.relation_embedding.norm()
		# 	)
		# 	loss = loss + regularization
		# 	regularization_log = {'regularization': regularization.item()}
		# else:
		# 	regularization_log = {}

		loss.backward()
		optimizer.step()

		log = {'loss': loss.item()
		}
		return log

	def eval_step(self, model, eval_dataloader, args):
		'''
        Evaluate the model on test or valid datasets
        '''

		model.eval()

		logs = []

		total = len(eval_dataloader)
		succeed = 0
		with torch.no_grad():
			for step, eval_data in enumerate(eval_dataloader):
				if self.args['model_name'] == 'NodeClassifier':
					node_id = eval_data[0].to(device)
					label = eval_data[1].to(device)
					out = self.model(node_id)
					succeed += (out == label).squeeze()
				elif self.args['model_name'] == 'NodeMatching':
					node1_id = eval_data[0].to(device)
					node2_id = eval_data[1].to(device)
					label = eval_data[2].to(device)
					out = self.model(node1_id, node2_id)
					succeed += (out == label).squeeze()
				elif self.args['model_name'] == 'LinkPrediction':
					head_id = eval_data[0].to(device)
					relation_id = eval_data[1].to(device)
					tail_id = eval_data[2].to(device)
					out = self.model(head_id, relation_id, tail_id)

				elif self.args['model_name'] == 'RelationPrediction':
					head_id = eval_data[0].to(device)
					tail_id = eval_data[1].to(device)
					out = self.model(head_id, tail_id)
					
		acc = succeed/total
		logs.append({
			'acc': acc
					})
		# 'HITS@1': 1.0 if ranking <= 1 else 0.0,
		# 'HITS@3': 1.0 if ranking <= 3 else 0.0,
		# 'HITS@10': 1.0 if ranking <= 10 else 0.0,
		return logs

