import sys, os, io
import random
import json
import torch
import logging
import argparse
import gensim
import struct
import math
import itertools
import copy
import time
import warnings

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from torch.autograd import Variable
from sklearn.utils import shuffle
from pytorch_pretrained_bert import BertTokenizer, BertModel

from .mimo_modules.utils import *
from .mimo_modules.MIMO import *
from .mimo_modules.data_center import *

warnings.filterwarnings('ignore')



from ..model import openieModel

@openieModel.register("openie", "PyTorch")
class openie_learn(openieModel):
	def __init__(self, args, name: str = 'openieModel', ):
		super().__init__()
		self.name = name
		self.args = args

	def run(self):
		args = self.args
		if torch.cuda.is_available():
			if not args.cuda:
				print("WARNING: You have a CUDA device, so you should probably run with --cuda")
			else:
				device_id = torch.cuda.current_device()
				print('using device', device_id, torch.cuda.get_device_name(device_id))

		device = torch.device("cuda" if args.cuda else "cpu")
		print('DEVICE:', device)


		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)

		logging.debug(args)

		models = []
		multi_head = None
		multi_head_two = None
		max_f1 = 0 # max macro-f1 of validation
		max_std = 0 # max std of macro-f1 of validation
		batch_size = 35
		dim = 50 # dimension of WE
		input_size = dim # input size of encoder
		hidden_dim = 300 # the number of LSTM units in encoder layer
		bert_hidden_dim = 768
		dataCenter = DataCenter(args.train, args.eval)
		
		_weight_classes_fact = []
		for _id in range(len(dataCenter.ID2Tag_fact)):
			if args.is_semi:
				_weight_classes_fact.append(1.0)
			else:
				_weight_classes_fact.append((1.0/dataCenter.Tag2Num[dataCenter.ID2Tag_fact[_id]])*1000)
		weight_classes_fact = torch.FloatTensor(_weight_classes_fact)
		print(weight_classes_fact)
		weight_classes_fact = weight_classes_fact.to(device)

		_weight_classes_condition = []
		for _id in range(len(dataCenter.ID2Tag_condition)):
			if args.is_semi:
				_weight_classes_condition.append(1.0)
			else:
				_weight_classes_condition.append((1.0/dataCenter.Tag2Num[dataCenter.ID2Tag_condition[_id]])*1000)
		weight_classes_condition = torch.FloatTensor(_weight_classes_condition)
		print(weight_classes_condition)
		weight_classes_condition = weight_classes_condition.to(device)

		out_model_name = args.out_model+'_'+args.model_name
		out_file = args.out_file+'_'+args.model_name

		LM_model = None
		LM_corpus = None
		tokenizer = None

		if args.model_name in ['MIMO_LSTM', 'MIMO_LSTM_TF', 'MIMO_BERT_LSTM', 'MIMO_BERT_LSTM_TF']:
			config_list = [bool(int(i)) for i in args.config]
			assert len(config_list) == 9
			lm_config = config_list[:3]
			postag_config = config_list[3:6]
			cap_config = config_list[6:9]

			print('lm config', lm_config)
			print('postag config', postag_config)
			print('cap config', cap_config)

			# decodeing config for multi_heads
			configs = []
			if True in lm_config:
				configs.append([lm_config, [False]*3, [False]*3])
			if True in postag_config:
				configs.append([[False]*3, postag_config, [False]*3])
			if True in cap_config:
				configs.append([[False]*3, [False]*3, cap_config])
			print(configs)
			if len(configs) > 1:
				print('There are more than one featrues, thus multi_heads will be used.')
			if len(configs) == 0:
				print('The model without any input-features is to be trained.')
				configs.append([[False]*3, [False]*3, [False]*3])

			if not args.model_name.startswith('MIMO_BERT'):
				wv = Gensim(args.wordembed, dim)
				word2vec = wv.word2vec_dict
				PAD = '<pad>'
				WordEmbedding = [word2vec[PAD].view(1, -1),]
				Word2ID = dict()
				ID2Word = dict()
				Word2ID[PAD] = 0
				ID2Word[0] = PAD
				for word in word2vec:
					if word == PAD or word in Word2ID:
						continue
					_id = len(WordEmbedding)
					Word2ID[word] = _id
					ID2Word[_id] = word
					WordEmbedding.append(word2vec[word].view(1, -1))
				WordEmbedding = torch.cat(WordEmbedding)

			for config in configs:
				print('creating model by', config)
				lm_config, postag_config, cap_config = config

				if args.model_name == 'MIMO_LSTM':
					mimo = MIMO_LSTM(WordEmbedding, Word2ID, dataCenter.POS2ID, dataCenter.CAP2ID, dim, input_size, hidden_dim, len(dataCenter.Tag2ID_fact), len(dataCenter.Tag2ID_condition), lm_config, postag_config, cap_config, device)
				elif args.model_name == 'MIMO_LSTM_TF':
					mimo = MIMO_LSTM_TF(WordEmbedding, Word2ID, dataCenter.POS2ID, dataCenter.CAP2ID, dim, input_size, hidden_dim, len(dataCenter.Tag2ID_fact), len(dataCenter.Tag2ID_condition), lm_config, postag_config, cap_config, device)
				elif args.model_name == 'MIMO_BERT_LSTM':
					tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
					assert '[UNK]' in tokenizer.vocab
					mimo = MIMO_BERT_LSTM(dataCenter.POS2ID, dataCenter.CAP2ID, bert_hidden_dim, len(dataCenter.Tag2ID_fact), len(dataCenter.Tag2ID_condition), lm_config, postag_config, cap_config, device)
				else:
					tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
					assert '[UNK]' in tokenizer.vocab
					mimo = MIMO_BERT_LSTM_TF(dataCenter.POS2ID, dataCenter.CAP2ID, bert_hidden_dim, len(dataCenter.Tag2ID_fact), len(dataCenter.Tag2ID_condition), lm_config, postag_config, cap_config, device)

				config_str = ''.join([str(int(i)) for i in np.reshape(config, 9)])
				if (not args.pretrain) or len(configs) > 1 or args.is_semi:
					print('loading pretrained model ...')
					name = 'models/pre_supervised_model_'+args.model_name+'_'+config_str+'.torch'
					if not args.pretrain and len(configs) > 1:
						name = 'models/supervised_model_'+args.model_name+'_'+config_str
					print(name)
					try:
						mimo = torch.load(name)
					except:
						print('please train the single figure model first:', config_str)
						sys.exit(1)

					if len(configs) > 1:
						for param in mimo.parameters():
							param.requires_grad = False
				
				if args.run_eval:
					if args.pretrain:
						name = 'models/pre_supervised_model_'+args.model_name+'_'+config_str+'.torch'
						print(name)
					else:
						name = 'models/supervised_model_'+args.model_name+'_'+config_str
						print(name)
					mimo = torch.load(name)

				mimo.to(device)
				models.append(mimo)
			
			assert len(models) == len(configs)

			if len(configs) > 1:
				_hidden_dim = bert_hidden_dim if 'BERT' in args.model_name else hidden_dim*2

				multi_head = Multi_head_Net(_hidden_dim, len(dataCenter.Tag2ID_fact))
				multi_head.to(device)

				if args.run_eval:
					print('loading multi_head model ...')
					if args.pretrain:
						multi_head = torch.load('models/pre_supervised_model_'+args.model_name+'_'+args.config+'.torch_multi_head')
					else:
						multi_head = torch.load('models/supervised_model_'+args.model_name+'_'+args.config+'_multi_head')

			with open(args.language_model, 'rb') as f:
				LM_model = torch.load(f)
				LM_model.eval()
			LM_corpus = Corpus()

			out_model_name += ('_'+args.config)
			out_file += ('_'+args.config)

		elif args.model_name == 'MIMO_BERT':
			tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			assert '[UNK]' in tokenizer.vocab
			mimo = MIMO_BERT('bert-base-uncased')
			mimo.to(device)

		mimo_extractor_fact = None
		mimo_extractor_cond = None
		if not args.pretrain:
			_hidden_dim = bert_hidden_dim if 'BERT' in args.model_name else hidden_dim*2
			mimo_extractor_fact = Extractor(_hidden_dim, len(dataCenter.Tag2ID_fact), 'fact')
			mimo_extractor_cond = Extractor(_hidden_dim, len(dataCenter.Tag2ID_condition), 'cond')
			if args.run_eval:
				print('loading mo_extractor ...')
				mimo_extractor_fact = torch.load('models/supervised_model_'+args.model_name+'_'+args.config+'_extractor_fact')
				mimo_extractor_cond = torch.load('models/supervised_model_'+args.model_name+'_'+args.config+'_extractor_cond')

			mimo_extractor_fact.to(device)
			mimo_extractor_cond.to(device)
		
		if args.is_semi:
			out_model_name += '_SeT'
			out_file += '_SeT'
			data_file = './auto_ldata/labeled'

			if args.AR:
				out_model_name += '_AR'
				out_file += '_AR'
				data_file += '_AR'
			if args.TC:
				out_model_name += '_TC'
				out_file += '_TC'
				data_file += '_TC'
			if args.TCDEL:
				out_model_name += '_TCDEL'
				out_file += '_TCDEL'
				data_file += '_TCDEL'
			if args.SH:
				out_model_name += '_SH'
				out_file += '_SH'
				data_file += '_SH'
			if args.DEL:
				out_model_name += '_DEL'
				out_file += '_DEL'
				data_file += '_DEL'

			udata_file = args.udata+'_part-1.tsv'
			data_file += '_'+udata_file.split('/')[-1]
			nu_datasets = args.nu_datasets
		else:
			nu_datasets = 1

		out_file += '.txt'
		if args.pretrain:
			out_model_name = out_model_name.replace('supervised', 'pre_supervised') + '.torch'
			out_file = out_file.replace('evaluation', 'pre_evaluation')
			
		print('out_model_name =', out_model_name)
		print('out_file =', out_file)

		for index in range(nu_datasets):
			if args.is_semi:
				udata_file = udata_file.replace('part'+str(index-1), 'part'+str(index))
				data_file = data_file.replace('part'+str(index-1), 'part'+str(index))

				print('udata_file =', udata_file)
				print('data_file =', data_file)

				dataCenter.loading_dataset(None, None, udata_file)
				auto_labeling(models, dataCenter, device, data_file, args.AR, args.TC, args.TCDEL, args.SH, args.DEL, LM_model=LM_model, LM_corpus=LM_corpus, tokenizer=tokenizer, mimo_extractor_fact=mimo_extractor_fact, mimo_extractor_cond=mimo_extractor_cond, pretrain=args.pretrain, multi_head=multi_head, multi_head_two=multi_head_two)
				dataCenter.loading_dataset(None, None, data_file)

			for epoch in range(args.epochs):
				if not args.run_eval:
					print('[epoch-%d] training ..' % epoch)
					models_update = apply_model(models, batch_size, dataCenter, device, weight_classes_fact, weight_classes_condition, LM_model=LM_model, LM_corpus=LM_corpus, tokenizer=tokenizer, mimo_extractor_fact=mimo_extractor_fact, mimo_extractor_cond=mimo_extractor_cond, pretrain=args.pretrain, multi_head=multi_head, multi_head_two=multi_head_two, is_semi=args.is_semi, eval_pack=[out_file, max_f1, max_std, out_model_name, args.num_pass])

					if args.is_semi:
						print('empty_cache')
						torch.cuda.empty_cache()
						for model in models:
							print("loading model parameters...")
							model = torch.load(out_model_name+model.name)
							print("loading done.")
						max_f1 = 0
						max_std = 0
				else:
					models_update = []
					args.num_pass = 1

				if not args.is_semi:
					print('validation ...')
					max_f1, max_std = evaluation(models, out_file, dataCenter, 0, 0, max_f1, max_std, out_model_name, args.num_pass, False, write_prediction=True, file_name2='./predicts/'+out_file.split('/')[-1], LM_model=LM_model, just_PR=False, LM_corpus=LM_corpus, tokenizer=tokenizer, mimo_extractor_fact=mimo_extractor_fact, mimo_extractor_cond=mimo_extractor_cond, pretrain=args.pretrain, multi_head=multi_head, models_update=models_update, multi_head_two=multi_head_two)
