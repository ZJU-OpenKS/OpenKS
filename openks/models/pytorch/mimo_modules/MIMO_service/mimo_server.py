import sys, os, io
sys.path.append('../')
os.environ[" CUDA_VISIBLE_DEVICES"]="1"

import random
import json
import torch
import argparse
import struct
import math
import itertools
import copy
import re
import time
import warnings
import gevent.pywsgi

warnings.filterwarnings('ignore')

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
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from pytorch_pretrained_bert import BertTokenizer, BertModel
from collections import defaultdict as ddict
from collections import Counter
from collections import OrderedDict

from utils import *
from MIMO import *
from data_center import *
from filter import *

from flask import Flask, request
from flask import render_template
import json, argparse
app = Flask(__name__)

clusters = dict()

parser = argparse.ArgumentParser(description='Implement of SISO, SIMO, MISO, MIMO for Conditional Statement Extraction')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/stmts-eval-sort.tsv',
					help='location of the evaluation set')
parser.add_argument('--output', type=str, default='./predicts/tuples.txt',
					help='location of the saved results')
parser.add_argument('--seed', type=int, default=160824,
					help='random seed')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--local', action='store_true')
parser.add_argument('--wv', action='store_true')
parser.add_argument('--port', type=int, default='9997')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

def device_model(dumped_models):
	models, tokenizer, LM_model, LM_corpus, mimo_extractor_fact, mimo_extractor_cond, multi_head, multi_head_two = dumped_models
	for model in models:
		model.to(device)
		model.device = device
		model.model_LSTM_decoder.device = device
	LM_model.to(device)
	mimo_extractor_fact.to(device)
	mimo_extractor_cond.to(device)
	multi_head.to(device)
	multi_head_two.to(device)
	dumped_models = [models, tokenizer, LM_model, LM_corpus, mimo_extractor_fact, mimo_extractor_cond, multi_head, multi_head_two]
	return dumped_models

def getEmbeddings(phr_list):
	embed_list = []
	for phr in phr_list:
		if phr in wv_model.vocab:
			embed_list.append(wv_model.word_vec(phr))
		# else:
		# 	vec = np.zeros(50, np.float32)
		# 	wrds = word_tokenize(phr)
		# 	for wrd in wrds:
		# 		if wrd in wv_model.vocab: 	
		# 			vec += wv_model.word_vec(wrd)
		# 		else: vec += np.random.randn(50)
		# 	embed_list.append(vec / len(wrds))
		else:
			vecs = []
			wrds = word_tokenize(phr)
			for wrd in wrds:
				if wrd in wv_model.vocab: 	
					vecs.append(wv_model.word_vec(wrd))
				else: vecs.append(np.random.randn(50))
			vec = np.max(np.asarray(vecs), axis=0)
			embed_list.append(vec)

	embs = np.array(embed_list)
	embs = embs/np.expand_dims(np.linalg.norm(embs,ord=2,axis=1), 1)
	return embs

def CFE_Data_Transform(text):
	dataCenter.WPC_Data_Transform(text)
	DATA = dataCenter.get_data_to_predict()
	CFE_list = run_mimo(dumped_models, dataCenter, DATA, device=device)

	statements = {}
	stmts_tagging = [[] for i in range(len(CFE_list))]
	stmt_id = 1
	for CFE in CFE_list:
		sentence = CFE[0]
		index = int(CFE[3].stmt_id.split('stmt ')[-1])-1
		facts, conds = CFE[1:3]
		concpet_indx = []
		attr_indx = []
		predicate_indx = []
		if len(facts) == 0 and len(conds) == 0:
			stmt_id += 1
			stmts_tagging[index] = [sentence, concpet_indx, attr_indx, predicate_indx]
			continue
		fact_tuples = []
		for fact in facts:
			fact_tuples.append([x[0].replace('_', ' ') if x!='NIL' else 'NIL' for x in fact])
			assert len(fact) == 5
			for i in range(len(fact)):
				unit = fact[i]
				if i in [0, 3]:
					concpet_indx.extend(list(range(unit[1], unit[2])) if unit!='NIL' else [])
				if i in [1, 4]:
					attr_indx.extend(list(range(unit[1], unit[2])) if unit!='NIL' else [])
				if i == 2:
					predicate_indx.extend(list(range(unit[1], unit[2])) if unit!='NIL' else [])
			# c1, a1, r, c3, a3 = fact_tuples[-1]
		cond_tuples = []
		for cond in conds:
			cond_tuples.append([x[0].replace('_', ' ') if x!= 'NIL' else 'NIL' for x in cond])
			assert len(cond) == 5
			for i in range(len(cond)):
				unit = cond[i]
				if i in [0, 3]:
					concpet_indx.extend(list(range(unit[1], unit[2])) if unit!='NIL' else [])
				if i in [1, 4]:
					attr_indx.extend(list(range(unit[1], unit[2])) if unit!='NIL' else [])
				if i == 2:
					predicate_indx.extend(list(range(unit[1], unit[2])) if unit!='NIL' else [])
			# c1, a1, r, c3, a3 = cond_tuples[-1]

		concpet_indx = sorted(list(set(concpet_indx)))
		attr_indx = sorted(list(set(attr_indx)))
		predicate_indx = sorted(list(set(predicate_indx)))
		_statement = {'text': ' '.join(sentence), 'fact tuples': fact_tuples, 'condition tuples': cond_tuples, 'concept_indx': concpet_indx, 'attr_indx': attr_indx, 'predicate_indx': predicate_indx}

		_statement = tuple_filter(_statement)

		statements[CFE[3].stmt_id] = _statement
		stmt_id += 1
		stmts_tagging[index] = [sentence, concpet_indx, attr_indx, predicate_indx]

	return statements, stmts_tagging

def get_nice_tuple(_tuple):
	_subject = _tuple[0]
	relation = _tuple[2]
	_object = _tuple[3]
	if _tuple[1] != 'NIL' and _tuple[1] != 'nil':
		_subject = f'{{{_subject}: {_tuple[1]}}}'
	if _tuple[4] != 'NIL' and _tuple[4] != 'nil':
		_object = f'{{{_object}: {_tuple[4]}}}'

	nice_form = f'({_subject}, {relation}, {_object})'
	nice_form = nice_form.replace("'", "`").replace('"', '``')
	return nice_form

def run_search(text):
	statements, stmts_tagging = CFE_Data_Transform(text)

	return statements, stmts_tagging

@app.route('/', methods=['GET', 'POST'])
def home():
	return render_template('index.html')

@app.route('/mimo', methods=['GET', 'POST'])
def mimo():
	if request.method == 'POST':
		start_time = time.time()
		text = request.get_json()['text']

		statements, stmts_tagging = run_search(text)
		print('Searching elapse: ', time.time() - start_time)

		return json.dumps({'statements': statements})
	else:
		return 'Error! in mimo'


def debug():
	text = 'Histone deacetylase inhibitor valproic acid (VPA) has been used to increase the reprogramming efficiency of induced pluripotent stem cell (iPSC) from somatic cells, yet the specific molecular mechanisms underlying this effect is unknown. Here, we demonstrate that reprogramming with lentiviruses carrying the iPSC-inducing factors (Oct4-Sox2-Klf4-cMyc, OSKM) caused senescence in mouse fibroblasts, establishing a stress barrier for cell reprogramming. Administration of VPA protected cells from reprogramming-induced senescent stress. Using an in vitro pre-mature senescence model, we found that VPA treatment increased cell proliferation and inhibited apoptosis through the suppression of the p16/p21 pathway. In addition, VPA also inhibited the G2/M phase blockage derived from the senescence stress. These findings highlight the role of VPA in breaking the cell senescence barrier required for the induction of pluripotency.'

	print('================================================================')
	print(text)

	statements, stmts_tagging = run_search(text)
	print(statements)
	
if __name__ == "__main__":
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# dataCenter = DataCenter(None, args.data)
	# DATA = dataCenter.get_data_to_predict()
	dataCenter = DataCenter()
	print('loading models')
	dumped_models = device_model(torch.load('../resources/dumped_models.pt', map_location=device))
	print('done.')

	print('loading word2vector models')
	if args.wv:
		wv_model = gensim.models.KeyedVectors.load_word2vec_format('./word2vector/glove.6B.50d_word2vec.txt', binary=False)
		threshold = 0.8
	else:
		wv_model = gensim.models.KeyedVectors.load_word2vec_format('../resources/pubmed-vectors=50.bin', binary=True)
		# pickle.dump(wv_model, open('./word2vector/pubmed-wv_model.pkl','wb'))
		# wv_model = pickle.load(open('./word2vector/pubmed-wv_model.pkl','rb'))
		threshold = 0.8
	
	print('done.')

	if args.debug:
		debug()
	else:
		print("MIMO Server Running")
		app.run(host='127.0.0.1', port=args.port)
		# app_server = gevent.pywsgi.WSGIServer(('127.0.0.1', args.port), app)
		# app_server.serve_forever()
