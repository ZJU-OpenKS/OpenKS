import sys, os, io
import random
import json
import torch
import argparse
import gensim
import struct
import math
import itertools
import copy
import nltk

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

from MIMO import *

class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)

class Corpus(object):
	def __init__(self):
		self.dictionary = Dictionary()
		self.load_dictionary()

	def load_dictionary(self):
		print('loading dictionary for language model ...')
		index = 0
		filepath1 = '../resources/LM_dictionary1.txt'
		filepath2 = '../resources/LM_dictionary2.txt'
		filepath3 = '../resources/LM_dictionary3.txt'
		with io.open(filepath1, 'r', encoding="utf8") as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)
				index += 1
				if index%100000 == 0:
					print(index, 'done.')
		with io.open(filepath2, 'r', encoding="utf8") as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)
				index += 1
				if index%10000 == 0:
					print(index, 'done.')
		with io.open(filepath3, 'r', encoding="utf8") as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)
				index += 1
				if index%10000 == 0:
					print(index, 'done.')
		print('done.')

	def tokenize(self, seqword_list):
		seq_len = len(seqword_list[0])
		ids_list = []
		index = 0
		word_set = set()
		for seqword in seqword_list:
			ids = []
			for word in seqword:
				if word not in self.dictionary.word2idx:
					ids.append(self.dictionary.word2idx['<eos>'])
					word_set.add(word)
				else:
					ids.append(self.dictionary.word2idx[word])
			while len(ids) != seq_len:
				ids.append(self.dictionary.word2idx['<eos>'])
			ids_list.append(ids)
			index += 1
			if index%10000 == 0:
				print(index, 'done.')
		return ids_list

def getStmtLabel(docid_filelabel):

	# load concepts, attributes, predicates, and stmts

	docid2struc = {}

	for [docid,filelabel] in docid_filelabel:

		nid2tuple = {}
		hid2tuple = {}
		fid2tuple = {}
		cid2tuple = {}
		sid2stmts = {}

		fr = open(filelabel,'r')
		for line in fr:
			text = line.strip()
			if text == '': continue
			head = text[0]
			if head == '#':
				continue
			elif head == 'n':
				pos = text.find('[')
				arr = text[pos+1:-1].split(',')
				assert(len(arr) == 3 and arr[1] == 'as')
				_id = text[:pos-1]
				assert(not _id in nid2tuple)
				nid2tuple[_id] = [['C',arr[0]],arr[1],['C',arr[2]]]
			elif head == 'h':
				pos = text.find('[')
				arr = text[pos+1:-1].split(',')
				assert(len(arr) == 3 and arr[1] == 'contain')
				_id = text[:pos-1]
				assert(not _id in hid2tuple)		
				hid2tuple[_id] = [['C',arr[0]],arr[1],['C',arr[2]]]
			elif head == 'f':
				pos = text.find('[')
				arr = text[pos+1:-1].split(',')
				assert(len(arr) == 3)
				_tuple = [[],'',[]]
				for i in [0,2]:
					if ':' in arr[i]:
						_arr = arr[i][1:-1].split(':')
						assert(len(_arr) == 2)
						_tuple[i] = ['A',_arr[0],_arr[1]]
					else:
						if arr[i] == 'NIL':
							_tuple[i] = ['N',arr[i]]
						else:
							_tuple[i] = ['C',arr[i]]
				_tuple[1] = arr[1]
				_id = text[:pos-1]
				try:
					assert(not _id in fid2tuple)  
				except:
					print(text)
					sys.exit(1)
				fid2tuple[_id] = _tuple
			elif head == 'c':
				pos = text.find('[')
				arr = text[pos+1:-1].split(',')
				assert(len(arr) == 3)
				_tuple = [[],'',[]]		
				for i in [0,2]:
					if ':' in arr[i]:
						_arr = arr[i][1:-1].split(':')
						assert(len(_arr) == 2)
						_tuple[i] = ['A',_arr[0],_arr[1]]
					else:
						if arr[i] == 'NIL':
							_tuple[i] = ['N',arr[i]]
						else:
							_tuple[i] = ['C',arr[i]]
				_tuple[1] = arr[1]
				_id = text[:pos-1]
				assert(not _id in cid2tuple)
				cid2tuple[_id] = _tuple
			elif head == 's':
				if text[:4] == 'stmt':
					arr = text.split(' ')
					stmt = [[],[],'NIL']
					assert(arr[1] == '=')
					for i in range(2,len(arr)):
						_id = arr[i]
						if _id[0] == 'f':
							assert(_id in fid2tuple)
							stmt[0].append(_id)
						elif _id[0] == 'c':
							assert(_id in cid2tuple)
							stmt[1].append(_id)
						elif _id[0] == '(' and _id[-1] == ')':
							stmt[2] = _id[1:-1]
						else:
							assert(False)
					sid = int(arr[0][4:])
					if not sid in sid2stmts:
						sid2stmts[sid] = []
					sid2stmts[sid].append(stmt)
				elif text[:4] == 's???':
					continue
				else:
					assert(False)
			else:
				assert(False)
		fr.close()

		docid2struc[docid] = [nid2tuple,hid2tuple,fid2tuple,cid2tuple,sid2stmts]

	return docid2struc

def parsing(text):
	seqword,seqpostag,seqanno = [],[],[]
	elems = text.split(' ')
	n = len(elems)
	for i in range(n):
		elem = elems[i]
		if elem.startswith('$C'):
			_arr = elem.split(':')
			phrase = _arr[1]
			arrphrase = phrase.split('_')
			arrpostag = _arr[2].split('_')
			_n = len(arrphrase)
			for j in range(_n):
				seqword.append(arrphrase[j].lower())
				seqpostag.append(arrpostag[j])
				if j == 0:
					seqanno.append('B-C')
				else:
					seqanno.append('I-C')
		elif elem.startswith('$A'):
			_arr = elem.split(':')
			arrphrase = _arr[1].split('_')
			arrpostag = _arr[2].split('_')
			_n = len(arrphrase)
			for j in range(_n):
				seqword.append(arrphrase[j].lower())
				seqpostag.append(arrpostag[j])
				if j == 0:
					seqanno.append('B-A')
				else:
					seqanno.append('I-A')
		elif elem.startswith('$P'):
			_arr = elem.split(':')
			arrphrase = _arr[1].split('_')
			arrpostag = _arr[2].split('_')
			_n = len(arrphrase)
			for j in range(_n):
				seqword.append(arrphrase[j].lower())
				seqpostag.append(arrpostag[j])
				if j == 0:
					seqanno.append('B-P')
				else:
					seqanno.append('I-P')
		else:
			_arr = elem.split(':')
			seqword.append(_arr[0].lower())
			seqpostag.append(_arr[1])
			seqanno.append('O')
	assert len(seqword) == len(seqpostag) == len(seqanno)
	return seqword, seqpostag, seqanno

def getTag2ID(fileName):
	tag2ID = dict()
	with open(fileName, 'r') as f:
		for line in f:
			tag, _id = line.strip().split(' ')
			tag2ID[tag] = int(_id)
	return tag2ID

def getOneHot(index, lenth):
	assert index < lenth
	vector = np.asarray([0]*lenth)
	vector[index] = 1
	return vector

def splitSentence(paragraph):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(paragraph)
	return sentences

class AR_Correcter(object):
	"""docstring for AR_Correcter"""
	def __init__(self, AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold):
		super(AR_Correcter, self).__init__()
		self.A2B_fact = dict()
		self.A2conf_fact = dict()
		self.A2B_cond = dict()
		self.A2conf_cond = dict()

		self._load_AR_file(AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold)

	def _load_AR_file(self, AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold):
		fi = open(AR_fact_file_name, 'r')
		ci = open(AR_condition_file_name, 'r')

		for line in fi:
			A_B, support, confidence = line.strip().split('#')
			support = int(support)
			confidence = float(confidence)
			A, B = A_B.split('-->')

			if support < support_threshold or confidence < confidence_threshold:
				continue

			if not self._is_good_rule(A.split('\t'), B.split('\t')):
				continue

			if A in self.A2B_fact:
				if self.A2conf_fact[A] < confidence:
					self.A2B_fact[A] = B
					self.A2conf_fact[A] = confidence
			else:
				self.A2B_fact[A] = B
				self.A2conf_fact[A] = confidence

		for A in self.A2B_fact:
			print(A, self.A2B_fact[A], self.A2conf_fact[A])

		
		for line in ci:
			A_B, support, confidence = line.strip().split('#')
			support = int(support)
			confidence = float(confidence)
			A, B = A_B.split('-->')

			if support < support_threshold or confidence < confidence_threshold:
				continue

			if not self._is_good_rule(A.split('\t'), B.split('\t')):
				continue

			if A in self.A2B_cond:
				if self.A2conf_cond[A] < confidence:
					self.A2B_cond[A] = B
					self.A2conf_cond[A] = confidence
			else:
				self.A2B_cond[A] = B
				self.A2conf_cond[A] = confidence

		for A in self.A2B_cond:
			print(A, self.A2B_cond[A], self.A2conf_cond[A])

		fi.close()
		ci.close()

	def _is_good_rule(self, pos_sequence, tag_sequence):
		role_set = set()
		for tag in tag_sequence:
			if tag == 'O':
				continue
			role = tag[3]
			role_set.add(role)
		if len(role_set) < 2 or ('2' not in role_set):
			return False
		return True

def smooth_tag_sequence(tag_sequence):
	new_tag_sequence = ['O']
	index = 0
	lenth = len(tag_sequence)
	flag = False
	while index < lenth:
		tag = tag_sequence[index]

		if tag == 'O':
			new_tag = 'O'
		elif not tag.endswith('2P') and not tag.endswith('A'):
			if new_tag_sequence[-1].endswith('2P') or new_tag_sequence[-1].endswith('A'):
				new_tag = 'B'+tag[1:]
			elif new_tag_sequence[-1].startswith('B') or new_tag_sequence[-1].startswith('I'):
				new_tag = 'I'+new_tag_sequence[-1][1:]
			else:
				new_tag = 'B'+tag[1:]
		elif tag.endswith('2P'):
			if new_tag_sequence[-1].endswith('2P'):
				assert tag[1:] == new_tag_sequence[-1][1:]
				new_tag = 'I'+new_tag_sequence[-1][1:]
			else:
				new_tag = 'B'+tag[1:]
		else:
			assert tag.endswith('A')
			if new_tag_sequence[-1].endswith('A'):
				new_tag = 'I'+new_tag_sequence[-1][1:]
			else:
				new_tag = 'B'+tag[1:]

		if new_tag != tag:
			flag = True

		new_tag_sequence.append(new_tag)
		index += 1

	assert len(new_tag_sequence[1:]) == len(tag_sequence)
	return new_tag_sequence[1:], flag

def is_discarded(tag_sequence):
	role_set = set()
	role_type_set = set()
	predicate_set = set()
	for index in range(len(tag_sequence)):
		tag = tag_sequence[index]
		if tag == 'O':
			continue
		if '2P' in tag:
			predicate_set.add(index)
		role = tag[3]
		role_type = tag[3:]
		role_set.add(role)
		role_type_set.add(role_type)

	if len(role_set) < 3: # or '2P' not in role_type_set:
		return True, predicate_set

	if '1A' in role_type_set and '1C' not in role_type_set:
		return True, predicate_set

	if '3A' in role_type_set and '3C' not in role_type_set:
		return True, predicate_set

	return False, predicate_set

class Metrics(object):
	"""docstring for Metrics"""
	def __init__(self):
		super(Metrics, self).__init__()
		self.TP = 0
		self.FP = 0
		self.TN = 0
		self.FN = 0

	def Precision(self):
		if (self.TP == self.FP) and (self.TP == 0):
			return 0
		return float(self.TP) / (self.TP + self.FP)

	def Recall(self):
		if (self.TP == self.FN) and (self.FN == 0):
			return 0
		return float(self.TP) / (self.TP + self.FN)

	def F_1(self):
		precision = self.Precision()
		recall = self.Recall()
		if (precision == recall) and (precision == 0):
			return 0
		return 2 * (precision * recall) / (precision + recall)

# def _get_precision_recall(matrix):
#	precisions = np.asarray(matrix)
#	precisions = np.asarray([precisions[i].max() for i in range(len(precisions))])
#	precision = precisions.sum()/float(len(predicted_tuples)*5)

#	recalls = np.asarray(matrix).transpose()
#	recalls = np.asarray([recalls[i].max() for i in range(len(recalls))])
#	recall = recalls.sum()/float(len(truth_tuples)*5)

#	return (precision, recall, precisions, recalls)

def match_score(truth_tuples, predicted_tuples, _type):
	g_tag_seq = dict()
	p_tag_seq = dict()

	if _type == 'f':
		rolemap = {0:'f1C', 1:'f1A', 2:'f2P', 3:'f3C', 4:'f3A'}
	else:
		assert _type == 'c'
		rolemap = {0:'c1C', 1:'c1A', 2:'c2P', 3:'c3C', 4:'c3A'}

	if len(truth_tuples) == 0:
		truth_tuples = [['NIL', 'NIL', 'NIL', 'NIL', 'NIL']]
	if len(predicted_tuples) == 0:
		predicted_tuples = [['NIL', 'NIL', 'NIL', 'NIL', 'NIL']]

	matrix = []

	for truth_tuple in truth_tuples:
		# tag generation
		for index in range(len(truth_tuple)):
			unit = truth_tuple[index]
			if unit == 'NIL':
				continue
			start = int(unit[1])
			end = int(unit[2])
			token_list = unit[0].split('_')
			assert len(token_list) == (end - start)
			if start not in g_tag_seq:
				g_tag_seq[start] = set()
			g_tag_seq[start].add('B-'+rolemap[index])
			for _index in range(start+1, end):
				if _index not in g_tag_seq:
					g_tag_seq[_index] = set()
				g_tag_seq[_index].add('I-'+rolemap[index])

	for predicted_tuple in predicted_tuples:
		# tg generation
		for index in range(len(predicted_tuple)):
			unit = predicted_tuple[index]
			if unit == 'NIL':
				continue
			start = int(unit[1])
			end = int(unit[2])
			token_list = unit[0].split('_')
			assert len(token_list) == (end - start)
			if start not in p_tag_seq:
				p_tag_seq[start] = set()
			p_tag_seq[start].add('B-'+rolemap[index])
			for _index in range(start+1, end):
				if _index not in p_tag_seq:
					p_tag_seq[_index] = set()
				p_tag_seq[_index].add('I-'+rolemap[index])

		scores = []
		for truth_tuple in truth_tuples:
			assert len(truth_tuple) == 5 and len(predicted_tuple) == 5
			score = 0
			for index in range(len(truth_tuple)):
				t_part = truth_tuple[index]
				p_part = predicted_tuple[index]
				if t_part == p_part:
					score += 1
			scores.append(score)
		matrix.append(scores)

	precisions = np.asarray(matrix)
	precisions = np.asarray([precisions[i].max() for i in range(len(precisions))])
	precision = precisions.sum()/float(len(predicted_tuples)*5)

	recalls = np.asarray(matrix).transpose()
	recalls = np.asarray([recalls[i].max() for i in range(len(recalls))])
	recall = recalls.sum()/float(len(truth_tuples)*5)

	return precision, recall, precisions, recalls, g_tag_seq, p_tag_seq

def is_blocked(start, end, predicate_set):
	if start > end:
		return True
	for predicate in predicate_set:
		if predicate[1] > start and predicate[1] < end:
			return True

# tuple from tag sequence
def post_decoder(words, predicted_fact_tags, ID2Tag=None):
	facts = []

	f1_set = set()
	f1a_set = set()
	f2_set = set()
	f3_set = set()
	f3a_set = set()

	index = 0
	while index < len(words):
		if type(predicted_fact_tags[index]) == type(1):
			tagID = predicted_fact_tags[index]
			tag = ID2Tag[tagID]
		else:
			tag = predicted_fact_tags[index]

		if tag.startswith('B-'):
			string_tag = tag
			string = words[index]
			string_start = index
			index += 1
			if index < len(words):
				if type(predicted_fact_tags[index]) == type(1):
					tagID = predicted_fact_tags[index]
					tag = ID2Tag[tagID]
				else:
					tag = predicted_fact_tags[index]
				while tag.startswith('I'):
					string += ('_' + words[index])
					index += 1
					if index < len(words):
						if type(predicted_fact_tags[index]) == type(1):
							tagID = predicted_fact_tags[index]
							tag = ID2Tag[tagID]
						else:
							tag = predicted_fact_tags[index]
					else:
						break
			string_end = index
			if string_tag.endswith('1C'):
				f1_set.add((string, string_start, string_end))
			elif string_tag.endswith('1A'):
				f1a_set.add((string, string_start, string_end))
				#f1a_set = set()
			elif string_tag.endswith('2P'):
				f2_set.add((string, string_start, string_end))
			elif string_tag.endswith('3C'):
				f3_set.add((string, string_start, string_end))
			elif string_tag.endswith('3A'):
				f3a_set.add((string, string_start, string_end))
				#f3a_set = set()
			else:
				print('error!', string, string_tag, string_start, string_end)
				sys.exit(1)

		else:
			index += 1

	f1_set = sorted(list(f1_set))
	f1a_set = sorted(list(f1a_set))
	f2_set = sorted(list(f2_set))
	f3_set = sorted(list(f3_set))
	f3a_set = sorted(list(f3a_set))

	MIN = 30
	subject2predicate = dict()
	for subject in f1_set:
		min_dis = MIN
		t_predicate = None
		for predicate in f2_set: 
			if is_blocked(subject[1], predicate[1], f2_set):
				continue
			dis = predicate[1]-subject[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_predicate = predicate
		subject2predicate[subject] = t_predicate
	# print subject2predicate

	object2predicate = dict()
	for _object in f3_set:
		min_dis = MIN
		t_predicate = None
		for predicate in f2_set:
			if is_blocked(predicate[1], _object[1], f2_set):
				continue
			dis = _object[1]-predicate[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_predicate = predicate
		object2predicate[_object] = t_predicate
	# print object2predicate

	predicate2subject = dict()
	for predicate in f2_set:
		min_dis = MIN
		t_subject = None
		for subject in f1_set:
			if is_blocked(subject[1], predicate[1], f2_set):
				continue
			dis = predicate[1]-subject[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_subject = subject
		predicate2subject[predicate] = t_subject
	# print predicate2subject

	predicate2object = dict()
	for predicate in f2_set:
		min_dis = MIN
		t_object = None
		for _object in f3_set:
			if is_blocked(predicate[1], _object[1], f2_set):
				continue
			dis = _object[1]-predicate[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_object = _object
		predicate2object[predicate] = t_object
	# print predicate2object

	subject2object = dict()
	for subject in f1_set:
		min_dis = MIN
		t_object = None
		for _object in f3_set:
			dis = _object[1]-subject[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_object = _object
		subject2object[subject] = t_object
	# print subject2object

	object2subject = dict()
	for _object in f3_set:
		min_dis = MIN
		t_subject = None
		for subject in f1_set:
			dis = _object[1]-subject[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_subject = subject
		object2subject[_object] = t_subject
	# print object2subject

	attrib2subject = dict()
	for attrib in f1a_set:
		min_dis = 3
		t_subject = None
		for subject in f1_set:
			dis = subject[1] - attrib[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_subject = subject
		attrib2subject[attrib] = t_subject
	# print attrib2subject

	attrib12predicate = dict()
	for attrib in f1a_set:
		min_dis = 5
		t_predicate = None
		for predicate in f2_set:
			dis = predicate[1] - attrib[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_predicate = predicate
		attrib12predicate[attrib] = t_predicate

	attrib32predicate = dict()
	for attrib in f3a_set:
		min_dis = 5
		t_predicate = None
		for predicate in f2_set:
			dis = attrib[1] - predicate[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_predicate = predicate
		attrib32predicate[attrib] = t_predicate

	attrib2object = dict()
	for attrib in f3a_set:
		min_dis = 3
		t_object = None
		for _object in f3_set:
			dis = _object[1] - attrib[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_object = _object
		attrib2object[attrib] = t_object
	# print attrib2object

	sets = [f1_set, f2_set, f3_set]
	for _set in sets:
		_set.append('NIL')

	candidate_facts = itertools.product(f1_set, set(['NIL']), f2_set, f3_set, set(['NIL']))

	for fact in candidate_facts:
		# print 'candidate_facts:', fact
		fact = list(fact)
		subject = fact[0]
		predicate = fact[2]
		_object = fact[3]

		if subject == 'NIL' and _object == 'NIL':
			if predicate!='NIL' and (predicate2subject[predicate]==None and predicate2object[predicate]==None):
				facts.append(fact)
			else:
				continue
		elif predicate == 'NIL':
			if subject == 'NIL' or _object == 'NIL':
				continue
			elif (subject2object[subject] != _object) and (object2subject[_object] != subject):
				continue
			elif (subject2predicate[subject] != None) or (object2predicate[_object] != None):
				continue
			else:
				facts.append(fact)
		else:
			if subject == 'NIL' and (predicate2subject[predicate]!=None or object2subject[_object]!=None):
				continue
			elif _object == 'NIL' and (predicate2object[predicate]!=None or subject2object[subject]!=None):
				continue
			else:
				if subject != 'NIL' and (subject2predicate[subject] != predicate) and (predicate2subject[predicate] != subject):
					#print '1'
					continue
				elif _object != 'NIL' and (object2predicate[_object] != predicate) and (predicate2object[predicate] != _object):
					#print '2'
					continue
				elif (subject != 'NIL' and _object != 'NIL') and (subject2object[subject] != _object) and (object2subject[_object] != subject):
					#print '3'
					continue
				else:
					facts.append(fact)
	
	extend_facts = []
	for attrib in f1a_set:
		subject = attrib2subject[attrib]
		if subject == None:
			predicate = attrib12predicate[attrib]
			for fact in facts:
				t_predicate = fact[2]
				if t_predicate != predicate:
					continue
				if fact[0] == 'NIL' and fact[1] == 'NIL':
					fact[1] = attrib
			continue
		for fact in facts:
			if fact[2][0] == 'in':
				continue
			t_subject = fact[0]
			if t_subject != subject:
				continue
			if fact[1] == 'NIL':
				fact[1] = attrib
			elif fact[1] != attrib:
				new_fact = copy.deepcopy(fact)
				new_fact[1] = attrib
				extend_facts.append(new_fact)
			for _fact in facts:
				if _fact == fact:
					continue
				if _fact[2:] == fact[2:] and _fact[0] != 'NIL':
					if _fact[0][1] - subject[2] < 0 or _fact[0][1] - subject[2] > 3:
						continue
					if _fact[1] == 'NIL':
						_fact[1] = attrib
					elif _fact[1] != attrib:
						new_fact = copy.deepcopy(_fact)
						new_fact[1] = attrib
						extend_facts.append(new_fact)
	facts.extend(extend_facts)

	extend_facts = []
	for attrib in f3a_set:
		_object = attrib2object[attrib]
		if _object == None:
			predicate = attrib32predicate[attrib]
			for fact in facts:
				t_predicate = fact[2]
				if t_predicate != predicate:
					continue
				if fact[3] == 'NIL' and fact[4] == 'NIL':
					fact[4] = attrib
			continue
		for fact in facts:
			t_object = fact[3]
			if t_object != _object:
				continue
			if fact[4] == 'NIL':
				fact[4] = attrib
			elif fact[4] != attrib:
				new_fact = copy.deepcopy(fact)
				new_fact[4] = attrib
				extend_facts.append(new_fact)
			for _fact in facts:
				if _fact == fact:
					continue
				if _fact[:2] == fact[:2] and _fact[3] != 'NIL':
					if _fact[3][1] - _object[2] < 0 or _fact[3][1] - _object[2] > 3:
						continue
					if _fact[4] == 'NIL':
						_fact[4] = attrib
					elif _fact[4] != attrib:
						new_fact = copy.deepcopy(_fact)
						new_fact[4] = attrib
						extend_facts.append(new_fact)
	facts.extend(extend_facts)

	return facts

def revise_max(distribs, tags, threshold):
	assert len(distribs) == len(tags)
	for index in range(len(tags)):
		if distribs[index].item() > torch.log(torch.Tensor([threshold])).item():
			continue
		tags[index] = 0

def get_f1(precision, recall):
	if (precision == recall) and (precision == 0):
		return 0
	return 2 * (precision * recall) / (precision + recall)

def metric_to_list(metric):
	metric_list = []
	for tag in sorted(metric):
		tag2metric = [metric[tag].Precision()*100, metric[tag].Recall()*100, metric[tag].F_1()*100]
		metric_list.append(tag2metric)
	return metric_list

def tag2tag_to_list(tag2tag):
	tag2tag_list = []
	for tag in sorted(tag2tag):
		row = []
		for vtag in sorted(tag2tag[tag]):
			row.append(tag2tag[tag][vtag])
		tag2tag_list.append(row)
	return tag2tag_list

# def prediction(models, file_name, dataCenter, threshold_fact, threshold_cond, max_f1, max_std, out_model_name, num_pass, just_eval, write_prediction=False, file_name2=None, just_PR=False, tokenizer=None, LM_model=None, LM_corpus=None, mimo_extractor_fact=None, mimo_extractor_cond=None, pretrain=False, ensemble=None, models_update=[], ensemble_two=None):
# 	DATA = dataCenter.get_data_to_predict()
# 	_prediction(models, dataCenter, DATA, threshold_fact, threshold_cond, write_prediction, tokenizer=tokenizer, LM_model=LM_model, LM_corpus=LM_corpus, mimo_extractor_fact=mimo_extractor_fact, mimo_extractor_cond=mimo_extractor_cond, pretrain=pretrain, ensemble=ensemble, ensemble_two=ensemble_two)

def evaluation(models, file_name, dataCenter, threshold_fact, threshold_cond, max_f1, max_std, out_model_name, num_pass, just_eval, write_prediction=False, file_name2=None, just_PR=False, tokenizer=None, LM_model=None, LM_corpus=None, mimo_extractor_fact=None, mimo_extractor_cond=None, pretrain=False, ensemble=None, models_update=[], ensemble_two=None, device=None):

	valid_tag_levels = []
	valid_tuple_levels = []
	valid_tag2metric_list = []
	valid_tag2tag_fact_list = []
	valid_tag2tag_cond_list = []

	test_tag_levels = []
	test_tuple_levels = []
	test_tag2metric_list = []
	test_tag2tag_fact_list = []
	test_tag2tag_cond_list = []

	for i in range(num_pass):
		VALID_DATA, TEST_DATA = dataCenter.get_evaluation(1.0/num_pass)
		valid_tag_level, valid_Metrics, valid_tuple_level, valid_predicted_str = _evaluation(models, dataCenter, VALID_DATA, threshold_fact, threshold_cond, write_prediction, device, tokenizer=tokenizer, LM_model=LM_model, LM_corpus=LM_corpus, mimo_extractor_fact=mimo_extractor_fact, mimo_extractor_cond=mimo_extractor_cond, pretrain=pretrain, ensemble=ensemble, ensemble_two=ensemble_two)
		valid_tag_levels.append(valid_tag_level)
		valid_tuple_levels.append(valid_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = valid_Metrics
		valid_tag2metric_list.append(metric_to_list(Tag2Metrics))
		valid_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		valid_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

		if len(TEST_DATA[0]) == 0:
			continue

		test_tag_level, test_Metrics, test_tuple_level, test_predicted_str = _evaluation(models, dataCenter, TEST_DATA, threshold_fact, threshold_cond, write_prediction, device, tokenizer=tokenizer, LM_model=LM_model, LM_corpus=LM_corpus, mimo_extractor_fact=mimo_extractor_fact, mimo_extractor_cond=mimo_extractor_cond, pretrain=pretrain, ensemble=ensemble, ensemble_two=ensemble_two)
		test_tag_levels.append(test_tag_level)
		test_tuple_levels.append(test_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = test_Metrics
		test_tag2metric_list.append(metric_to_list(Tag2Metrics))
		test_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		test_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

	valid_tag_levels_mean = np.asarray(valid_tag_levels).mean(0)
	valid_tag_levels_std = np.asarray(valid_tag_levels).std(0)

	valid_tag2metric_mean = np.asarray(valid_tag2metric_list).mean(0)
	valid_tag2metric_std = np.asarray(valid_tag2metric_list).std(0)
	valid_tag2tag_fact_mean = np.asarray(valid_tag2tag_fact_list).mean(0)
	valid_tag2tag_fact_std = np.asarray(valid_tag2tag_fact_list).std(0)
	valid_tag2tag_cond_mean = np.asarray(valid_tag2tag_cond_list).mean(0)
	valid_tag2tag_cond_std = np.asarray(valid_tag2tag_cond_list).std(0)

	valid_tuple_levels_mean = np.asarray(valid_tuple_levels).mean(0)
	valid_tuple_levels_std = np.asarray(valid_tuple_levels).std(0)

	print(valid_tag_levels_mean[-1][-1], valid_tuple_levels_mean[-1][-1])

	if len(TEST_DATA[0]) != 0:
		test_tag_levels_mean = np.asarray(test_tag_levels).mean(0)
		test_tag_levels_std = np.asarray(test_tag_levels).std(0)

		test_tag2metric_mean = np.asarray(test_tag2metric_list).mean(0)
		test_tag2metric_std = np.asarray(test_tag2metric_list).std(0)
		test_tag2tag_fact_mean = np.asarray(test_tag2tag_fact_list).mean(0)
		test_tag2tag_fact_std = np.asarray(test_tag2tag_fact_list).std(0)
		test_tag2tag_cond_mean = np.asarray(test_tag2tag_cond_list).mean(0)
		test_tag2tag_cond_std = np.asarray(test_tag2tag_cond_list).std(0)

		test_tuple_levels_mean = np.asarray(test_tuple_levels).mean(0)
		test_tuple_levels_std = np.asarray(test_tuple_levels).std(0)
		
		print(test_tag_levels_mean[-1][-1], test_tuple_levels_mean[-1][-1])

	macro_F1 = valid_tag_levels_mean[-1][-1]
	macro_std = valid_tag_levels_std[-1][-1]

	if just_PR:
		return
	if macro_F1 > max_f1:
		fo = open(file_name2, 'w')
		fo.write(valid_predicted_str)
		if len(TEST_DATA[0]) != 0:
			fo.write(test_predicted_str)
		fo.close()

		max_f1 = macro_F1
		max_std = macro_std
		better = True
		print(max_f1, max_std)
		if not just_eval:
			print('saving model ...')
			# assert len(models_update) != 0
			for model in models_update:
				#torch.save(model.state_dict(), out_model_name+model.name)
				torch.save(model, out_model_name+model.name)
			print('saving done.')

		fo = open(file_name, 'w')
		if len(TEST_DATA[0]) != 0:
			for i in range(len(test_tag_levels_mean)):
				for j in range(len(test_tag_levels_mean[i])):
					fo.write('%.2f+/-%.2f\t' % (test_tag_levels_mean[i][j], test_tag_levels_std[i][j]))
			for i in range(len(test_tuple_levels_mean)):
				for j in range(len(test_tuple_levels_mean[i])):
					fo.write('%.2f+/-%.2f\t' % (test_tuple_levels_mean[i][j], test_tuple_levels_std[i][j]))

		for i in range(len(valid_tag_levels_mean)):
			for j in range(len(valid_tag_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tag_levels_mean[i][j], valid_tag_levels_std[i][j]))
		for i in range(len(valid_tuple_levels_mean)):
			for j in range(len(valid_tuple_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tuple_levels_mean[i][j], valid_tuple_levels_std[i][j]))

		fo.write('\n')

		i = 0
		assert len(Tag2Metrics) == len(valid_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(valid_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(valid_tag2tag_fact_mean) == len(valid_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = valid_tag2tag_fact_mean[i][j]
				std = valid_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(valid_tag2tag_cond_mean) == len(valid_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = valid_tag2tag_cond_mean[i][j]
				std = valid_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		if len(TEST_DATA[0]) == 0:
			fo.close()
			return max_f1, max_std

		i = 0
		assert len(Tag2Metrics) == len(test_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(test_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(test_tag2tag_fact_mean) == len(test_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = test_tag2tag_fact_mean[i][j]
				std = test_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(test_tag2tag_cond_mean) == len(test_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = test_tag2tag_cond_mean[i][j]
				std = test_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		fo.close()

	return max_f1, max_std

def is_non(tagseq):
	for tag in tagseq:
		if tag != 'O':
			return False
	return True

def _core_evaluation(OUTs, SENTENCEs, instance_list, predict_fact_batch, predict_condition_batch, predict_fact_tuples_batch, predict_cond_tuples_batch, dataCenter, threshold_fact, threshold_cond, write_prediction=False, is_filter=False):
	assert len(OUTs) == len(predict_fact_batch) == len(predict_condition_batch)
	pretrain = False
	if len(predict_fact_tuples_batch) == 0 and len(predict_cond_tuples_batch) == 0:
		pretrain = True

	Tag2Metrics = dict()
	Tag2Metrics_fact = dict()
	Tag2Metrics_condition = dict()

	for tag in dataCenter.Tag2ID_fact:
		Tag2Metrics[tag] = Metrics()
		Tag2Metrics_fact[tag] = Metrics()
	for tag in dataCenter.Tag2ID_condition:
		Tag2Metrics[tag] = Metrics()
		Tag2Metrics_condition[tag] = Metrics()

	tag2tag_fact = dict()
	tag2tag_cond = dict()

	precision_sum_f = 0
	recall_sum_f = 0
	precision_sum_c = 0
	recall_sum_c = 0

	precisions_f = []
	recalls_f = []
	precisions_c = []
	recalls_c = []

	for tag in dataCenter.Tag2ID_fact:
		if tag not in tag2tag_fact:
			tag2tag_fact[tag] = dict()
			for _tag in dataCenter.Tag2ID_fact:
				tag2tag_fact[tag][_tag] = 0

	for tag in dataCenter.Tag2ID_condition:
		if tag not in tag2tag_cond:
			tag2tag_cond[tag] = dict()
			for _tag in dataCenter.Tag2ID_condition:
				tag2tag_cond[tag][_tag] = 0

	predicted_str = ''
	for i in range(len(predict_fact_batch)):
		seq_len = len(instance_list[i].SENTENCE)
		predicted_fact_tag_distribs, predicted_fact_tags = torch.max(predict_fact_batch[i], 1)
		predicted_conditions_tag_distribs, predicted_conditions_tags = torch.max(predict_condition_batch[i], 1)

		merge_fact_tags = [dataCenter.ID2Tag_fact[tag_id.item()] for tag_id in predicted_fact_tags[:seq_len]]
		merge_cond_tags = [dataCenter.ID2Tag_condition[tag_id.item()] for tag_id in predicted_conditions_tags[:seq_len]]

		instance = instance_list[i]
		assert len(instance.SENTENCE) == len(SENTENCEs[i])

		facts = []
		conditions = []
		multiple_fact_tags = []
		multiple_cond_tags = []
		if not pretrain:
			# [multiple, seqlen]
			_, predicted_fact_tuple_tags = torch.max(predict_fact_tuples_batch[i], 2)
			for k in range(len(predicted_fact_tuple_tags)):
				fact_tags = [dataCenter.ID2Tag_fact[tag_id.item()] for tag_id in predicted_fact_tuple_tags[k][:seq_len]]
				multiple_fact_tags.append(fact_tags)
				facts.extend(post_decoder(instance.multi_input[0][1], fact_tags, dataCenter.ID2Tag_fact))

			_, predicted_cond_tuple_tags = torch.max(predict_cond_tuples_batch[i], 2)
			for k in range(len(predicted_cond_tuple_tags)):
				cond_tags = [dataCenter.ID2Tag_condition[tag_id.item()] for tag_id in predicted_cond_tuple_tags[k][:seq_len]]
				multiple_cond_tags.append(cond_tags)
				conditions.extend(post_decoder(instance.multi_input[0][1], cond_tags, dataCenter.ID2Tag_condition))

			max_multi_fact_nu = max(len(OUTs[i][0])-1, len(multiple_fact_tags))
			max_multi_cond_nu = max(len(OUTs[i][1])-1, len(multiple_cond_tags))
		else:
			multiple_fact_tags = [merge_fact_tags]
			multiple_cond_tags = [merge_cond_tags] 
			facts = post_decoder(instance.multi_input[0][1], merge_fact_tags, dataCenter.ID2Tag_fact)
			conditions = post_decoder(instance.multi_input[0][1], merge_cond_tags, dataCenter.ID2Tag_condition)
			max_multi_fact_nu = 1
			max_multi_cond_nu = 1



		new_fact_tags = ['O']*seq_len
		for tag_seq in multiple_fact_tags:
			assert len(tag_seq) == seq_len
			for _index in range(len(tag_seq)):
				if tag_seq[_index] != 'O':
					new_fact_tags[_index] = tag_seq[_index]
		new_fact_tags = [new_fact_tags]

		new_cond_tags = ['O']*seq_len
		for tag_seq in multiple_cond_tags:
			assert len(tag_seq) == seq_len
			for _index in range(len(tag_seq)):
				if tag_seq[_index] != 'O':
					new_cond_tags[_index] = tag_seq[_index]
		new_cond_tags = [new_cond_tags]

		new_t_fact_tags = ['O']*seq_len
		for tag_seq in OUTs[i][0][1:]:
			assert len(tag_seq) == seq_len
			for _index in range(len(tag_seq)):
				if tag_seq[_index] != 0:
					new_t_fact_tags[_index] = dataCenter.ID2Tag_fact[tag_seq[_index]]
		new_t_fact_tags = [new_t_fact_tags]

		new_t_cond_tags = ['O']*seq_len
		for tag_seq in OUTs[i][1][1:]:
			assert len(tag_seq) == seq_len
			for _index in range(len(tag_seq)):
				if tag_seq[_index] != 0:
					new_t_cond_tags[_index] = dataCenter.ID2Tag_condition[tag_seq[_index]]
		new_t_cond_tags = [new_t_cond_tags]

		max_multi_fact_nu = 1
		max_multi_cond_nu = 1
		
		index_ids = (predicted_fact_tags==dataCenter.Tag2ID_fact['B-f2P']).nonzero()
		for k in range(max_multi_fact_nu):
			for j in range(seq_len):
				if k >= len(OUTs[i][0])-1:
					y_true = 'O'
				else:
					if not pretrain:
						# y_true = dataCenter.ID2Tag_fact[OUTs[i][0][k+1][j]]
						y_true = new_t_fact_tags[k][j]
					else:
						y_true = dataCenter.ID2Tag_fact[OUTs[i][0][k][j]]
				if k >= len(new_fact_tags):
					y_predict = 'O'
				else:
					y_predict = new_fact_tags[k][j]

				if y_true == y_predict:
					tag = y_predict
					Tag2Metrics[tag].TP += 1
					Tag2Metrics_fact[tag].TP += 1
					tag2tag_fact[tag][tag] += 1
				elif y_true != y_predict:
					tag_true = y_true
					tag_predict = y_predict
					Tag2Metrics[tag_true].FN += 1
					Tag2Metrics[tag_predict].FP += 1
					Tag2Metrics_fact[tag_true].FN += 1
					Tag2Metrics_fact[tag_predict].FP += 1
					tag2tag_fact[tag_true][tag_predict] += 1

		for k in range(max_multi_cond_nu):
			for j in range(seq_len):
				if k >= len(OUTs[i][1])-1:
					y_true = 'O'
				else:
					if not pretrain:
						# y_true = dataCenter.ID2Tag_condition[OUTs[i][1][k+1][j]]
						y_true = new_t_cond_tags[k][j]
					else:
						y_true = dataCenter.ID2Tag_condition[OUTs[i][1][k][j]]
				if k >= len(new_cond_tags):
					y_predict = 'O'
				else:
					y_predict = new_cond_tags[k][j]

				if y_true == y_predict:
					tag = y_predict
					Tag2Metrics[tag].TP += 1
					Tag2Metrics_condition[tag].TP += 1
					tag2tag_cond[tag][tag] += 1
				elif y_true != y_predict:
					tag_true = y_true
					tag_predict = y_predict
					Tag2Metrics[tag_true].FN += 1
					Tag2Metrics[tag_predict].FP += 1
					Tag2Metrics_condition[tag_true].FN += 1
					Tag2Metrics_condition[tag_predict].FP += 1
					tag2tag_cond[tag_true][tag_predict] += 1

		_facts = []
		_conditions = []
		for out in instance.multi_output:
			if out[0].startswith('f'):
				__facts = post_decoder(instance.multi_input[0][1], out[1], dataCenter.ID2Tag_fact)
				_facts.extend(__facts)
			else:
				__conditions = post_decoder(instance.multi_input[0][1], out[1], dataCenter.ID2Tag_condition)
				_conditions.extend(__conditions)

		p_f, r_f, ps_f, rs_f, fg_tag_seq, fp_tag_seq = match_score(_facts, facts, 'f')
		p_c, r_c, ps_c, rs_c, cg_tag_seq, cp_tag_seq = match_score(_conditions, conditions, 'c')

		precision_sum_f += p_f
		recall_sum_f += r_f
		precision_sum_c += p_c
		recall_sum_c += r_c

		precisions_f.extend(ps_f)
		recalls_f.extend(rs_f)
		precisions_c.extend(ps_c)
		recalls_c.extend(rs_c)

		# for _index in range(seq_len):
		# 	if _index not in fg_tag_seq:
		# 		fg_tag_seq[_index] = set()
		# 		fg_tag_seq[_index].add('O')
		# 	if _index not in fp_tag_seq:
		# 		fp_tag_seq[_index] = set()
		# 		fp_tag_seq[_index].add('O')
		# 	if _index not in cg_tag_seq:
		# 		cg_tag_seq[_index] = set()
		# 		cg_tag_seq[_index].add('O')
		# 	if _index not in cp_tag_seq:
		# 		cp_tag_seq[_index] = set()
		# 		cp_tag_seq[_index].add('O')
		# print(fg_tag_seq, fp_tag_seq)
		# print(cg_tag_seq, cp_tag_seq)

		# for _index in fp_tag_seq:
		# 	for tag in fp_tag_seq[_index]:
		# 		if tag not in fg_tag_seq[_index]:
		# 			Tag2Metrics[tag].FP += 1
		# 			Tag2Metrics_fact[tag].FP += 1
		# 			for _tag in fg_tag_seq[_index]:
		# 				tag2tag_fact[_tag][tag] += 1
		# 				break
		# 		else:
		# 			Tag2Metrics[tag].TP += 1
		# 			Tag2Metrics_fact[tag].TP += 1
		# 			tag2tag_fact[tag][tag] += 1

		# 	for tag in fg_tag_seq[_index]:
		# 		if tag not in fp_tag_seq[_index]:
		# 			Tag2Metrics[tag].FN += 1
		# 			Tag2Metrics_fact[tag].FN += 1
		# 			for _tag in fp_tag_seq[_index]:
		# 				tag2tag_fact[tag][_tag] += 1
		# 				break

		# 	for tag in cp_tag_seq[_index]:
		# 		if tag not in cg_tag_seq[_index]:
		# 			Tag2Metrics[tag].FP += 1
		# 			Tag2Metrics_condition[tag].FP += 1
		# 			for _tag in cg_tag_seq[_index]:
		# 				tag2tag_cond[_tag][tag] += 1
		# 				break
		# 		else:
		# 			Tag2Metrics[tag].TP += 1
		# 			Tag2Metrics_condition[tag].TP += 1
		# 			tag2tag_cond[tag][tag] += 1

		# 	for tag in cg_tag_seq[_index]:
		# 		if tag not in cp_tag_seq[_index]:
		# 			Tag2Metrics[tag].FN += 1
		# 			Tag2Metrics_condition[tag].FN += 1
		# 			for _tag in cp_tag_seq[_index]:
		# 				tag2tag_cond[tag][_tag] += 1
		# 				break
					

		if write_prediction:
			predicted_str += ('===== %s stmt%s =====\n' % (str(instance_list[i].paper_id), str(instance_list[i].stmt_id)))
			predicted_str += ('WORD\t'+'\t'.join(instance_list[i].SENTENCE)+'\n')
			predicted_str += ('POSTAG\t'+'\t'.join(instance_list[i].POSTAG)+'\n')
			predicted_str += ('CAP\t'+'\t'.join(instance_list[i].CAP)+'\n')

			predicted_str += ('f\t'+'\t'.join(merge_fact_tags)+'\n')
			if not pretrain:
				for k in range(len(multiple_fact_tags)):
					fact_tags = multiple_fact_tags[k]
					predicted_str += ('f\t'+'\t'.join(fact_tags)+'\n')

			predicted_str += ('c\t'+'\t'.join(merge_cond_tags)+'\n')
			if not pretrain:
				for k in range(len(multiple_cond_tags)):
					cond_tags = multiple_cond_tags[k]
					predicted_str += ('c\t'+'\t'.join(cond_tags)+'\n')

			if not pretrain:
				for k in range(1, len(instance_list[i].OUT[0])):
					fact_g_tags = [dataCenter.ID2Tag_fact[tag_id] for tag_id in instance_list[i].OUT[0][k]]
					predicted_str += ('f_g\t'+'\t'.join(fact_g_tags)+'\n')
				for k in range(1, len(instance_list[i].OUT[1])):
					cond_g_tags = [dataCenter.ID2Tag_condition[tag_id] for tag_id in instance_list[i].OUT[1][k]]
					predicted_str += ('c_g\t'+'\t'.join(cond_g_tags)+'\n')
			else:
				fact_g_tags = [dataCenter.ID2Tag_fact[tag_id] for tag_id in instance_list[i].OUT[0][0]]
				predicted_str += ('f_g\t'+'\t'.join(fact_g_tags)+'\n')
				cond_g_tags = [dataCenter.ID2Tag_condition[tag_id] for tag_id in instance_list[i].OUT[1][0]]
				predicted_str += ('c_g\t'+'\t'.join(cond_g_tags)+'\n')

	microEval = Metrics()
	microEval_fact = Metrics()
	microEval_condition = Metrics()
	macro_P = 0
	macro_R = 0
	macro_P_fact = 0
	macro_R_fact = 0
	macro_P_condition = 0
	macro_R_condition = 0
	# print threshold_fact, threshold_cond
	for tag in Tag2Metrics:
		if tag == 'O':
			continue

		macro_P += Tag2Metrics[tag].Precision()*100
		macro_R += Tag2Metrics[tag].Recall()*100
		microEval.TP += Tag2Metrics[tag].TP
		microEval.FP += Tag2Metrics[tag].FP
		microEval.FN += Tag2Metrics[tag].FN

		if tag in dataCenter.Tag2ID_fact:
			macro_P_fact += Tag2Metrics_fact[tag].Precision()*100
			macro_R_fact += Tag2Metrics_fact[tag].Recall()*100
			microEval_fact.TP += Tag2Metrics_fact[tag].TP
			microEval_fact.FP += Tag2Metrics_fact[tag].FP
			microEval_fact.FN += Tag2Metrics_fact[tag].FN
		elif tag in dataCenter.Tag2ID_condition:
			macro_P_condition += Tag2Metrics_condition[tag].Precision()*100
			macro_R_condition += Tag2Metrics_condition[tag].Recall()*100
			microEval_condition.TP += Tag2Metrics_condition[tag].TP
			microEval_condition.FP += Tag2Metrics_condition[tag].FP
			microEval_condition.FN += Tag2Metrics_condition[tag].FN
		else:
			print('error')
			sys.exit(1)

	macro_P /= (len(Tag2Metrics) - 1)
	macro_R /= (len(Tag2Metrics) - 1)
	macro_P_fact /= (len(Tag2Metrics_fact) - 1)
	macro_R_fact /= (len(Tag2Metrics_fact) - 1)
	macro_P_condition /= (len(Tag2Metrics_condition) - 1)
	macro_R_condition /= (len(Tag2Metrics_condition) - 1)

	if (macro_P == macro_R == 0):
		macro_F1 = 0
	else:
		macro_F1 = 2 * (macro_P * macro_R) / (macro_P + macro_R)

	if (macro_P_fact == macro_R_fact == 0):
		macro_F1_fact = 0
	else:
		macro_F1_fact = 2 * (macro_P_fact * macro_R_fact) / (macro_P_fact + macro_R_fact)

	if (macro_P_condition == macro_R_condition == 0):
		macro_F1_condition = 0
	else:
		macro_F1_condition = 2 * (macro_P_condition * macro_R_condition) / (macro_P_condition + macro_R_condition)

	tag_level_fact = [microEval_fact.Precision()*100, microEval_fact.Recall()*100, microEval_fact.F_1()*100, macro_P_fact, macro_R_fact, macro_F1_fact]
	tag_level_cond = [microEval_condition.Precision()*100, microEval_condition.Recall()*100, microEval_condition.F_1()*100, macro_P_condition, macro_R_condition, macro_F1_condition]
	tag_level = [microEval.Precision()*100, microEval.Recall()*100, microEval.F_1()*100, macro_P, macro_R, macro_F1]

	precisions_f = np.asarray(precisions_f)
	precisions_c = np.asarray(precisions_c)
	recalls_f = np.asarray(recalls_f)
	recalls_c = np.asarray(recalls_c)

	micro_precision_f = precisions_f.sum()/float(len(precisions_f)*5)
	micro_recall_f = recalls_f.sum()/float(len(recalls_f)*5)
	micro_f1_f = get_f1(micro_precision_f, micro_recall_f)

	micro_precision_c = precisions_c.sum()/float(len(precisions_c)*5)
	micro_recall_c = recalls_c.sum()/float(len(recalls_c)*5)
	micro_f1_c = get_f1(micro_precision_c, micro_recall_c)
	macro_precision_f = precision_sum_f/len(OUTs)
	macro_recall_f = recall_sum_f/len(OUTs)
	macro_f1_f = get_f1(macro_precision_f, macro_recall_f)

	macro_precision_c = precision_sum_c/len(OUTs)
	macro_recall_c = recall_sum_c/len(OUTs)
	macro_f1_c = get_f1(macro_precision_c, macro_recall_c)

	precisions = np.concatenate((precisions_f, precisions_c))
	recalls = np.concatenate((recalls_f, recalls_c))
	micro_precision = precisions.sum()/float(len(precisions)*5)
	micro_recall = recalls.sum()/float(len(recalls)*5)
	micro_f1 = get_f1(micro_precision, micro_recall)

	macro_precision = (macro_precision_f+macro_precision_c)/2
	macro_recall = (macro_recall_f+macro_recall_c)/2
	macro_f1 = get_f1(macro_precision, macro_recall)

	tuple_level_fact = [micro_precision_f*100, micro_recall_f*100, micro_f1_f*100, macro_precision_f*100, macro_recall_f*100, macro_f1_f*100]
	tuple_level_cond = [micro_precision_c*100, micro_recall_c*100, micro_f1_c*100, macro_precision_c*100, macro_recall_c*100, macro_f1_c*100]
	tuple_level = [micro_precision*100, micro_recall*100, micro_f1*100, macro_precision*100, macro_recall*100, macro_f1*100]

	return [tag_level_fact, tag_level_cond, tag_level], [Tag2Metrics, tag2tag_fact, tag2tag_cond], [tuple_level_fact, tuple_level_cond, tuple_level], predicted_str

def _core_predict(SENTENCEs, POSTAGs, CAPs, instance_list, models, dataCenter, device=torch.device("cuda"), tokenizer=None, LM_model=None, LM_corpus=None, mimo_extractor_fact=None, mimo_extractor_cond=None, pretrain=False, ensemble=None, ensemble_two=None, writer=None):
	predict_fact = []
	predict_condition = []
	predict_fact_tuples = []
	predict_cond_tuples = []
	CFE_list = []
	batch_size = 20
	batches = len(SENTENCEs) // batch_size
	if len(SENTENCEs) % batch_size != 0:
		batches += 1
	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size: (index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size: (index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size: (index+1)*batch_size]
		instances_batch = instance_list[index*batch_size: (index+1)*batch_size]
		predict_fact_batch, predict_condition_batch, hidden_out_batch = Tag_Extractor(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, models, tokenizer, LM_model, LM_corpus, device, ensemble, ensemble_two)

		if not pretrain:
			predict_fact_tuples_batch, predict_cond_tuples_batch = Tuple_Extractor(predict_fact_batch, predict_condition_batch, hidden_out_batch, mimo_extractor_fact, mimo_extractor_cond, dataCenter)
			# if writer == None:
			# 	predict_fact_tuples.extend(predict_fact_tuples_batch)
			# 	predict_cond_tuples.extend(predict_cond_tuples_batch)

			CFE_list.extend(write_tuples(instances_batch, predict_fact_tuples_batch, predict_cond_tuples_batch, dataCenter, writer))
		# if writer == None:
		# 	for row in predict_fact_batch:
		# 		predict_fact.append(row)
		# 	for row in predict_condition_batch:
		# 		predict_condition.append(row)
		if (index+1)*batch_size % 100 == 0:
			print((index+1)*batch_size, 'done.')
	#assert len(predict_fact) == len(predict_fact_tuples)
	return predict_fact, predict_condition, predict_fact_tuples, predict_cond_tuples, CFE_list

def write_tuples(instance_list, fact_tuples_batch, cond_tuples_batch, dataCenter, writer):
	assert len(fact_tuples_batch) == len(cond_tuples_batch)
	#writer = open(tuple_file, 'w')
	CFE_list = []
	for i in range(len(fact_tuples_batch)):
		seq_len = len(instance_list[i].SENTENCE)
		instance = instance_list[i]
		facts = []
		conditions = []
		multiple_fact_tags = []
		multiple_cond_tags = []

		_, fact_tuple_tags = torch.max(fact_tuples_batch[i], 2)
		for k in range(len(fact_tuple_tags)):
			fact_tags = [dataCenter.ID2Tag_fact[tag_id.item()] for tag_id in fact_tuple_tags[k][:seq_len]]
			facts.extend(post_decoder(instance.multi_input[0][1], fact_tags, dataCenter.ID2Tag_fact))

		_, cond_tuple_tags = torch.max(cond_tuples_batch[i], 2)
		for k in range(len(cond_tuple_tags)):
			cond_tags = [dataCenter.ID2Tag_condition[tag_id.item()] for tag_id in cond_tuple_tags[k][:seq_len]]
			conditions.extend(post_decoder(instance.multi_input[0][1], cond_tags, dataCenter.ID2Tag_condition))
		if writer != None:
			writer.write('===== %s stmt%s =====\n%s\n' % (instance.paper_id, instance.stmt_id, ' '.join(instance.SENTENCE)))

			for i, fact in enumerate(facts):
				writer.write('fact%d:\t%s\n' % (i+1, str(fact)))
			for i, cond in enumerate(conditions):
				writer.write('condition%d:\t%s\n' % (i+1, str(cond)))
		print(f'[Done]: {instance.SENTENCE}')
		CFE_list.append([instance.SENTENCE, facts, conditions, instance])
	return CFE_list

def run_mimo(dumped_models, dataCenter, data, threshold_fact=0, threshold_cond=0, write_prediction=True, device=None, pretrain=False, output_file=None):
	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_OUTs, raw_instance_list = data
	#SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	MICS = zip(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_OUTs, raw_instance_list)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)

	if output_file != None:
		writer = open(output_file, 'w')
	else:
		writer = None
	models, tokenizer, LM_model, LM_corpus, mimo_extractor_fact, mimo_extractor_cond, multi_head, multi_head_two = dumped_models
	predict_fact, predict_condition, predict_fact_tuples, predict_cond_tuples, CFE_list = _core_predict(SENTENCEs, POSTAGs, CAPs, instance_list, models, dataCenter, device, tokenizer, LM_model, LM_corpus, mimo_extractor_fact, mimo_extractor_cond, pretrain, multi_head, multi_head_two, writer=writer)
	return CFE_list

def prediction(models, dataCenter, data, threshold_fact, threshold_cond, write_prediction=False, device=None, tokenizer=None, LM_model=None, LM_corpus=None, mimo_extractor_fact=None, mimo_extractor_cond=None, pretrain=False, ensemble=None, ensemble_two=None, output_file=None):

	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_OUTs, raw_instance_list = data
	#SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	MICS = zip(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_OUTs, raw_instance_list)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)
	print(len(SENTENCEs[0]))

	writer = open(output_file, 'w')

	predict_fact, predict_condition, predict_fact_tuples, predict_cond_tuples = _core_predict(SENTENCEs, POSTAGs, CAPs, instance_list, models, dataCenter, device, tokenizer, LM_model, LM_corpus, mimo_extractor_fact, mimo_extractor_cond, pretrain, ensemble, ensemble_two, writer=writer)

	#write_tuples(instance_list, predict_fact_tuples, predict_cond_tuples, dataCenter, output_file)

	#assert len(predict_fact) == len(SENTENCEs) == len(OUTs)

#	return _core_evaluation(OUTs, SENTENCEs, instance_list, predict_fact, predict_condition, predict_fact_tuples, predict_cond_tuples, dataCenter, threshold_fact, threshold_cond, write_prediction)

def _evaluation(models, dataCenter, data, threshold_fact, threshold_cond, write_prediction=False, device=None, tokenizer=None, LM_model=None, LM_corpus=None, mimo_extractor_fact=None, mimo_extractor_cond=None, pretrain=False, ensemble=None, ensemble_two=None):

	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_OUTs, raw_instance_list = data
	#SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	MICS = zip(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_OUTs, raw_instance_list)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)

	predict_fact, predict_condition, predict_fact_tuples, predict_cond_tuples = _core_predict(SENTENCEs, POSTAGs, CAPs, instance_list, models, dataCenter, device, tokenizer, LM_model, LM_corpus, mimo_extractor_fact, mimo_extractor_cond, pretrain, ensemble, ensemble_two)

	write_tuples(instance_list, predict_fact_tuples, predict_cond_tuples, dataCenter, './predicts/tuples.txt')

	assert len(predict_fact) == len(SENTENCEs) == len(OUTs)

	return _core_evaluation(OUTs, SENTENCEs, instance_list, predict_fact, predict_condition, predict_fact_tuples, predict_cond_tuples, dataCenter, threshold_fact, threshold_cond, write_prediction)

def Tag_Extractor(SENTENCEs, POSTAGs, CAPs, models, tokenizer, LM_model, LM_corpus, device, ensemble, ensemble_two=None):
	hiddens = [None, None, None]
	for model in models:
		if isinstance(model, (MIMO_LSTM, MIMO_LSTM_TF, MIMO_BERT_LSTM, MIMO_BERT_LSTM_TF)):
			LM_SENTENCEs = []
			LM_hidden = LM_model.init_hidden(len(SENTENCEs))
			lm_data = torch.LongTensor(LM_corpus.tokenize(SENTENCEs))
			lm_data = lm_data.view(len(lm_data[0]), -1)
			output = LM_model(lm_data.to(device), LM_hidden)
			LM_SENTENCEs = output.view(len(SENTENCEs), len(SENTENCEs[0]), -1)

			if not isinstance(model, MIMO_LSTM):
				max_len = len(SENTENCEs[0])
				input_ids = []
				input_mask = []
				for _index in range(len(SENTENCEs)):
					seqword = SENTENCEs[_index]
					new_seqword = []
					for i in range(len(seqword)):
						w = seqword[i]
						assert w == w.lower()
						if w not in tokenizer.vocab:
							new_seqword.append('[UNK]')
						else:
							new_seqword.append(w)
					new_seqword += ['[UNK]']*(max_len-len(new_seqword))
					input_ids.append(tokenizer.convert_tokens_to_ids(new_seqword))
					input_mask.append([1]*len(seqword)+[0]*(max_len-len(seqword)))			

			if isinstance(model, (MIMO_LSTM_TF, MIMO_BERT_LSTM, MIMO_BERT_LSTM_TF)):
				input_ids = torch.LongTensor(input_ids).to(device)
				attention_mask = torch.LongTensor(input_mask).to(device)
			else:
				attention_mask = None
			# (batch_size, multi_nu, seq_len, dim)
			if isinstance(model, (MIMO_BERT_LSTM, MIMO_BERT_LSTM_TF)):
				predict_fact_batch, predict_condition_batch, _, _, hidden_out_batch = model((POSTAGs, CAPs, LM_SENTENCEs), len(SENTENCEs), input_ids, attention_mask=attention_mask)
			else:
				predict_fact_batch, predict_condition_batch, _, _, hidden_out_batch = model((SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs), len(SENTENCEs), attention_mask)
		elif isinstance(model, MIMO_BERT):
			input_ids = []
			input_mask = []
			max_len = len(SENTENCEs[0])
			for seqword in SENTENCEs:
				new_seqword = []
				for w in seqword:
					assert w == w.lower()
					if w not in tokenizer.vocab:
						new_seqword.append('[UNK]')
					else:
						new_seqword.append(w)
				assert len(new_seqword) == len(seqword)
				input_mask.append([1]*len(new_seqword)+[0]*(max_len-len(new_seqword)))
				new_seqword += ['[UNK]']*(max_len-len(new_seqword))
				input_ids.append(tokenizer.convert_tokens_to_ids(new_seqword))
			input_ids = torch.LongTensor(input_ids).to(device)
			input_mask = torch.LongTensor(input_mask).to(device)
			predict_fact_batch, predict_condition_batch, _, _, hidden_out_batch = model(input_ids=input_ids, attention_mask=input_mask)

		if ensemble:
			hidden_index = None
			if True in model.model_LSTM_decoder.lm_config:
				hidden_index = 0
			elif True in model.model_LSTM_decoder.postag_config:
				hidden_index = 1
			else:
				assert True in model.model_LSTM_decoder.cap_config
				hidden_index = 2
			hiddens[hidden_index] = hidden_out_batch
		# model.to(torch.device("cpu"))

	if ensemble != None:
		if len(models) == 2:
			predict_fact_batch, predict_condition_batch, hidden_out_batch = ensemble(hiddens)
		else:
			# ensemble.load_state_dict(torch.load('models/supervised_model_MIMO_BERT_LSTM_111111000_multi_head', map_location=device))
			_, _, hidden_out_batch = ensemble([hiddens[0], hiddens[1], None])
			
			predict_fact_batch, predict_condition_batch, hidden_out_batch = ensemble_two([hidden_out_batch, hiddens[2]])
	else:
		assert len(models) == 1
	
	return predict_fact_batch, predict_condition_batch, hidden_out_batch

def Tuple_Extractor(predict_fact_batch, predict_condition_batch, hidden_out_batch, mimo_extractor_fact, mimo_extractor_cond, dataCenter):
	seq_length = len(predict_fact_batch[0])
	predict_fact_tuples_batch = []
	predict_cond_tuples_batch = []
	for i in range(len(hidden_out_batch)):
		hidden_out = hidden_out_batch[i]
		_, predicted_fact_tags = torch.max(predict_fact_batch[i], 1)
		_, predicted_cond_tags = torch.max(predict_condition_batch[i], 1)
		indexs = torch.arange(seq_length, dtype=torch.long, device=hidden_out.device)
		# tmp = [dataCenter.ID2Tag_fact[tid.item()] for tid in predicted_fact_tags]

		index_ids = (predicted_fact_tags[:seq_length]==dataCenter.Tag2ID_fact['B-f2P']).nonzero().to(hidden_out.device)
		if len(index_ids) != 0:
			position_ids = indexs - index_ids + 100
		else:
			position_ids = indexs.view(1, -1) + 101
		position_ids[position_ids>=300] = 299
		position_ids[position_ids<0] = 0
		predict_fact_tuples_batch.append(mimo_extractor_fact(hidden_out.repeat(len(position_ids), 1, 1), position_ids))


		index_ids = (predicted_cond_tags[:seq_length]==dataCenter.Tag2ID_condition['B-c2P']).nonzero().to(hidden_out.device)
		if len(index_ids) != 0:
			position_ids = indexs - index_ids + 100
		else:
			position_ids = indexs.view(1, -1) + 101
		position_ids[position_ids>=300] = 299
		position_ids[position_ids<0] = 0
		predict_cond_tuples_batch.append(mimo_extractor_cond(hidden_out.repeat(len(position_ids), 1, 1), position_ids))
	# print(predict_fact_tuples_batch.size(), predict_cond_tuples_batch.size())
	assert len(predict_fact_batch) == len(predict_fact_tuples_batch) == len(predict_cond_tuples_batch)
	return predict_fact_tuples_batch, predict_cond_tuples_batch



def apply_model(models, batch_size, dataCenter, device, weight_classes_fact=None, weight_classes_condition=None, LM_model=None, LM_corpus=None, tokenizer=None, mimo_extractor_fact=None, mimo_extractor_cond=None, pretrain=False, ensemble=None, ensemble_two=None, is_semi=False, eval_pack=[]):

	if is_semi:
		out_file, max_f1, max_std, out_model_name, num_pass = eval_pack
	
	if pretrain:
		assert mimo_extractor_fact==None and mimo_extractor_cond==None
	if ensemble != None:
		assert len(models) > 1
	if len(models) > 1:
		assert ensemble != None

	oov = set()

	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_OUTs = dataCenter.get_trainning_data(is_semi)
	SENTENCEs, POSTAGs, CAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_OUTs)

	batches = len(SENTENCEs) // batch_size
	Tag2Metrics = dict()

	params = []
	models_update = []
	if ensemble == None:
		for model in models:
			for param in model.parameters():
				if param.requires_grad:
					params.append(param)
			models_update.append(model)
			# for param in model.parameters():
			# 	if isinstance(model, MIMO_LSTM) and (not pretrain):
			# 		param.requires_grad = False
			# 	else:
			# 		if param.requires_grad:
			# 			params.append(param)
			# if (not isinstance(model, MIMO_LSTM)) or pretrain:
			# 	models_update.append(model)
	else:
		for model in models:
			for param in model.parameters():
				param.requires_grad = False

		if len(models) != 3:
			for param in ensemble.parameters():
				if param.requires_grad:
					params.append(param)
			models_update.append(ensemble)
		else:
			for param in ensemble_two.parameters():
				if param.requires_grad:
					params.append(param)
			models_update.append(ensemble_two)

	if not pretrain:
		for param in mimo_extractor_fact.parameters():
			if param.requires_grad:
				params.append(param)
		for param in mimo_extractor_cond.parameters():
			if param.requires_grad:
				params.append(param)
		models_update.append(mimo_extractor_fact)
		models_update.append(mimo_extractor_cond)

	print('to update params of models: ')
	for model in models_update:
		print(type(model))

	optimizer = optim.SGD(params, lr = 0.01, weight_decay=0.0005, momentum=0.9)

	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size: (index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size: (index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size: (index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size: (index+1)*batch_size]
		
		MICS = zip(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, OUTs_batch)
		MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
		SENTENCEs_batch, POSTAGs_batch, CAPs_batch, OUTs_batch = zip(*MICS)
		SENTENCEs_batch = list(SENTENCEs_batch)
		POSTAGs_batch = list(POSTAGs_batch)
		CAPs_batch = list(CAPs_batch)
		OUTs_batch = list(OUTs_batch)
		
		assert len(SENTENCEs_batch) == len(OUTs_batch)

		optimizer.zero_grad()
		for model in models_update:
			model.zero_grad()

		predict_fact_batch, predict_condition_batch, hidden_out_batch = Tag_Extractor(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, models, tokenizer, LM_model, LM_corpus, device, ensemble, ensemble_two)
		
		if not pretrain:
			predict_fact_tuples_batch, predict_cond_tuples_batch = Tuple_Extractor(predict_fact_batch, predict_condition_batch, hidden_out_batch, mimo_extractor_fact, mimo_extractor_cond, dataCenter)
		else:
			predict_fact_tuples_batch = None
			predict_cond_tuples_batch = None

		assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)

		loss = 0
		nu_instances = 0
		for i in range(len(OUTs_batch)):
			seq_len = len(OUTs_batch[i][0][0])
			fact_nu = len(OUTs_batch[i][0]) if not pretrain else 1
			for k in range(fact_nu):
				if not pretrain:
					if k > len(predict_fact_tuples_batch[i]):
						break
				for j in range(seq_len):
					tagID = OUTs_batch[i][0][k][j]
					if k == 0:
						_loss = (-weight_classes_fact[tagID] * predict_fact_batch[i][j][tagID])
					else:
						_loss = (-weight_classes_fact[tagID] * predict_fact_tuples_batch[i][k-1][j][tagID])
					loss += _loss
					nu_instances += 1
			seq_len = len(OUTs_batch[i][1][0])
			cond_nu = len(OUTs_batch[i][1]) if not pretrain else 1
			for k in range(cond_nu):
				if not pretrain:
					if k > len(predict_cond_tuples_batch[i]):
						break
				for j in range(seq_len):
					tagID = OUTs_batch[i][1][k][j]
					if k == 0:
						_loss = (-weight_classes_condition[tagID] * predict_condition_batch[i][j][tagID])
					else:
						_loss = (-weight_classes_condition[tagID] * predict_cond_tuples_batch[i][k-1][j][tagID])
					loss += _loss
					nu_instances += 1

		loss /= len(OUTs_batch)
		loss.to(device)

		print('batch-%d-aver_loss = %.6f(%d)' % (index, loss.item(), len(SENTENCEs_batch[0])))

		loss.backward()
		for model in models_update:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		# for param in model.parameters():
		#	if param.requires_grad:
		#		print 'param:', param.grad
		optimizer.step()

		if is_semi:
			pre_max_f1 = max_f1
			max_f1, max_std = evaluation(models, out_file, dataCenter, 0, 0, max_f1, max_std, out_model_name, num_pass, False, write_prediction=True, file_name2='./predicts/'+out_file.split('/')[-1], LM_model=LM_model, just_PR=False, LM_corpus=LM_corpus, tokenizer=tokenizer, mimo_extractor_fact=mimo_extractor_fact, mimo_extractor_cond=mimo_extractor_cond, pretrain=pretrain, ensemble=ensemble, models_update=models_update, ensemble_two=ensemble_two)
			if max_f1 == pre_max_f1:
				for model in models:
					print("loading model parameters...")
					model = torch.load(out_model_name+model.name)
					print("loading done.")

	print('oov:', len(oov))

	return models_update

def get_position(VB_index, index):
	min_dis = 999
	position = -1
	if index in VB_index:
		return 0
	for vi in VB_index:
		if math.fabs(index-vi) <= min_dis:
			min_dis = math.fabs(index-vi)
			position = -1 if (index-vi<0) else 1
	return position

def auto_labeling(models, dataCenter, device, data_file, AR, TC, TCDEL, SH, DEL, LM_model=None, LM_corpus=None, tokenizer=None, mimo_extractor_fact=None, mimo_extractor_cond=None, pretrain=False, ensemble=None, ensemble_two=None):
	AR_fact_file_name = './association_rules_fact.txt'
	AR_condition_file_name = './association_rules_condition.txt'
	support_threshold = 3
	confidence_threshold = 0.7

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)

	ar_correcter = AR_Correcter(AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold)

	MICS = zip(dataCenter.ex_TRAIN_SENTENCEs, dataCenter.ex_TRAIN_POSTAGs, dataCenter.ex_TRAIN_CAPs, dataCenter.ex_TRAIN_OUTs, dataCenter.instance_ex_TRAIN)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)
		
	assert len(SENTENCEs) == len(OUTs)
	print(len(SENTENCEs))

	batch_size = 200
	batches = len(SENTENCEs) // batch_size

	tag_outFile = open(data_file, 'w')
	count = 0
	for index in range(batches+1):
		SENTENCEs_batch = SENTENCEs[index*batch_size:(index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size:(index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size:(index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size:(index+1)*batch_size]
		instance_list_batch = instance_list[index*batch_size:(index+1)*batch_size]

		predict_fact_batch, predict_condition_batch, hidden_out_batch = Tag_Extractor(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, models, tokenizer, LM_model, LM_corpus, device, ensemble, ensemble_two)
		
		if not pretrain:
			predict_fact_tuples_batch, predict_cond_tuples_batch = Tuple_Extractor(predict_fact_batch, predict_condition_batch, hidden_out_batch, mimo_extractor_fact, mimo_extractor_cond, dataCenter)
		else:
			predict_fact_tuples_batch = None
			predict_cond_tuples_batch = None

		assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)

		for i in range(len(predict_fact_batch)):
			_, predicted_fact_tags = torch.max(predict_fact_batch[i], 1)
			_, predicted_conditions_tags = torch.max(predict_condition_batch[i], 1)
			assert len(OUTs_batch[i][0][0]) == len(instance_list_batch[i].OUT[0][0])
			fact_tags = []
			cond_tags = []
			if len(OUTs_batch[i][0][0]) > 15:
				continue
			for j in range(len(OUTs_batch[i][0][0])):
				y_predict = predicted_fact_tags[j].item()
				tag = dataCenter.ID2Tag_fact[y_predict]
				fact_tags.append(tag)
				y_predict = predicted_conditions_tags[j].item()
				tag = dataCenter.ID2Tag_condition[y_predict]
				cond_tags.append(tag)

			if AR:
				# print('using AR')
				sentence = SENTENCEs_batch[i]
				postag = POSTAGs_batch[i]
				VB_index = []
				for j in range(len(postag)):
					if postag[j].startswith('VB'):
						VB_index.append(j)
				j = 0
				while j < len(sentence):
					flag = False
					for k in range(len(sentence), j, -1):
						_A = postag[j:k]
						for kk in range(len(_A)):
							if _A[kk] == 'IN':
								_A[kk] += (':'+sentence[j+kk])
							_A[kk] += (':'+str(get_position(VB_index, j+kk)))
						_A = '\t'.join(_A)
						if _A in ar_correcter.A2B_fact:
							tags = ar_correcter.A2B_fact[_A].split('\t')
							flag = True
							fact_tags[j:k] = tags
							j = k
							break
					if not flag:
						j += 1

				IN_index = []
				for j in range(len(postag)):
					if postag[j] == 'IN':
						IN_index.append(j)

				j = 0
				while j < len(sentence):
					flag = False
					for k in range(len(sentence), j, -1):
						_A = postag[j:k]
						for kk in range(len(_A)):
							if _A[kk] == 'IN':
								_A[kk] += (':'+sentence[j+kk])
							_A[kk] += (':'+str(get_position(IN_index, j+kk)))
						_A = '\t'.join(_A)
						if _A in ar_correcter.A2B_cond:
							tags = ar_correcter.A2B_cond[_A].split('\t')
							flag = True
							cond_tags[j:k] = tags
							j = k
							break
					if not flag:
						j += 1

			if TC:
				fact_tags, corrected_fact = smooth_tag_sequence(fact_tags)
				cond_tags, corrected_cond = smooth_tag_sequence(cond_tags)

			if DEL:
				# print('using DEL')
				is_discarded_fact, fact_predicate_set = is_discarded(fact_tags)
				is_discarded_cond, cond_predicate_set = is_discarded(cond_tags)

				if is_discarded_fact or is_discarded_cond:
					continue
				if fact_predicate_set & cond_predicate_set != set():
					continue
			if TCDEL:
				# print('using STDEL')
				fact_tags, corrected_fact = smooth_tag_sequence(fact_tags)
				cond_tags, corrected_cond = smooth_tag_sequence(cond_tags)
				if corrected_fact or corrected_cond:
					continue

			tag_outFile.write('===== '+str(instance_list_batch[i].paper_id)+' stmt'+str(instance_list_batch[i].stmt_id)+' =====\n')
			tag_outFile.write('WORD\t%s\n' % '\t'.join(instance_list_batch[i].SENTENCE))
			tag_outFile.write('POSTAG\t%s\n' % '\t'.join(POSTAGs_batch[i]))
			tag_outFile.write('CAP\t%s\n' % '\t'.join(CAPs_batch[i]))
			tag_outFile.write('f\t%s\n' % '\t'.join(fact_tags))
			tag_outFile.write('c\t%s\n' % '\t'.join(cond_tags))
			count += 1

	tag_outFile.write('#'+str(count)+'\n')
	tag_outFile.close()

	for param in params:
		param.requires_grad = True

