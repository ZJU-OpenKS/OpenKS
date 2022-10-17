import sys, os, io
import random
import torch
import logging
import argparse

class Instance(object):
	"""docstring for Instance"""
	def __init__(self, paper_id, stmt_id, multi_input, multi_output):
		super(Instance, self).__init__()

		self.paper_id = paper_id
		self.stmt_id = stmt_id
		self.multi_input = multi_input
		self.multi_output = multi_output

		self.SENTENCE = None
		self.POSTAG = None
		self.CAP = None
		self.LM_SENTENCE = None
		self.OUT = None

class DataCenter(object):
	"""docstring for Instance"""
	def __init__(self, train_file, eval_file):
		super(DataCenter, self).__init__()

		self.POS2ID, self.ID2POS = self.getTag2ID('./resources/PosTag2ID.txt')
		self.CAP2ID, self.ID2CAP = self.getTag2ID('./resources/CAPTag2ID.txt')

		self.Tag2ID_fact, self.ID2Tag_fact = self.getTag2ID('./resources/OutTag2ID_fact.txt')
		self.Tag2ID_condition, self.ID2Tag_condition = self.getTag2ID('./resources/OutTag2ID_condition.txt')

		self.Tag2Num = dict()

		self.TRAIN_SENTENCEs = []
		self.TRAIN_POSTAGs = []
		self.TRAIN_CAPs = []
		self.TRAIN_OUTs = []

		self.ex_TRAIN_SENTENCEs = []
		self.ex_TRAIN_POSTAGs = []
		self.ex_TRAIN_CAPs = []
		self.ex_TRAIN_OUTs = []

		self.EVAL_SENTENCEs = []
		self.EVAL_POSTAGs = []
		self.EVAL_CAPs = []
		self.EVAL_OUTs = []

		self.max_outpus = 0

		self.instance_TRAIN = []
		self.instance_EVAL = []
		self.instance_ex_TRAIN = []

		self.paper_id_set_TRAIN = set()
		self.paper_id_set_EVAL = set()
		self.paper_id_set_ex_TRAIN = set()

		self.loading_dataset(train_file, eval_file)

	def getTag2ID(self, fileName):
		tag2ID = dict()
		ID2Tag = dict()
		with open(fileName, 'r') as f:
			for line in f:
				tag, _id = line.strip().split(' ')
				tag2ID[tag] = int(_id)
				ID2Tag[int(_id)] = tag
		return tag2ID, ID2Tag

	def _add_instance(self, dataset_type, paper_id, stmt_id, multi_input, multi_output, attr_tuple):
		if len(multi_input) == 0:
			return

		SENTENCEs, POSTAGs, CAPs, OUTs = attr_tuple
		instance = Instance(paper_id, stmt_id, multi_input, multi_output)
		senLen = len(instance.multi_input[0][-1])
		# print(instance.multi_input[0][-1])
		for _input in instance.multi_input:
			seq_name = _input[0]
			seq = _input[1]
			if seq_name == 'WORD':
				sentence = []
				for word in seq:
					sentence.append(word.lower())
				assert len(sentence) == senLen
				SENTENCEs.append(sentence)
				instance.SENTENCE = seq
				# formatted_anno_file.write('WORD:\t'+'\t'.join(sentence)+'\n')
			elif seq_name == 'POSTAG':
				assert len(seq) == senLen
				POSTAGs.append(seq)
				instance.POSTAG = seq
				# formatted_anno_file.write('POSTAG:\t'+'\t'.join(seq)+'\n')
			else:
				assert len(seq) == senLen
				CAPs.append(seq)
				instance.CAP = seq
				# formatted_anno_file.write('CAP:\t'+'\t'.join(seq)+'\n')

		# print 'OUT:'
		facts_out = [self.Tag2ID_fact['O']] * senLen
		conditions_out = [self.Tag2ID_condition['O']] * senLen

		predicate_fact = dict()
		predicate_cond = dict()
		for _output in instance.multi_output:
			key = _output[0]
			seq = _output[1]
			if key.startswith('f'):
				if 'B-f2P' in seq:
					p_index = seq.index('B-f2P')
					if p_index not in predicate_fact:
						predicate_fact[p_index] = []
					predicate_fact[p_index].append([self.Tag2ID_fact[tag] for tag in seq])
					# print('F_G: ', seq)
				else:
					if -1 not in predicate_fact:
						predicate_fact[-1] = []
					# print('F_G: ', seq)
					predicate_fact[-1].append([self.Tag2ID_fact[tag] for tag in seq])
			else:
				if 'B-c2P' in seq:
					p_index = seq.index('B-c2P')
					if p_index not in predicate_cond:
						predicate_cond[p_index] = []
					# print('C_G: ', seq)
					predicate_cond[p_index].append([self.Tag2ID_condition[tag] for tag in seq])
				else:
					if -1 not in predicate_cond:
						predicate_cond[-1] = []
					# print('C_G: ', seq)
					predicate_cond[-1].append([self.Tag2ID_condition[tag] for tag in seq])

			for index in range(len(seq)):
				tag = seq[index]
				# print tag,
				if key.startswith('f'):
					if tag != 'O':
						facts_out[index] = self.Tag2ID_fact[tag]
				else:
					if tag != 'O':
						conditions_out[index] = self.Tag2ID_condition[tag]
						
		for index in range(len(facts_out)):
			tag_id_fact = facts_out[index]
			tag_id_condition = conditions_out[index]
			if dataset_type == 'TRAIN':
				self.count_tag(self.ID2Tag_fact[tag_id_fact])
				self.count_tag(self.ID2Tag_condition[tag_id_condition])

		multi_facts = [facts_out, ]
		multi_conds = [conditions_out, ]
		# print([self.ID2Tag_fact[tag_id] for tag_id in facts_out])
		# print([self.ID2Tag_condition[tag_id] for tag_id in conditions_out])
		# print(predicate_fact.keys())
		for p_index in sorted(predicate_fact):
			facts_out = [self.Tag2ID_fact['O']] * senLen
			for seq in predicate_fact[p_index]:
				for index in range(len(seq)):
					tag_id = seq[index]
					if tag_id != 0:
						facts_out[index] = tag_id
			# print([self.ID2Tag_fact[tag_id] for tag_id in facts_out])
			multi_facts.append(facts_out)
		for p_index in sorted(predicate_cond):
			conds_out = [self.Tag2ID_condition['O']] * senLen
			for seq in predicate_cond[p_index]:
				for index in range(len(seq)):
					tag_id = seq[index]
					if tag_id != 0:
						conds_out[index] = tag_id
			# print([self.ID2Tag_condition[tag_id] for tag_id in conds_out])
			multi_conds.append(conds_out)
		# print('===========================================================')
		assert len(facts_out) == len(conditions_out) == senLen

		# print(multi_facts)
		# print(multi_conds)
		outs = [multi_facts, multi_conds]
		OUTs.append(outs)
		instance.OUT = outs

		assert len(SENTENCEs) == len(POSTAGs) ==len(CAPs) == len(OUTs)
		if len(SENTENCEs) % 10000 == 0:
			print(len(SENTENCEs), 'done')

		instance_list = getattr(self, 'instance_'+dataset_type)
		instance_list.append(instance)

	def count_tag(self, tag):
		if tag not in self.Tag2Num:
			self.Tag2Num[tag] = 0
		self.Tag2Num[tag] += 1

	def _loading_dataset(self, dataset_type, dataFile):

		SENTENCEs = getattr(self, dataset_type+'_SENTENCEs')
		POSTAGs = getattr(self, dataset_type+'_POSTAGs')
		CAPs = getattr(self, dataset_type+'_CAPs')
		OUTs = getattr(self, dataset_type+'_OUTs')
		instance_list = getattr(self, 'instance_'+dataset_type)

		del SENTENCEs[:]
		del POSTAGs[:]
		del CAPs[:]
		del OUTs[:]
		del instance_list[:]

		attr_tuple = (SENTENCEs, POSTAGs, CAPs, OUTs)

		logging.debug('loading '+dataset_type+' data from '+dataFile)

		paper_id_set = getattr(self, 'paper_id_set_'+dataset_type)
		paper_id = 'none'
		stmt_id = '0'
		multi_input = []
		multi_output = []
		previous = False

		# formatted_anno_file = open(dataFile.replace('.tsv', '_formatted.tsv'), 'w')

		with open(dataFile, 'r') as fd:
			for line in fd:
				if line.startswith('=====') or line.startswith('#'):
					# conclude the previous instance
					if previous:
						self._add_instance(dataset_type, paper_id, stmt_id, multi_input, multi_output, attr_tuple)

					# start a new instance
					if not line.startswith('====='):
						continue
					paper_id = line.strip().split('===== ')[-1].split(' stmt')[0]
					paper_id_set.add(paper_id)
					stmt_id = line.split('stmt')[-1].split(' =====')[0]
					# logging.debug('doing the paper '+paper_id+', stmt '+stmt_id)
					multi_input = []
					multi_output = []
					previous = True
					# formatted_anno_file.write(line)
					continue
				line_list = line.strip('\n').split('\t')
				seq_name = line_list[0]
				seq = line_list[1:]
				if seq_name in ['WORD', 'POSTAG', 'CAP']:
					multi_input.append((seq_name, seq))
				else:
					multi_output.append((seq_name, seq))

		# formatted_anno_file.close()

		instance_list = getattr(self, 'instance_'+dataset_type)

		print(len(SENTENCEs), len(POSTAGs), len(CAPs))

		print('done.')

	# def loading_dataset(self, trainFile, validFile, testFile):
	def loading_dataset(self, trainFile, evalFile, exTrainFile=None):

		if trainFile != None:
			self._loading_dataset('TRAIN', trainFile)

		if evalFile != None:
			self._loading_dataset('EVAL', evalFile)

		if exTrainFile != None:
			self._loading_dataset('ex_TRAIN', exTrainFile)

	def get_evaluation(self, valid_prop, dataset_type='EVAL'):

		VALID_SENTENCEs = []
		VALID_POSTAGs = []
		VALID_CAPs = []
		VALID_OUTs = []
		VALID_instances = []

		TEST_SENTENCEs = []
		TEST_POSTAGs = []
		TEST_CAPs = []
		TEST_OUTs = []
		TEST_instances = []

		SENTENCEs = getattr(self, dataset_type+'_SENTENCEs')
		POSTAGs = getattr(self, dataset_type+'_POSTAGs')
		CAPs = getattr(self, dataset_type+'_CAPs')
		OUTs = getattr(self, dataset_type+'_OUTs')
		instance_list = getattr(self, 'instance_'+dataset_type)

		#print range(len(self.EVAL_SENTENCEs))
		id_list = random.sample(range(len(SENTENCEs)), int(len(SENTENCEs)*valid_prop))
		#print id_list
		for index in range(len(SENTENCEs)):
			if index not in id_list:
				TEST_SENTENCEs.append(SENTENCEs[index])
				TEST_POSTAGs.append(POSTAGs[index])
				TEST_CAPs.append(CAPs[index])
				TEST_OUTs.append(OUTs[index])
				TEST_instances.append(instance_list[index])
			else:
				VALID_SENTENCEs.append(SENTENCEs[index])
				VALID_POSTAGs.append(POSTAGs[index])
				VALID_CAPs.append(CAPs[index])
				VALID_OUTs.append(OUTs[index])
				VALID_instances.append(instance_list[index])

		VALID_DATA = (VALID_SENTENCEs, VALID_POSTAGs, VALID_CAPs, VALID_OUTs, VALID_instances)

		TEST_DATA = (TEST_SENTENCEs, TEST_POSTAGs, TEST_CAPs, TEST_OUTs, TEST_instances)

		return VALID_DATA, TEST_DATA

	def get_trainning_data(self, is_semi=False):
		SENTENCEs = getattr(self, 'TRAIN_SENTENCEs')
		POSTAGs = getattr(self, 'TRAIN_POSTAGs')
		CAPs = getattr(self, 'TRAIN_CAPs')
		OUTs = getattr(self, 'TRAIN_OUTs')
		if is_semi:
			ex_SENTENCEs = getattr(self, 'ex_TRAIN_SENTENCEs')
			ex_POSTAGs = getattr(self, 'ex_TRAIN_POSTAGs')
			ex_CAPs = getattr(self, 'ex_TRAIN_CAPs')
			ex_OUTs = getattr(self, 'ex_TRAIN_OUTs')
			SENTENCEs.extend(ex_SENTENCEs)
			POSTAGs.extend(ex_POSTAGs)
			CAPs.extend(ex_CAPs)
			OUTs.extend(ex_OUTs)
		print(len(SENTENCEs), len(POSTAGs), len(CAPs), len(OUTs))
		return SENTENCEs, POSTAGs, CAPs, OUTs
