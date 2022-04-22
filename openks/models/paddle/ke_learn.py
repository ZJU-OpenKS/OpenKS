# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import numpy as np
import paddle
import paddle.fluid as fluid
import time
import ast
from ..model import KELearnModel


@KELearnModel.register("KELearn", "Paddle")
class KELearnPaddle(KELearnModel):

	def __init__(self, name='paddle-default', dataset=None, model=None, args=None):
		self.name = name
		self.dataset = dataset
		self.args = args
		self.model = model

		train_set = self.dataset.bodies[0]
		valid_set = self.dataset.bodies[1]
		words = set()
		labels = set()
		for sentence in train_set:
			word_list = ast.literal_eval(sentence[0])
			label_list = ast.literal_eval(sentence[1])
			words.update(word_list)
			labels.update(label_list)
		for sentence in valid_set:
			word_list = ast.literal_eval(sentence[0])
			label_list = ast.literal_eval(sentence[1])
			words.update(word_list)
			labels.update(label_list)
		index = 0
		self.word_dict = {}
		for word in words:
			self.word_dict[word] = index
			index += 1
		index = 0
		self.label_dict = {}
		for label in labels:
			self.label_dict[label] = index
			index += 1

		self.word_dict_len = len(self.word_dict)
		self.label_dict_len = len(self.label_dict)


	def generator_creator(self, train_set):

		def data_generator():
			for data in train_set:
				word_list = ast.literal_eval(data[0])
				label_list = ast.literal_eval(data[1])
				word_index_list = []
				label_index_list = []
				for word in word_list:
					word_index_list.append(self.word_dict[word])
				for label in label_list:
					label_index_list.append(self.label_dict[label])
				yield word_index_list, label_index_list
		
		return data_generator


	def train(self, use_cuda, save_dirname=None):

		model = self.model(
			word_dict_len=self.word_dict_len,
			label_dict_len=self.label_dict_len,
			mix_hidden_lr=self.args['mix_hidden_lr'],
			word_dim=self.args['word_dim'],
			hidden_lr=self.args['hidden_lr'],
			hidden_size=self.args['hidden_size'],
			depth=self.args['depth'])

		train_data = paddle.batch(self.generator_creator(self.dataset.bodies[0]), batch_size=self.args['batch_size'])

		place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

		feeder = fluid.DataFeeder(feed_list=[model.word, model.target], place=place)
		exe = fluid.Executor(place)

		def train_loop(main_program):
			exe.run(fluid.default_startup_program())
			embedding_param = fluid.global_scope().find_var('emb').get_tensor()

			start_time = time.time()
			batch_id = 0
			for pass_id in range(0, self.args['epoch']):
				for data in train_data():
					cost = exe.run(
						main_program, feed=feeder.feed(data), fetch_list=[model.loss])
					cost = cost[0]

					if batch_id % 10 == 0:
						print("avg_cost:" + str(cost))
						if batch_id != 0:
							print("second per batch: " + str((
								time.time() - start_time) / batch_id))
						# Set the threshold low to speed up the CI test
						if float(cost) < 40.0:
							print("kpis\ttrain_cost\t%f" % cost)

							if save_dirname is not None:
								fluid.io.save_inference_model(save_dirname, ['word_data'], [model.feature_out], exe)
							return

					batch_id = batch_id + 1

		train_loop(fluid.default_main_program())


	def infer(self, use_cuda, save_dirname=None):
		if save_dirname is None:
			return

		place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
		exe = fluid.Executor(place)

		inference_scope = fluid.core.Scope()
		with fluid.scope_guard(inference_scope):
			[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

			lod = [[3, 4, 2]]
			base_shape = [1]
			word = fluid.create_random_int_lodtensor(lod, base_shape, place, low=0, high=self.word_dict_len - 1)
			assert feed_target_names[0] == 'word_data'

			results = exe.run(
				inference_program,
				feed={feed_target_names[0]: word},
				fetch_list=fetch_targets,
				return_numpy=False)
			print(results[0].lod())
			np_data = np.array(results[0])
			print("Inference Shape: ", np_data.shape)


	def run(self):
		if self.args['gpu'] and not fluid.core.is_compiled_with_cuda():
			return

		# Directory for saving the trained model
		save_dirname = self.args['model_dir']

		self.train(self.args['gpu'], save_dirname)
		self.infer(self.args['gpu'], save_dirname)

