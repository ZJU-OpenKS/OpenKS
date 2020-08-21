import numpy as np
import paddle
import paddle.fluid as fluid
import time
import ast
from ..model import GeneralModel


@GeneralModel.register("general", "Paddle")
class GeneralPaddle(GeneralModel):
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

		self.word_dim = 32
		self.hidden_dim = 512
		self.depth = 8
		self.mix_hidden_lr = 1e-3
		self.hidden_lr = 1e-3

		self.PASS_NUM = 10
		self.BATCH_SIZE = 10

		self.embedding_name = 'emb'

	def db_lstm(self, word):
		word_input = [word]
		emb_layers = [
			fluid.embedding(
				size=[self.word_dict_len, self.args['word_dim']], 
				input=x, 
				param_attr=fluid.ParamAttr(name=self.embedding_name, learning_rate=self.args['hidden_lr'], trainable=True))
			for x in word_input
		]

		hidden_0_layers = [fluid.layers.fc(input=emb, size=self.args['hidden_size'], act='tanh') for emb in emb_layers]

		hidden_0 = fluid.layers.sums(input=hidden_0_layers)

		lstm_0 = fluid.layers.dynamic_lstm(
			input=hidden_0,
			size=self.args['hidden_size'],
			candidate_activation='relu',
			gate_activation='sigmoid',
			cell_activation='sigmoid')

		# stack L-LSTM and R-LSTM with direct edges
		input_tmp = [hidden_0, lstm_0]

		for i in range(1, self.args['depth']):
			mix_hidden = fluid.layers.sums(input=[
				fluid.layers.fc(input=input_tmp[0], size=self.args['hidden_size'], act='tanh'),
				fluid.layers.fc(input=input_tmp[1], size=self.args['hidden_size'], act='tanh')
			])

			lstm = fluid.layers.dynamic_lstm(
				input=mix_hidden,
				size=self.args['hidden_size'],
				candidate_activation='relu',
				gate_activation='sigmoid',
				cell_activation='sigmoid',
				is_reverse=((i % 2) == 1))

			input_tmp = [mix_hidden, lstm]

		feature_out = fluid.layers.sums(input=[
			fluid.layers.fc(input=input_tmp[0], size=self.label_dict_len, act='tanh'),
			fluid.layers.fc(input=input_tmp[1], size=self.label_dict_len, act='tanh')
		])

		return feature_out


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
		# define data layers
		word = fluid.data(name='word_data', shape=[None, 1], dtype='int64', lod_level=1)

		fluid.default_startup_program().random_seed = 90
		fluid.default_main_program().random_seed = 90

		# define network topology
		feature_out = self.db_lstm(word)
		target = fluid.layers.data(name='target', shape=[1], dtype='int64', lod_level=1)
		crf_cost = fluid.layers.linear_chain_crf(
			input=feature_out,
			label=target,
			param_attr=fluid.ParamAttr(name='crfw', learning_rate=self.mix_hidden_lr))

		avg_cost = fluid.layers.mean(crf_cost)

		sgd_optimizer = fluid.optimizer.SGD(
			learning_rate=fluid.layers.exponential_decay(
				learning_rate=0.01,
				decay_steps=100000,
				decay_rate=0.5,
				staircase=True))

		sgd_optimizer.minimize(avg_cost)

		crf_decode = fluid.layers.crf_decoding(
			input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

		train_data = paddle.batch(self.generator_creator(self.dataset.bodies[0]), batch_size=self.args['batch_size'])

		place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

		feeder = fluid.DataFeeder(feed_list=[word, target], place=place)
		exe = fluid.Executor(place)

		def train_loop(main_program):
			exe.run(fluid.default_startup_program())
			embedding_param = fluid.global_scope().find_var(self.embedding_name).get_tensor()

			start_time = time.time()
			batch_id = 0
			for pass_id in range(0, self.args['epoch']):
				for data in train_data():
					cost = exe.run(
						main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
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
								# TODO(liuyiqun): Change the target to crf_decode
								fluid.io.save_inference_model(save_dirname, ['word_data'], [feature_out], exe)
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

