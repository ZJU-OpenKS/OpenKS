# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import tensorflow as tf
from .rec_operator import RecOperator
from ...models import *
from ...abstract.mtg import MTG
from ...abstract.mmd import MMD

class IteractionOnlyRec(RecOperator):
	'''
	reference to: https://github.com/xiangwang1223/neural_graph_collaborative_filtering
	'''

	def __init__(self, platform: str, executor: str, model: str):
		self.platform = platform
		self.executor = executor
		self.model = model
		self.saver = tf.train.Saver()
		config = tf.ConfigProto()
		self.sess = tf.Session(config=config)

	def train(self, dataset: MMD):
		args = {
			'lr': 0.0005, 
			'embed_size': 64, 
			'batch_size': 1024, 
			'layer_size': [64,64,64], 
			'regs': [1e-5], 
			'epoch': 200, 
			'node_dropout': [0.1], 
			'mess_dropout': [0.1,0.1,0.1], 
			'ranks': [20, 40, 60, 80, 100],
			'model_dir': './'
		}
		executor = OpenKSModel.get_module(self.platform, self.executor)
		self.model_obj = executor(dataset=dataset, model=OpenKSModel.get_module(self.platform, self.model), args=args)
		self.model_obj.run()

	def rec_user_embed(self, user_ids, item_ids, model_path):
		self.saver.restore(self.sess, model_path)
		rate_batch = self.sess.run(self.model_obj, feed_dict={model.users: user_ids, model.pos_items: item_ids})
		return rate_batch
		

		