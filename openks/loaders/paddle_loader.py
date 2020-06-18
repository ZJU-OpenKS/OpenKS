"""
Support Paddle data reader for Paddle model training process, MDD/HDG as input, Numpy Array iteratble object as output
"""
import numpy as np
import paddle
from ..abstract.mdd import MDD

class PaddleLoader(object):

	def __init__(self, mdd: MDD) -> None:
		self.mdd = mdd

	def mdd_reader_creator(sample_source: MDD, label_source: MDD = None, size: int) -> function:
		def reader():
			samples = np.array(sample_source.bodies[:size].insert(0, sample_source.headers)).astype('float32')
			if label_source:
				labels = np.array([label[0] for label in label_source.bodies[:size]].insert(0, label_source.headers)).astype('int')
				for i in range(size+1):
					yield samples[i, :], labels[i]
			else:
				for i in range(size+1):
					yield samples[i, :]
		return reader

	def test_batch_reader():
		reader = self.mdd_reader_creator(self.mdd, 1024)
		batch_reader = paddle.batch(reader(), 128)
