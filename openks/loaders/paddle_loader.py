"""
Support Paddle data reader for Paddle model training process, MMD/MTG as input, Numpy Array iteratble object as output
"""
from typing import Callable, Iterator
import numpy as np
import paddle
import paddle.fluid as fluid
from ..abstract.mmd import MMD

class PaddleLoader(object):

	def __init__(self, mmd: MMD) -> None:
		self.mmd = mmd
		self.pyreader=fluid.io.PyReader

	def mmd_reader_creator(sample_source: MMD, label_source: MMD = None, size: int) -> Callable:
		def reader() -> Iterator:
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
		reader = self.mmd_reader_creator(self.mmd, 1024)
		batch_reader = paddle.batch(reader(), 128)


class DataFeeder(fluid.DataFeeder):
    def __init__(self, feed_list, place, program=None):
        super(DataFeeder,self).__init__(feed_list,place,program)