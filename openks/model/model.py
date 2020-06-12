"""
An abstract class for openks models to be trained with Paddle
"""
import logging
from model import ModelParams
import paddle.fluid as fluid
import sys
sys.path.append('..')
from common.register import Register

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class KSModel(Register):

	def __init__(
		self, 
		name: str = 'default-name', 
		params: ModelParams = None, 
		data: MDD = None
		) -> None:
		self.name = name
		self.params = params
		self.data = data

	def constructor(self):
		pass

	def optimizer(self):
		pass

	def train_construct(self):
		pass

	def test_construct(self):
		pass
