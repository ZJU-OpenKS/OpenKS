"""
An abstract class for openks models to be trained with Paddle
"""
import logging
from typing import Tuple, Any
import paddle.fluid as fluid
from paddle.fluid import Variable
from .model_params import ModelParams
from ..common.register import Register
from ..abstract.mmd import MMD

logger = logging.getLogger(__name__)

class KSModel(object):
	"""
	The basic class for models in KS-Studio, defines standard model construction methods
	"""

	def __init__(self, *args) -> None:
		pass

	def model_program(self) -> None:
		""" model network construction for the whole data processing, model designing and training process """
		raise NotImplementedError

	def data_process(self, data: MMD) -> Any:
		""" preprocessing the loaded data for directly train """
		raise NotImplementedError

	def backward(self, *args) -> Any:
		""" model training process to accept loss and return optimized parameters """
		raise NotImplementedError

	def predict(self, *args) -> Any:
		""" to predict instances using the trained model """
		raise NotImplementedError

	def save(self, output_dir: str, *args) -> None:
		""" to serialize the model parameters into a specific directory """
		raise NotImplementedError

	def load(self, from_dir: str, *args) -> "KSModel":
		""" to load serialized parameters as a KSModel object """
		raise NotImplementedError

class KSPaddleModel(KSModel, Register):
	"""
	The basic class for models to be trained using PaddlePaddle. 
	The subclasses of it can be registered and train in standard format.
	"""

	def __init__(
		self, 
		name: str = 'default-name', 
		params: ModelParams = None, 
		data: MMD = None
		) -> None:
		self.name = name
		self.params = params
		self.data = data

	def model_program(self) -> None:
		self.startup_program = fluid.Program()
		self.train_program = fluid.Program()
		self.test_program = fluid.Program()

		with fluid.program_guard(self.train_program, self.startup_program):
			self.train_pos_input = fluid.layers.data(
				"pos_input", 
				dtype="float32", 
				shape=[None, None, None], 
				append_batch_size=False)
			self.train_neg_input = fluid.layers.data(
				"neg_input", 
				dtype="float32", 
				shape=[None, None, None], 
				append_batch_size=False)
			self.train_feed_list = ["pos_input", "neg_input"]
			self.train_feed_vars = [self.train_pos_input, self.train_neg_input]
			self.train_fetch_vars = self.train_construct()
			loss = self.train_fetch_vars[0]
			self.backward(loss, opt=self.params.optimizer['name'])

		with fluid.program_guard(self.test_program, self.startup_program):
			self.test_input = fluid.layers.data(
				"test_input",
				dtype="float32",
                shape=[None],
                append_batch_size=False)
			self.test_feed_list = ["test_input"]
			self.test_fetch_vars = self.test_construct()

	def backward(self, loss: Variable, opt="sgd") -> Tuple:
		optimizer_available = {
			"adam": fluid.optimizer.Adam,
			"sgd": fluid.optimizer.SGD,
			"momentum": fluid.optimizer.Momentum
		}
		if opt in optimizer_available:
			opt_func = optimizer_available[opt]
		else:
			opt_func = None
		if opt_func is None:
			raise ValueError("Unsupported optimizer. You should chose the optimizer in {}.".format(optimizer_available.keys()))
		else:
			optimizer = opt_func(learning_rate=self.params.optimizer['learning_rate'])
			"""
			Should support distributed training here according to the distributed params
			"""
			return optimizer.minimize(loss)

	def train_construct(self):
		raise NotImplementedError("Define a subclass to implement training program")

	def test_construct(self):
		raise NotImplementedError("Define a subclass to implement testing program")
