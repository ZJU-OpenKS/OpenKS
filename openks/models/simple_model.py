"""
A simple test model to illustrate how to use model registration
"""
from .model import KSPaddleModel

@KSPaddleModel.register("simple-model")
class SimpleModel(KSPaddleModel):

	def __init__(self) -> None:
		pass

	def test_register(self):
		print('yes')


if __name__ == '__main__':
	simple_model = SimpleModel()

	print(KSPaddleModel.list_modules())
