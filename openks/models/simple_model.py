"""
A simple test model to illustrate how to use model registration
"""
from model import KSModel

@KSModel.register("simple-model")
class SimpleModel(KSModel):

	def __init__(self) -> None:
		pass

	def test_register(self):
		print('yes')


if __name__ == '__main__':
	simple_model = SimpleModel()

	print(KSModel.list_modules())
