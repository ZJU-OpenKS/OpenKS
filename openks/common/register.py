"""
Any base class inherits this Register class gives its subclasses ability to be registered in a named registry with decoration
"""
from collections import defaultdict
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class Register(object):
	"""
	usage: 
	class BasicClass(Register): 
		...

	@BasicClass.register("sub-class")
	class SubClass(BasicClass):
		def __init__(self, ...):
			...
	Advantages of using registered way is:
	1. For developers, make a standard format for developers to integrate new models or algorithms into the toolkit
	2. For users, provide a simple way to easily access models and configure parameters for model training 
	"""

	_registry: Dict = defaultdict(dict)

	@classmethod
	def register(cls: object, name: str):
		def register_module(module: object):
			if name in cls._registry:
				logging.error("Name conflict. {} has already been registered as {}.".format(name, registry[name].__name__))
				raise Exception
			else:
				cls._registry[name] = module
				logging.info("Registration successfully. {} as been registered as {}.".format(name, module.__name__))
				return module
		return register_module

	@classmethod
	def get_module(cls: object, name: str) -> object:
		if name in cls._registry:
			return cls._registry[name]
		else:
			logging.error("Module not found. {} is not a registered name.".format(name))

	@classmethod
	def list_modules(cls) -> List[str]:
		names = list(cls._registry.keys())
		return names

