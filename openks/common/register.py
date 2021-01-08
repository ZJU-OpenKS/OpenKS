# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

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
	def register(cls: 'Register', name: str, platform: str):
		def register_module(module: object):
			if platform in cls._registry and name in cls._registry[platform]:
				logger.error("Name conflicts. {} has already been registered as {}.".format(registry[platform][name].__name__, name))
				raise Exception
			else:
				if platform not in cls._registry:
					cls._registry[platform] = {name: module}
				else:
					cls._registry[platform][name] = module
				logger.info("Registration succeds. {} as been registered as {}.".format(module.__name__, name))
				return module
		return register_module

	@classmethod
	def get_module(cls: 'Register', platform: str, name: str) -> object:
		if platform in cls._registry:
			if name in cls._registry[platform]:
				return cls._registry[platform][name]
		else:
			logger.error("Module not found. {} is not a registered name in platform {}.".format(name, platform))

	@classmethod
	def list_modules(cls: 'Register') -> List[str]:
		print("已注册模型：")
		for plat in cls._registry:
			print("框架类型：" + plat)
			print("模型名称：" + str(list(cls._registry[plat].keys())))
		print("-----------------------------------------------")

