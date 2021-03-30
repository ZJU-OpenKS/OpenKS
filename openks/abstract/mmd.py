# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

"""
Abstract dataset format MDD for Multi-modal Distributed Dataset
"""
from typing import List

class MMD(object):
	"""
	A structure for standard data format read from various sources
	"""
	def __init__(
		self, 
		headers: List = [], 
		bodies: List = [],
		name: str = ''
		) -> None:
		self._headers = headers
		self._bodies = bodies
		self._name = name

	@property
	def headers(self):
		return self._headers
	
	@headers.setter
	def headers(self, headers):
		self._headers = headers

	@property
	def bodies(self):
		return self._bodies
	
	@bodies.setter
	def bodies(self, bodies):
		self._bodies = bodies

	@property
	def name(self):
		return self._name
	
	@name.setter
	def name(self, name):
		self._name = name

	def info_display(self):
		print("\n")
		print("载入MMD数据集信息：")
		print("-----------------------------------------------")
		print("数据集名称：" + self.name)
		print("读入文件数量：" + str(len(self.headers)))
		print("字段名：" + str(self.headers))
		print("数据示例：")
		for data in self.bodies:
			print(data[0])
		print("-----------------------------------------------")
