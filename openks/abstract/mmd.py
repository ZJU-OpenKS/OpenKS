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
