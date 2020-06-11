"""
Abstract dataset format MDD for Multi-modal Distributed Dataset
"""

class MDD(object):

	def __init__(self):
		self._headers = []
		self._bodies = []
		self._name = ''

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
