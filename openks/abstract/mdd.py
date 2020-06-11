"""
Abstract dataset format MDD for Multi-modal Distributed Dataset
"""

class MDD(object):

	def __init__(self):
		self._headers = []
		self._body = []
		self._name = ''

	@property
	def headers(self):
		return self._headers
	
	@headers.setter
	def headers(self, headers):
		self._headers = headers

	@property
	def body(self):
		return self._body
	
	@body.setter
	def body(self, body):
		self._body = body

	@property
	def name(self):
		return self._name
	
	@name.setter
	def name(self, name):
		self._name = name
