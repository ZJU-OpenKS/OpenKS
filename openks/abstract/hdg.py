"""
Abstract dataset format HDG for Heterogeneous Distributed Graph
"""
from typing import List
from .mdd import MDD

class HDG(MDD):
	"""
	A structure for standard graph data format processed from MDD
	"""
	def __init__(
		self, 
		graph_name: str = '',
		entity_types: List = [], 
		relation_types: List = [], 
		entity_attrs: List = [], 
		relation_attrs: List = [], 
		entities: List = [], 
		relations: List = []
		) -> None:
		super(HDG, self).__init__()
		self._graph_name = graph_name
		self._entity_types = entity_types
		self._relation_types = relation_types
		self._entity_attrs = entity_attrs
		self._relation_attrs = relation_attrs
		self._entities = entities
		self._relations = relations

	@property
	def graph_name(self):
		return self._graph_name
	
	@graph_name.setter
	def graph_name(self, graph_name):
		self._graph_name = graph_name

	@property
	def entity_types(self):
		return self._entity_types
	
	@entity_types.setter
	def entity_types(self, entity_types):
		self._entity_types = entity_types

	@property
	def relation_types(self):
		return self._relation_types
	
	@relation_types.setter
	def relation_types(self, relation_types):
		self._relation_types = relation_types

	@property
	def entity_attrs(self):
		return self._entity_attrs
	
	@entity_attrs.setter
	def entity_attrs(self, entity_attrs):
		self._entity_attrs = entity_attrs

	@property
	def relation_attrs(self):
		return self._relation_attrs
	
	@relation_attrs.setter
	def relation_attrs(self, relation_attrs):
		self._relation_attrs = relation_attrs

	@property
	def entities(self):
		return self._entities
	
	@entities.setter
	def entities(self, entities):
		self._entities = entities

	@property
	def relations(self):
		return self._relations
	
	@relations.setter
	def relations(self, relations):
		self._relations = relations
