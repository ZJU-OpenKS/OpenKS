"""
Abstract dataset format MTG for Multi-type Knowledge Graph
"""
from typing import List, Dict, Any
from .mmd import MMD

class MTG(MMD):
	"""
	A structure for standard graph data format processed from MDD
	schema: {
		concepts: [
			{
				id: <id>,
				name: <name>,
				type: <entity/relation>,
				parent: <id>, // only for entities
				from: <id>, // only for relations
				to: <id> // only for relations
			}
		]
		attributes: [
			{
				id: <id>,
				name: <name>,
				type: <entity/relation>,
				concept: <id>,
				value: <str/int/float/list/date>
			}
		]
	}
	entities: {
		<concept_id>: {
			pointer: {
				id: <index>,
				<attr>: <index>,
				...
			},
			instances: [
				[<value_1>, <value_2>, ..., <value_n>],
				...
			]
		},
		...
	}
	relations: {
		<concept_id>: {
			pointer: {
				<from>_id: <index>,
				<to>_id: <index>,
				<attr>: <index>,
				...
			},
			instances: [
				[<value_1>, <value_2>, ..., <value_n>],
				...
			]
		},
		...

	}
	"""
	def __init__(
		self,
		name: str = '', 
		schema: Dict[str, List] = {},
		entities: Dict[str, Dict] = {},
		relations: Dict[str, Dict] = {},
		) -> None:
		super(MTG, self).__init__()
		self._name = name
		self._schema = schema
		self._entities = entities
		self._relations = relations

	@property
	def name(self):
		return self._name
	
	@name.setter
	def name(self, name):
		self.structure_check('name', name)
		self._name = name

	@property
	def schema(self):
		return self._schema
	
	@schema.setter
	def schema(self, schema):
		self.structure_check('schema', schema)
		self._schema = schema

	@property
	def entities(self):
		return self._entities
	
	@entities.setter
	def entities(self, entities):
		self.structure_check('entity', entities)
		self._entities = entities

	@property
	def relations(self):
		return self._relations
	
	@relations.setter
	def relations(self, relations):
		self.structure_check('relation', relations)
		self._relations = relations

	def structure_check(self, check_type, value):
		if check_type == 'name':
			if not isinstance(value, str):
				raise TypeError("Graph name must be String type")
		elif check_type == 'schema':
			if 'concepts' not in value or 'attributes' not in value:
				raise KeyError("'concepts' and 'attributes' must be in schema struct")
		elif check_type == 'entity' or check_type == 'relation':
			for item in list(value.values()):
				if 'pointer' not in item or 'instances' not in item:
					raise KeyError("'pointer' and 'instances' must be in entity and relation struct")
		return True
