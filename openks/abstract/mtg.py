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
				axiom: ((from_concept, to_concept), ...) // only for relations
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
				id: <index>, // 2
				<attr>: <index>,
				...
			},
			instances: [
				(<value_1>, <value_1>, <id>, ..., <value_n>),
				...
			]
		},
		...
	}
	relations: {
		<concept_id>: {
			pointer: {
				<from>_id: <index>, // 0
				<to>_id: <index>, // 2
				<attr>: <index>,
				...
			},
			instances: [
				(<from_id>, <value_1>, <to_id>, ..., <value_n>),
				...
			]
		},
		...

	}

	structure design considerations:
	1. Using Tuples for each instance for entity and relation structure, to achieve less space occupation and faster iteration.
	2. Using pointer for attribute position indication so that user do not need to order them before loading data, just keep what it was.
	3. Using schema structure to store concepts and attributes with complex information such as hierarchy for entity types and from/to for relation types.
	4. Supporting multiple and muti-typed attributes for both entities and relations, supporting relation directions for more complex KG.
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
