# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

"""
Abstract dataset format MTG for Multi-type Knowledge Graph
"""
from typing import List, Dict, Any
from .mmd import MMD

class MTG(MMD):
	"""
	A structure for standard graph data format processed from MDD
	schema: [
		{
			"type": "entity",
			"concept": <entity_type>,
			"properties": [
				{
					"name": <property_name>,
					"range": <property_value_type> // str, int, float, date, datetime, list ...
				},
			]
			"parent": <parent_entity_type>
		},
		...
		{
			"type": "relation",
			"concept": <relation_type>,
			"properties": [
				{
					"name": <property_name>,
					"range": <property_value_type>
				}
			],
			"members": [<head_entity_type>, <tail_entity_type>]
		},
		...
	]
	entities: [
		( <entity_id>, <entity_type>, (<entity_attr1>, <entity_attr2>, ...) ),
		...
	]
	triples: [
		( ( <head_entity_id>, <relation_type>, <tail_entity_id> ), ( <relation_attr1>, <relation_attr2>, ... ) )
	]

	structure design considerations:
	1. Using Tuples for each instance for entity and relation structure, to achieve less space occupation and faster iteration.
	2. Be able to get entity IDs or triplets conviently for graph only learning.
	3. Using schema structure to store concepts and attributes with complex information such as hierarchy for entity types and head/tail for relation types.
	4. Supporting multiple and muti-typed attributes for both entities and relations, supporting relation directions for more complex KG.
	"""
	def __init__(
		self,
		name: str = '', 
		schema: List = [],
		entities: List = [],
		triples: List = [],
		) -> None:
		super(MTG, self).__init__()
		self._name = name
		self._schema = schema
		self._entities = entities
		self._triples = triples

	@property
	def name(self):
		return self._name
	
	@name.setter
	def name(self, name):
		self._name = name

	@property
	def schema(self):
		return self._schema
	
	@schema.setter
	def schema(self, schema):
		self._schema = schema

	@property
	def entities(self):
		return self._entities
	
	@entities.setter
	def entities(self, entities):
		self._entities = entities

	@property
	def triples(self):
		return self._triples
	
	@triples.setter
	def triples(self, triples):
		self._triples = triples

	def hierarchy_construct(self):
		res = []
		for item in self.schema:
			if item['type'] == 'entity':
				if 'parent' in item:
					flag = 0
					for x in res:
						if item['parent'] in x:
							if x.index(item[parent]) != len(x) - 1:
								y = x[:x.index(item[parent])+1]
								y.append(item['concept'])
								res.append(y)
							else:
								x.append(item['concept'])
							flag = 1
							break
					if flag == 0:
						res.append([item['parent'], item['concept']])
				else:
					flag = 0
					for x in res:
						if item['concept'] in x:
							flag = 1
							break
					if flag == 0:
						res.append([item['concept']])
		return res

	def get_entity_num(self):
		return len(self.entities)

	def get_triple_num(self):
		return len(self.triples)

	def get_relation_num(self):
		return len([item['concept'] for item in self.schema if item['type'] == 'relation'])

	def relation_to_id(self):
		res = {}
		rel_list = [item['concept'] for item in self.schema if item['type'] == 'relation']
		index = 0
		for rel in rel_list:
			res[rel] = index
			index += 1
		return res

	def info_display(self):
		print("\n")
		print("载入MTG知识图谱信息：")
		print("-----------------------------------------------")
		print("图谱名称：" + self.name)
		print("图谱实体类型：" + str([item['concept'] for item in self.schema if item['type'] == 'entity']))
		print("图谱关系类型：" + str([item['concept'] for item in self.schema if item['type'] == 'relation']))
		print("图谱实体属性：" + str([item['properties'] for item in self.schema if item['type'] == 'entity' and 'properties' in item]))
		print("图谱关系属性：" + str([item['properties'] for item in self.schema if item['type'] == 'relation' and 'properties' in item]))
		print("图谱层级关系：" + str(self.hierarchy_construct()))
		print("图谱三元组数量：" + str(len(self.triples)))
		print("图谱实体示例：")
		for i in range(5):
			print(self.entities[i])
		print("图谱三元组示例：")
		for i in range(5):
			print(self.triples[i])
		print("-----------------------------------------------")
