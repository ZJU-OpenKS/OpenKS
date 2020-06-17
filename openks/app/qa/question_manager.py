"""
This module implements a basic parsing procedure for recoginzing entities and relation types from questions
"""
from typing import Dict, List
import paddle.fluid as fluid
from paddle.fluid import Variable

class EntityObj(object):
	"""
	For entity object standard that used in recoginzed entities from questions
	"""
	def __init__(
		self, 
		name: str = '', 
		ent_type: str = '', 
		ent_id: str = '', 
		embedding: Variable = fluid.data(name='entity', shape=[1, None], dtype='float32')
		) -> None:
		self._name = name
		self._ent_type = ent_type
		self._ent_id = ent_id
		self._embedding = embedding

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, name):
		self._name = name

	@property
	def ent_type(self):
		return self._ent_type
	
	@ent_type.setter
	def ent_type(self, ent_type):
		self._ent_type = ent_type

	@property
	def ent_id(self):
		return self._ent_id

	@ent_id.setter
	def ent_id(self, ent_id):
		self._ent_id = ent_id

	@property
	def embedding(self):
		return self._embedding
	
	@embedding.setter
	def embedding(self, embedding):
		self._embedding = embedding

class RelationObj(object):
	"""
	For relation object standard that used in extracted relations from questions
	"""
	def __init__(
		self, 
		rel_type: str = '', 
		rel_id: str = '', 
		embedding: Variable = fluid.data(name='relation', shape=[1, None], dtype='float32')
		) -> None:
		self._rel_type = rel_type
		self._rel_id = rel_id
		self._embedding = embedding
		# rel_id can be relation type id or specific relation id, depends on whether to distinguish relations among different entity pairs

	@property
	def rel_type(self):
		return self._rel_type
	
	@rel_type.setter
	def rel_type(self, rel_type):
		self._rel_type = rel_type

	@property
	def rel_id(self):
		return self._rel_id
	
	@rel_id.setter
	def rel_id(self, rel_id):
		self._rel_id = rel_id

	@property
	def embedding(self):
		return self._embedding
	
	@embedding.setter
	def embedding(self, embedding):
		self._embedding = embedding
		

class StrucQ(object):
	"""
	For structured question objects which can be utilized in various QA systems from simple to difficult.
	"""
	def __init__(
		self, 
		question: str = '', 
		parse: Dict = {}, 
		entities: List = [], 
		relations: List = [], 
		target_type: Dict = {}, 
		question_class: Dict = {}
		) -> None:
		self._text = question
		self._parse = parse
		self._entities = entities
		self._relations = relations
		self._target_type = target_type
		self._question_class = question_class

	@property
	def text(self):
		return self._text

	@text.setter
	def text(self, text):
		self._text = text

	@property
	def parse(self):
		return self._parse

	@parse.setter
	def parse(self, parse):
		self._parse = parse

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

	@property
	def target_type(self):
		return self._target_type

	@target_type.setter
	def target_type(self, target_type):
		self._target_type = target_type
	
	@property
	def question_class(self):
		return self._question_class

	@question_class.setter
	def question_class(self, question_class):
		self._question_class = question_class

struc_q = StrucQ()
entity_obj = EntityObj()
relation_obj = RelationObj()

class QuestionManager(object):
	"""
	question text string  -->  StrucQ
	"""
	def __init__(self, question, kg_name=None) -> None:
		self.question = question
		self.struc_q = struc_q
		self.struc_q.text = question
		self.kg_name = kg_name

	def sentence_parse(self) -> None:
		parsed_res = None
		""" To be implemented """
		self.struc_q.parse = parsed_res

	def entity_extract(self) -> None:
		"""
		tag: entity, [name: xxx, type: xxx]
		"""
		entities = []
		entities.append(entity_obj)
		""" To be implemented """
		self.struc_q.entities = entities

	def relation_extract(self) -> None:
		"""
		tag: relation, [type: xxx]
		"""
		relations = []
		relations.append(relation_obj)
		""" To be implemented """
		self.struc_q.relations = relations

	def target_detect(self) -> None:
		"""
		what the question is asking? Can be a relation or an entity
		class: relation/entity, type: xxx
		"""
		target_type = {}
		""" To be implemented """
		self.struc_q.target_type = target_type

	def question_classify(self) -> None:
		question_class = {}
		""" To be implemented """
		self.struc_q.question_class = question_class

	def entity_link(self) -> None:
		"""
		entity linking
		"""
		entities = []
		for ent in self.struc_q.entities:
			""" use self.kg_name to link entity and get id """
			""" replace entity with a dictionary with its id """
			""" To be implemented """
			entities.append(ent)
		self.struc_q.entities = entities

	def entity_relation_embed(self) -> None:
		"""
		transform entities and relations to embeding representations
		"""
		entities = []
		for ent in self.struc_q.entities:
			""" use self.kg_name and id to search embedding store and get representation """
			""" replace entity with a dictionary with its representation """
			""" To be implemented """
			entities.append(ent)
		self.struc_q.entities = entities
		relations = []
		for rel in self.struc_q.relations:
			""" To be implemented """
			relations.append(rel)
		self.struc_q.relations = relations

	def parse(self) -> StrucQ:
		"""
		1 entity and 1 relation type
		"""
		# self.sentence_parse()
		#self.entity_extract()
		#self.relation_extract()
		#self.target_detect()
		#self.question_classify()
		# self.entity_link()
		# self.entity_relation_embed()
		return self.struc_q


if __name__ == '__main__':
	parser = QuestionManager("Where are you", kg_name="kg1")
	res = parser.simple_parser()
	print(res.text)
	print(res.entities)
	print(res.relations)