"""
This module implements a basic parsing procedure for recoginzing entities and relation types from questions
"""
from ...abstract.mtg import MTG
from typing import Dict, List
import numpy as np

class StrucQ(object):
	"""
	For structured question objects which can be utilized in various QA systems from simple to difficult.
	"""
	def __init__(
		self, 
		question: str = '', 
		entities: List = [], 
		relations: List = [], 
		target_type: Dict = {}, 
		question_class: Dict = {},
		q_entity_embed: np.array = np.array([]),
		q_relation_embed: np.array = np.array([]),
		q_embed: np.array = np.array([]),
		neo_sqls: List = []
		) -> None:
		self._text = question
		self._entities = entities
		self._relations = relations
		self._target_type = target_type
		self._question_class = question_class
		self._q_entity_embed = q_entity_embed
		self._q_relation_embed = q_relation_embed,
		self._q_embed = q_embed
		self._neo_sqls = neo_sqls

	@property
	def text(self):
		return self._text

	@text.setter
	def text(self, text):
		self._text = text

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

	@property
	def q_entity_embed(self):
		return self._q_entity_embed

	@q_entity_embed.setter
	def q_entity_embed(self, q_entity_embed):
		self._q_entity_embed = q_entity_embed

	@property
	def q_relation_embed(self):
		return self._q_relation_embed

	@q_relation_embed.setter
	def q_relation_embed(self, q_relation_embed):
		self._q_relation_embed = q_relation_embed

	@property
	def q_embed(self):
		return self._q_embed

	@q_embed.setter
	def q_embed(self, q_embed):
		self._q_embed = q_embed

	@property
	def neo_sqls(self):
		return self._neo_sqls

	@neo_sqls.setter
	def neo_sqls(self, neo_sqls):
		self._neo_sqls = neo_sqls

struc_q = StrucQ()

class QuestionParser(object):
	"""
	question text string  -->  StrucQ
	"""
	def __init__(self, graph: MTG) -> None:
		self.struc_q = struc_q
		self.graph = graph

	def entity_extract(self) -> None:
		""" tag: entity, [name: xxx, type: xxx]
		 To be implemented """
		return NotImplemented

	def relation_extract(self) -> None:
		""" tag: relation, [type: xxx] 
		 To be implemented """
		return NotImplemented

	def target_detect(self) -> None:
		"""
		what the question is asking? Can be a relation or an entity
		class: relation/entity, type: xxx
		 To be implemented """
		return NotImplemented

	def question_classify(self) -> None:
		""" To be implemented """
		return NotImplemented

	def entity_link(self) -> None:
		""" entity linking
		 use self.graph to link entity and get id \
		 replace entity with a dictionary with its id \
		 To be implemented """
		return NotImplemented

	def question_embed(self) -> None:
		""" directly mapping question to the vectorized space the same as graph embeddings """
		return NotImplemented

	def question_entity_embed(self) -> None:
		""" mapping question to the entity vectorized space for getting similar entities in graph """
		return NotImplemented

	def question_relation_embed(self) -> None:
		""" mapping question to the relation vectorized space for getting similar relations in graph """
		return NotImplemented

	def sql_generate(self) -> None:
		return NotImplemented

	def struc_q_format(self) -> None:
		print("-----------------------------------------------")
		print("问题原文：" + struc_q.text)
		print("解析后结构化问题：")
		print("涉及实体：" + str(struc_q.entities))
		print("涉及关系：" + str(struc_q.relations))
		print("目标实体：" + str(struc_q.target_type))
		print("答案类型：" + str(struc_q.question_class))
		print("图谱请求SQL：" + str(struc_q.neo_sqls))
		print("-----------------------------------------------")

	def parse(self, question) -> StrucQ:
		""" actual question parsing function """
		self.struc_q.text = question
		# self.entity_extract()
		# self.relation_extract()
		# self.target_detect()
		# self.question_classify()
		# self.entity_link()
		# self.question_embed()
		# self.struc_q_format()
		return self.struc_q
