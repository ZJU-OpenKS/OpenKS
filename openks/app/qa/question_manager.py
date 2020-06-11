"""
This module implements a basic parsing procedure for recoginzing entities and relation types from questions
"""
import paddle.fluid as fluid

class EntityObj(object):

	def __init__(self):
		self.__name = ''
		self.__ent_type = ''
		self.__ent_id = ''
		self.__embedding = fluid.data(name='entity', shape=[1, None], dtype='float32')

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, name):
		self.__name = name

	@property
	def ent_type(self):
		return self._ent_type
	
	@ent_type.setter
	def ent_type(self, ent_type):
		self.__ent_type = ent_type

	@property
	def ent_id(self):
		return self.__ent_id

	@ent_id.setter
	def ent_id(self, ent_id):
		self.__ent_id = ent_id

	@property
	def embedding(self):
		return self.__embedding
	
	@embedding.setter
	def embedding(self, embedding):
		self.__embedding = embedding

class RelationObj(object):

	def __init__(self):
		self.__rel_type = ''
		self.__rel_id = ''
		self.__embedding = fluid.data(name='relation', shape=[1, None], dtype='float32')
		# rel_id can be relation type id or specific relation id, depends on whether to distinguish relations among different entity pairs

	@property
	def rel_type(self):
		return self._rel_type
	
	@rel_type.setter
	def rel_type(self, rel_type):
		self.__rel_type = rel_type

	@property
	def rel_id(self):
		return self._rel_id
	
	@rel_id.setter
	def rel_id(self, rel_id):
		self.__rel_id = rel_id

	@property
	def embedding(self):
		return self._embedding
	
	@embedding.setter
	def embedding(self, embedding):
		self.__embedding = embedding
		

class StrucQ(object):
	
	def __init__(self, question: str):
		self.__text = question
		self.__parse = None
		self.__entities = []
		self.__relations = []
		self.__target_type = {}
		self.__question_class = {}

	@property
	def text(self):
		return self.__text

	@text.setter
	def text(self, text):
		self.__text = text

	@property
	def parse(self):
		return self.__parse

	@parse.setter
	def parse(self, parse):
		self.__parse = parse

	@property
	def entities(self):
		return self.__entities

	@entities.setter
	def entities(self, entities):
		self.__entities = entities

	@property
	def relations(self):
		return self.__relations

	@relations.setter
	def relations(self, relations):
		self.__relations = relations

	@property
	def target_type(self):
		return self.__target_type

	@target_type.setter
	def target_type(self, target_type):
		self.__target_type = target_type
	
	@property
	def question_class(self):
		return self._question_class

	@question_class.setter
	def question_class(self, question_class):
		self.__question_class = question_class


class QuestionManager(object):
	"""
	question text string  -->  StrucQ
	"""
	def __init__(self, question, kg_name=None):
		self.question = question
		self.struc_q = StrucQ
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
		entities.append(EntityObj)
		""" To be implemented """
		self.struc_q.entities = entities

	def relation_extract(self) -> None:
		"""
		tag: relation, [type: xxx]
		"""
		relations = []
		relations.append(RelationObj)
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

	def simple_parser(self) -> StrucQ:
		"""
		1 entity and 1 relation type
		"""
		self.sentence_parse()
		self.entity_extract()
		self.relation_extract()
		self.entity_link()
		self.entity_relation_embed()
		return self.struc_q


if __name__ == '__main__':
	parser = QuestionManager("Where are you", kg_name="kg1")
	res = parser.simple_parser()
	print(res.text)
	print(res.entities)
	print(res.relations)