"""
This module implements a basic parsing procedure for recoginzing entities and relation types from questions
"""

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
		""" To be implemented """
		self.struc_q.entities = entities

	def relation_extract(self) -> None:
		"""
		tag: relation, [type: xxx]
		"""
		relations = []
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

	def link_and_embed(self) -> None:
		"""
		entity linking and transform entities and relations to embeding representations
		"""
		entities = []
		for ent in self.struc_q.entities:
			""" use self.kg_name to link entity and get id """
			""" use self.kg_name and id to search embedding store and get representation """
			""" replace entity with a dictionary with its id and representation """
			""" To be implemented """
			entities.append(ent)
		self.struc_q.entities = entities

	def simple_parser(self) -> StrucQ:
		"""
		1 entity and 1 relation type
		"""
		self.sentence_parse()
		self.entity_extract()
		self.relation_extract()
		self.link_and_embed()
		return self.struc_q


if __name__ == '__main__':
	parser = QuestionManager("Where are you", kg_name="kg1")
	res = parser.simple_parser()
	print(res.text)
	print(res.entities)