"""
This module implements a basic parsing procedure for recoginzing entities and relation types from questions
"""
import

class QuestionParser(object):
	
	def __init__(self):
		pass

	def parse_entity(self, question: str) -> list:
		raise NotImplemented

	def parse_relation(self, question: str) -> list:
		raise NotImplemented

	def parse(self, question: str) -> dict:
		entity_list = self.parse_entity(question)
		relation_list = self.parse_relation(question)
		return {'ent': entity_list, 'rel': relation_list}

if __name__ == '__main__':
	parser = QuestionParser()
	print(parser.parse()) 