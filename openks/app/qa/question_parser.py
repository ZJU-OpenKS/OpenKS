"""
This module implements a basic parsing procedure for recoginzing entities and relation types from questions
"""
import

class StrucQuestion(object):
	pass
	

class QuestionParser(object):
	
	def __init__(self, kg_name: str):
		self.kg_name = kg_name

	def parse_entity(self, question: str) -> dict:
		"""
		tag: entity, name: xxx, type: xxx
		"""
		raise NotImplemented

	def parse_relation(self, question: str) -> dict:
		"""
		tag: relation, type: xxx
		"""
		raise NotImplemented

	def parse(self, question: str) -> dict:
		entity_list = self.parse_entity(question)
		relation_list = self.parse_relation(question)
		return {'ent': entity_list, 'rel': relation_list}

	def kg_link(self, parse_res: dict) -> dict:
		linked_res = {'items': []}
		if parse_res['tag'] == 'entity':
			for item in parse_res['items']:
				item["id"] = kg_store[self.kg_name].search(mode='entity', name=item['name'], type=item['type'])
				linked_res['items'].append(item)
			linked_res['tag'] = 'entity'
		elif parse_res['tag'] == 'relation':
			for item in parse_res['items']:
				item["id"] = kg_store[self.kg_name].search(mode='relation', type=item['type'])
				linked_res['items'].append(item)
			linked_res['tag'] = 'relation'
		return linked_res

if __name__ == '__main__':
	parser = QuestionParser()
	parse_res = parser.parse()
	print(parser_res) 
	print(parser.kg_link(parser_res))