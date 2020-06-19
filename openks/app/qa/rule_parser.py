"""
Rule based question parser for multiple domains
"""
import ast
import re
import copy
from .question_manager import QuestionManager, StrucQ
from ...abstract.mtg import MTG

class RuleParserCom(QuestionManager):
	"""
	Rules for the domain of investor-company-patent dataset
	"""
	def __init__(self, question: str, graph: MTG) -> None:
		super(RuleParserCom, self).__init__(question)
		self.graph = graph
		self.question_type = {'entity': ['哪些', '哪家', '哪个', '哪几个', '谁'], 'quantity': ['多少', '几个', '几家']}
		self.question_target = {'company': ['公司', '企业'], 'patent': ['专利', '知识产权'], 'investor': ['投资人', '投资机构']}
		self.question_target_single = {'company': ['谁被.*投资', '谁申请了'], 'investor': ['谁投资了']}
		self.question_rel = {'company_investor': ['投资了', '被.*投资', '.*的投资人', '.*的投资机构'], 'company_patent': ['申请了.*[专利|知识产权]', '拥有.*[专利|知识产权]', '.*的[专利|知识产权]']}
		self.stop_words = ['信息', '信息技术', '智能']

	def entity_extract(self) -> None:
		entities = []
		entity_type = 'company'
		attrs = self.graph.entity_attrs[entity_type]
		index_alter_names = attrs.index('alter_names')
		index_name = attrs.index('name')
		index_id = attrs.index('id')
		for item in self.graph.entities[entity_type]:
			tmp = ast.literal_eval(item[index_alter_names])
			tmp.append(item[index_name])
			for name in tmp:
				if len(name) >= 2 and name not in self.stop_words and self.struc_q.text.find(name) != -1:
					entities.append({'id': item[index_id], 'name': name, 'type': 'company'})
					self.struc_q.entities = entities
					return None

	def relation_extract(self) -> None:
		relations = []
		for rel, pattern in self.question_rel.items():
			for p in pattern:
				if re.search(p, self.struc_q.text):
					relations.append(rel)
					# Only keeps single relation type for now
					self.struc_q.relations = relations
					return None

	def target_detect(self) -> None:
		target_type = {'target_type': ''}
		prefixes = copy.deepcopy(self.question_type['entity'])
		prefixes.extend(self.question_type['quantity'])
		for target, pattern in self.question_target.items():
			for p in pattern:
				for prefix in prefixes:
					if re.search(prefix + p, self.struc_q.text) or re.search(p + "[是|有]" + prefix, self.struc_q.text):
						target_type['target_type'] = target
						self.struc_q.target_type = target_type
						return None
		if not target_type['target_type']:
			for target, pattern in self.question_target_single.items():
				for p in pattern:
					if re.search(p, self.struc_q.text):
						target_type['target_type'] = target
						self.struc_q.target_type = target_type
						return None
		self.struc_q.target_type = target_type
		return None

	def question_classify(self) -> None:
		question_class = {'question_class': ''}
		for q_class, pattern in self.question_type.items():
			for p in pattern:
				if re.search(p, self.struc_q.text):
					question_class['question_class'] = q_class
					self.struc_q.question_class = question_class
					return None
		self.struc_q.target_type = target_type
		return None

	def parse(self) -> StrucQ:
		self.entity_extract()
		self.relation_extract()
		self.target_detect()
		self.question_classify()
		return self.struc_q
