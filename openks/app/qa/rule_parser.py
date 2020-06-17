"""
Rule based question parser for multiple domains
"""
import ast
import re
import copy
from .question_manager import QuestionManager, StrucQ
from ...abstract import HDG

class RuleParserCom(QuestionManager):
	"""
	Rules for the domain of investor-company-patent dataset
	"""
	def __init__(self, question: str, graph: HDG) -> None:
		super(RuleParserCom, self).__init__(question)
		self.graph = graph
		self.question_type = {'entity': ['哪些', '哪家', '哪个', '哪几个', '谁'], 'quantity': ['多少', '几个', '几家']}
		self.question_target = {'company': ['公司', '企业'], 'patent': ['专利', '知识产权'], 'investor': ['投资人', '投资机构']}
		self.question_rel = {'company_investor': ['投资了', '被.*投资'], 'company_patent': ['申请了.*专利', '拥有.*专利']}

	def entity_extract(self) -> None:
		entities = []
		index = self.graph.entity_types.index('company')
		attrs = self.graph.entity_attrs[index]['attrs']
		index_alter_names = attrs.index('alter_names')
		index_id = attrs.index('id')
		index_name = attrs.index('name')
		for item in self.graph.entities[index]['instances']:
			tmp = ast.literal_eval(item[index_alter_names])
			tmp.append(item[index_name])
			for name in tmp:
				if self.struc_q.text.find(name) != -1:
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
					if re.search(prefix + p, self.struc_q.text):
						target_type['target_type'] = target
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

	def parse(self) -> StrucQ:
		self.entity_extract()
		self.relation_extract()
		self.target_detect()
		self.question_classify()
		return self.struc_q
