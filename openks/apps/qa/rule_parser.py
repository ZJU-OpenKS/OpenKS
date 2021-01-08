# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

"""
Rule based question parser for multiple domains
"""
import os
import ast
import re
import copy
import ahocorasick
from .question_parser import QuestionParser, StrucQ
from ...abstract.mtg import MTG

class RuleParserCom(QuestionParser):
	"""
	Rules for the domain of investor-company-patent dataset
	"""
	def __init__(self, graph: MTG) -> None:
		super(RuleParserCom, self).__init__(graph)
		self.question_type = {'entity': ['哪些', '哪家', '哪个', '哪几个', '谁'], 'quantity': ['多少', '几个', '几家']}
		self.question_target = {'company': ['公司', '企业'], 'patent': ['专利', '知识产权'], 'investor': ['投资人', '投资机构']}
		self.question_target_single = {'company': ['谁被.*投资', '谁申请了'], 'investor': ['谁投资了']}
		self.question_rel = {'invests': ['投资了', '被.*投资', '.*的投资人', '.*的投资机构'], 'applies': ['申请了.*[专利|知识产权]', '拥有.*[专利|知识产权]', '.*的[专利|知识产权]']}
		self.stop_words = ['信息', '信息技术', '智能']

	def entity_extract(self) -> None:
		entities = []
		entity_type = 'company'
		props = [item['properties'] for item in self.graph.schema if item['concept'] == entity_type][0]
		index_alter_names = props.index({"name": "alter_names","range": "list"})
		index_name = props.index({"name": "name","range": "str"})
		index_id = 0
		for item in self.graph.entities:
			if item[1] == entity_type:
				tmp = ast.literal_eval(item[2][index_alter_names])
				tmp.append(item[2][index_name])
				for name in tmp:
					if len(name) >= 2 and name not in self.stop_words and self.struc_q.text.find(name) != -1:
						entities.append({'id': item[index_id], 'name': name, 'type': entity_type})
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
		target_type = {'type': ''}
		prefixes = copy.deepcopy(self.question_type['entity'])
		prefixes.extend(self.question_type['quantity'])
		for target, pattern in self.question_target.items():
			for p in pattern:
				for prefix in prefixes:
					if re.search(prefix + p, self.struc_q.text) or re.search(p + "[是|有]" + prefix, self.struc_q.text):
						target_type['type'] = target
						self.struc_q.target_type = target_type
						return None
		if not target_type['type']:
			for target, pattern in self.question_target_single.items():
				for p in pattern:
					if re.search(p, self.struc_q.text):
						target_type['type'] = target
						self.struc_q.target_type = target_type
						return None
		self.struc_q.target_type = target_type
		return None

	def question_classify(self) -> None:
		question_class = {'type': ''}
		for q_class, pattern in self.question_type.items():
			for p in pattern:
				if re.search(p, self.struc_q.text):
					question_class['type'] = q_class
					self.struc_q.question_class = question_class
					return None
		self.struc_q.question_class = question_class
		return None

	def parse(self, question: str) -> StrucQ:
		self.struc_q.text = question
		self.entity_extract()
		self.relation_extract()
		self.target_detect()
		self.question_classify()
		self.struc_q_format()
		return self.struc_q


class RuleParserMedical(QuestionParser):
	"""
	copied and modified from https://github.com/liuhuanyong/QASystemOnMedicalKG
	"""
	def __init__(self, graph: MTG):
		super(RuleParserMedical, self).__init__(graph)
		print('initializing model......')
		# 加载特征词
		self.disease_wds = [item[2][0] for item in self.graph.entities if item[1] == 'diseases']
		self.department_wds= [item[2][0] for item in self.graph.entities if item[1] == 'departments']
		self.check_wds= [item[2][0] for item in self.graph.entities if item[1] == 'checks']
		self.drug_wds= [item[2][0] for item in self.graph.entities if item[1] == 'drugs']
		self.food_wds= [item[2][0] for item in self.graph.entities if item[1] == 'foods']
		self.producer_wds= [item[2][0] for item in self.graph.entities if item[1] == 'producers']
		self.symptom_wds= [item[2][0] for item in self.graph.entities if item[1] == 'symptoms']
		self.region_words = set(self.department_wds + self.disease_wds + self.check_wds + self.drug_wds + self.food_wds + self.producer_wds + self.symptom_wds)
		self.deny_words = ['否', '非', '不', '无', '弗', '勿', '毋', '未', '没', '莫', '没有', '防止', '不再', '不会', '不能', '忌', '禁止', '防止', '难以', '忘记', '忽视', '放弃', '拒绝', '杜绝', '不是', '并未', '并无', '仍未', '难以出现', '切勿', '不要', '不可', '别', '管住', '注意', '小心', '少']
		# 问句疑问词
		self.symptom_qwds = ['症状', '表征', '现象', '症候', '表现']
		self.cause_qwds = ['原因','成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥', '为何', '如何才会', '怎么才会', '会导致', '会造成']
		self.acompany_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现', '伴随发生', '伴随', '共现']
		self.food_qwds = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜' ,'忌口', '补品', '保健品', '食谱', '菜谱', '食用', '食物','补品']
		self.drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']
		self.prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止','躲避','逃避','避开','免得','逃开','避开','避掉','躲开','躲掉','绕开','怎样才能不', '怎么才能不', '咋样才能不','咋才能不', '如何才能不','怎样才不', '怎么才不', '咋样才不','咋才不', '如何才不','怎样才可以不', '怎么才可以不', '咋样才可以不', '咋才可以不', '如何可以不','怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不']
		self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时', '几个小时', '多少年']
		self.cureway_qwds = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法', '咋治', '怎么办', '咋办', '咋治']
		self.cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性', '能治', '可治', '可以治', '可以医']
		self.easyget_qwds = ['易感人群', '容易感染', '易发人群', '什么人', '哪些人', '感染', '染上', '得上']
		self.check_qwds = ['检查', '检查项目', '查出', '检查', '测出', '试出']
		self.belong_qwds = ['属于什么科', '属于', '什么科', '科室']
		self.cure_qwds = ['治疗什么', '治啥', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用', '有何用', '用处', '用途','有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚', '需要', '要']
		# 构造领域actree
		self.actree = ahocorasick.Automaton()
		for index, word in enumerate(list(self.region_words)):
			self.actree.add_word(word, (index, word))
		self.actree.make_automaton()
		# 构建词典
		self.wd_dict = dict()
		for wd in self.region_words:
			self.wd_dict[wd] = []
			if wd in self.disease_wds:
				self.wd_dict[wd].append('disease')
			if wd in self.department_wds:
				self.wd_dict[wd].append('department')
			if wd in self.check_wds:
				self.wd_dict[wd].append('check')
			if wd in self.drug_wds:
				self.wd_dict[wd].append('drug')
			if wd in self.food_wds:
				self.wd_dict[wd].append('food')
			if wd in self.symptom_wds:
				self.wd_dict[wd].append('symptom')
			if wd in self.producer_wds:
				self.wd_dict[wd].append('producer')
		print('model init finished ......')

	def entity_extract(self):
		region_wds = []
		for i in self.actree.iter(self.struc_q.text):
			wd = i[1][1]
			region_wds.append(wd)
		stop_wds = []
		for wd1 in region_wds:
			for wd2 in region_wds:
				if wd1 in wd2 and wd1 != wd2:
					stop_wds.append(wd1)
		final_wds = [i for i in region_wds if i not in stop_wds]
		final_dict = {i: self.wd_dict.get(i) for i in final_wds}
		self.struc_q.entities = final_dict
		return None

	def question_classify(self):
		data = {}
		if not self.struc_q.entities:
			return {}
		data['args'] = self.struc_q.entities
		#收集问句当中所涉及到的实体类型
		types = []
		for type_ in self.struc_q.entities.values():
			types += type_
		question_type = 'others'

		question_types = []

		# 症状
		if self.check_words(self.symptom_qwds, self.struc_q.text) and ('disease' in types):
			question_type = 'disease_symptom'
			question_types.append(question_type)

		if self.check_words(self.symptom_qwds, self.struc_q.text) and ('symptom' in types):
			question_type = 'symptom_disease'
			question_types.append(question_type)

		# 原因
		if self.check_words(self.cause_qwds, self.struc_q.text) and ('disease' in types):
			question_type = 'disease_cause'
			question_types.append(question_type)
		# 并发症
		if self.check_words(self.acompany_qwds, self.struc_q.text) and ('disease' in types):
			question_type = 'disease_acompany'
			question_types.append(question_type)

		# 推荐食品
		if self.check_words(self.food_qwds, self.struc_q.text) and 'disease' in types:
			deny_status = self.check_words(self.deny_words, self.struc_q.text)
			if deny_status:
				question_type = 'disease_not_food'
			else:
				question_type = 'disease_do_food'
			question_types.append(question_type)

		#已知食物找疾病
		if self.check_words(self.food_qwds+self.cure_qwds, self.struc_q.text) and 'food' in types:
			deny_status = self.check_words(self.deny_words, self.struc_q.text)
			if deny_status:
				question_type = 'food_not_disease'
			else:
				question_type = 'food_do_disease'
			question_types.append(question_type)

		# 推荐药品
		if self.check_words(self.drug_qwds, self.struc_q.text) and 'disease' in types:
			question_type = 'disease_drug'
			question_types.append(question_type)

		# 药品治啥病
		if self.check_words(self.cure_qwds, self.struc_q.text) and 'drug' in types:
			question_type = 'drug_disease'
			question_types.append(question_type)

		# 疾病接受检查项目
		if self.check_words(self.check_qwds, self.struc_q.text) and 'disease' in types:
			question_type = 'disease_check'
			question_types.append(question_type)

		# 已知检查项目查相应疾病
		if self.check_words(self.check_qwds+self.cure_qwds, self.struc_q.text) and 'check' in types:
			question_type = 'check_disease'
			question_types.append(question_type)

		#　症状防御
		if self.check_words(self.prevent_qwds, self.struc_q.text) and 'disease' in types:
			question_type = 'disease_prevent'
			question_types.append(question_type)

		# 疾病医疗周期
		if self.check_words(self.lasttime_qwds, self.struc_q.text) and 'disease' in types:
			question_type = 'disease_lasttime'
			question_types.append(question_type)

		# 疾病治疗方式
		if self.check_words(self.cureway_qwds, self.struc_q.text) and 'disease' in types:
			question_type = 'disease_cureway'
			question_types.append(question_type)

		# 疾病治愈可能性
		if self.check_words(self.cureprob_qwds, self.struc_q.text) and 'disease' in types:
			question_type = 'disease_cureprob'
			question_types.append(question_type)

		# 疾病易感染人群
		if self.check_words(self.easyget_qwds, self.struc_q.text) and 'disease' in types :
			question_type = 'disease_easyget'
			question_types.append(question_type)

		# 若没有查到相关的外部查询信息，那么则将该疾病的描述信息返回
		if question_types == [] and 'disease' in types:
			question_types = ['disease_desc']

		# 若没有查到相关的外部查询信息，那么则将该疾病的描述信息返回
		if question_types == [] and 'symptom' in types:
			question_types = ['symptom_disease']

		# 将多个分类结果进行合并处理，组装成一个字典
		data['types'] = question_types
		self.struc_q.question_class = data
		return None

	def relation_extract(self):
		relations = []
		question_types = self.struc_q.question_class['types']
		for question_type in question_types:
			if question_type == 'disease_symptom' or question_type == 'symptom_disease':
				relations.append('has_symptom')
			elif question_type == 'disease_acompany':
				relations.append('acompany_with')
			elif question_type == 'disease_not_food' or question_type == 'food_not_disease':
				relations.append('no_eat')
			elif question_type == 'disease_do_food' or question_type == 'food_do_disease':
				relations.append('do_eat')
				relations.append('recommand_eat')
			elif question_type == 'disease_drug' or question_type == 'drug_disease':
				relations.append('common_drug')
				relations.append('recommand_drug')
			elif question_type == 'disease_check' or question_type == 'check_disease':
				relations.append('need_check')
		self.struc_q.relations = relations
		return None

	def sql_generate(self):
		args = self.struc_q.question_class['args']
		entity_dict = {}
		for arg, types in self.struc_q.entities.items():
			for type in types:
				if type not in entity_dict:
					entity_dict[type] = [arg]
				else:
					entity_dict[type].append(arg)
		question_types = self.struc_q.question_class['types']
		sqls = []
		for question_type in question_types:
			sql_ = {}
			sql_['type'] = question_type
			sql = []
			if question_type == 'disease_symptom':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'symptom_disease':
				sql = self.sql_transfer(question_type, entity_dict.get('symptom'))

			elif question_type == 'disease_cause':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'disease_acompany':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'disease_not_food':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'disease_do_food':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'food_not_disease':
				sql = self.sql_transfer(question_type, entity_dict.get('food'))

			elif question_type == 'food_do_disease':
				sql = self.sql_transfer(question_type, entity_dict.get('food'))

			elif question_type == 'disease_drug':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'drug_disease':
				sql = self.sql_transfer(question_type, entity_dict.get('drug'))

			elif question_type == 'disease_check':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'check_disease':
				sql = self.sql_transfer(question_type, entity_dict.get('check'))

			elif question_type == 'disease_prevent':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'disease_lasttime':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'disease_cureway':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'disease_cureprob':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'disease_easyget':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			elif question_type == 'disease_desc':
				sql = self.sql_transfer(question_type, entity_dict.get('disease'))

			if sql:
				sql_['sql'] = sql
				sqls.append(sql_)
			self.struc_q.neo_sqls = sqls
		return None

	def check_words(self, wds, sent):
		for wd in wds:
			if wd in sent:
				return True
		return False

	def sql_transfer(self, question_type, entities):
		if not entities:
			return []

		# 查询语句
		sql = []
		# 查询疾病的原因
		if question_type == 'disease_cause':
			sql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.cause".format(i) for i in entities]

		# 查询疾病的防御措施
		elif question_type == 'disease_prevent':
			sql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.prevent".format(i) for i in entities]

		# 查询疾病的持续时间
		elif question_type == 'disease_lasttime':
			sql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.cure_lasttime".format(i) for i in entities]

		# 查询疾病的治愈概率
		elif question_type == 'disease_cureprob':
			sql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.cured_prob".format(i) for i in entities]

		# 查询疾病的治疗方式
		elif question_type == 'disease_cureway':
			sql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.cure_way".format(i) for i in entities]

		# 查询疾病的易发人群
		elif question_type == 'disease_easyget':
			sql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.easy_get".format(i) for i in entities]

		# 查询疾病的相关介绍
		elif question_type == 'disease_desc':
			sql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.desc".format(i) for i in entities]

		# 查询疾病有哪些症状
		elif question_type == 'disease_symptom':
			sql = ["MATCH (m:diseases)-[r:has_symptom]->(n:symptoms) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

		# 查询症状会导致哪些疾病
		elif question_type == 'symptom_disease':
			sql = ["MATCH (m:diseases)-[r:has_symptom]->(n:symptoms) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

		# 查询疾病的并发症
		elif question_type == 'disease_acompany':
			sql1 = ["MATCH (m:diseases)-[r:acompany_with]->(n:diseases) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql2 = ["MATCH (m:diseases)-[r:acompany_with]->(n:diseases) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql = sql1 + sql2
		# 查询疾病的忌口
		elif question_type == 'disease_not_food':
			sql = ["MATCH (m:diseases)-[r:no_eat]->(n:foods) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

		# 查询疾病建议吃的东西
		elif question_type == 'disease_do_food':
			sql1 = ["MATCH (m:diseases)-[r:do_eat]->(n:foods) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql2 = ["MATCH (m:diseases)-[r:recommand_eat]->(n:foods) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql = sql1 + sql2

		# 已知忌口查疾病
		elif question_type == 'food_not_disease':
			sql = ["MATCH (m:diseases)-[r:no_eat]->(n:foods) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

		# 已知推荐查疾病
		elif question_type == 'food_do_disease':
			sql1 = ["MATCH (m:diseases)-[r:do_eat]->(n:foods) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql2 = ["MATCH (m:diseases)-[r:recommand_eat]->(n:foods) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql = sql1 + sql2

		# 查询疾病常用药品－药品别名记得扩充
		elif question_type == 'disease_drug':
			sql1 = ["MATCH (m:diseases)-[r:common_drug]->(n:drugs) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql2 = ["MATCH (m:diseases)-[r:recommand_drug]->(n:drugs) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql = sql1 + sql2

		# 已知药品查询能够治疗的疾病
		elif question_type == 'drug_disease':
			sql1 = ["MATCH (m:diseases)-[r:common_drug]->(n:drugs) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql2 = ["MATCH (m:diseases)-[r:recommand_drug]->(n:drugs) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
			sql = sql1 + sql2
		# 查询疾病应该进行的检查
		elif question_type == 'disease_check':
			sql = ["MATCH (m:diseases)-[r:need_check]->(n:checks) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

		# 已知检查查询疾病
		elif question_type == 'check_disease':
			sql = ["MATCH (m:diseases)-[r:need_check]->(n:checks) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

		return sql

	def parse(self, question: str) -> StrucQ:
		self.struc_q.text = question
		self.entity_extract()
		self.question_classify()
		self.relation_extract()
		self.sql_generate()
		self.struc_q_format()
		return self.struc_q
