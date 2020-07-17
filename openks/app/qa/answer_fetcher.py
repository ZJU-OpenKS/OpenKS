"""
Answer fetch progam to receive structured question and get possible answers
"""
import logging
from typing import TypeVar
from .question_parser import StrucQ
from ...abstract.mtg import MTG

logger = logging.getLogger(__name__)
T = TypeVar('T')

class AnswerFetcher(object):

	def __init__(self, struc_q: StrucQ, graph: MTG) -> None:
		self.struc_q = struc_q
		self.graph = graph

	def struc_q_rule_check(self) -> bool:
		if len(self.struc_q.relations) == 0:
			logger.warn("No relation found from the question.")
			return False
		elif len(self.struc_q.entities) == 0:
			logger.warn("No entity found from the question.")
			return False
		else:
			return True

	def struc_q_embed_check(self) -> bool:
		if self.struc_q.q_entity_embed.size == 0 and self.struc_q.q_relation_embed.size == 0 and self.struc_q.q_embed.size == 0:
			logger.warn("No embedding computed from the question.")
			return False
		else:
			return True

	def fetch_by_matching(self) -> T:
		""" fetch the answer through matching MTG knowledge graph dataset """
		if not self.struc_q_rule_check():
			return None

		entity_info = self.struc_q.entities[0]
		relation_type = self.struc_q.relations[0]
		target_type = self.struc_q.target_type['type']
		question_type = self.struc_q.question_class['type']

		entity_id = entity_info['id']
		entity_type = entity_info['type']
		source_rel_col_index = 0
		target_rel_col_index = 0
		for item in self.graph.schema:
			if item['type'] == 'relation' and item['concept'] == relation_type:
				if item['members'].index(entity_type) == 0:
					source_rel_col_index = 0
					target_rel_col_index = 2
				else:
					source_rel_col_index = 2
					target_rel_col_index = 0

		target_ids = []
		for rel in self.graph.triples:
			if rel[0][1] == relation_type:
				if rel[0][source_rel_col_index] == entity_id:
					target_ids.append(rel[0][target_rel_col_index])

		target_items = []
		for ent in self.graph.entities:
			if ent[1] == target_type:
				for tar_id in target_ids:
					if ent[0] == tar_id:
						target_items.append(ent)
		target_props = [item['properties'] for item in self.graph.schema if item['type'] == 'entity' and item['concept'] == target_type]
		target_cols = [item['name'] for item in target_props[0]]
		res = []
		for item in target_items:
			tmp = {}
			for key, value in zip(target_cols, item[2]):
				tmp[key] = value
			res.append(tmp)
		if question_type == 'entity':
			return res
		elif question_type == 'quantity':
			return len(res)

	def fetch_by_db_query(self) -> T:
		""" fetch the answer through querying outside knowledge databases """
		if not self.struc_q_rule_check():
			return None
		else:
			return NotImplemented

	def fetch_by_similarity(self) -> T:
		""" fetch the answer through calculating vector similarities """
		if not self.struc_q_embed_check():
			return None

		else:
			# should calculate embedding similarities between question and graph nodes
			return NotImplemented



