"""
Answer fetch progam to receive structured question and get possible answers
"""
import logging
from typing import Any
from .question_manager import StrucQ
from ...abstract.mtg import MTG

logger = logging.getLogger(__name__)

class AnswerFetcher(object):

	def __init__(self, struc_q: StrucQ, graph: MTG) -> None:
		self.struc_q = struc_q
		self.graph = graph

	def struc_q_check(self) -> bool:
		if len(self.struc_q.relations) == 0:
			logger.warn("No relation found from the question.")
			return False
		elif len(self.struc_q.entities) == 0:
			logger.warn("No entity found from the question.")
			return False
		else:
			return True 

	def fetch_by_one_hop(self) -> Any:
		if not self.struc_q_check():
			return None

		entity_info = self.struc_q.entities[0]
		relation_type = self.struc_q.relations[0]
		target_type = self.struc_q.target_type['type']
		question_type = self.struc_q.question_class['type']

		entity_id = entity_info['id']
		entity_type = entity_info['type']
		source_rel_col_index = self.graph.relations[relation_type]['pointer'][entity_type + '_id']
		target_rel_col_index = self.graph.relations[relation_type]['pointer'][target_type + '_id']

		target_ids = []
		for rel in self.graph.relations[relation_type]['instances']:
			if rel[source_rel_col_index] == entity_id:
				target_ids.append(rel[target_rel_col_index])

		target_id_index = self.graph.entities[target_type]['pointer']['id']
		target_items = []
		for ent in self.graph.entities[target_type]['instances']:
			for tar_id in target_ids:
				if ent[target_id_index] == tar_id:
					target_items.append(ent)
		target_cols = self.graph.entities[target_type]['pointer'].keys()
		res = []
		for item in target_items:
			tmp = {}
			for key, value in zip(target_cols, item):
				tmp[key] = value
			res.append(tmp)
		if question_type == 'entity':
			return res
		elif question_type == 'quantity':
			return len(res)


