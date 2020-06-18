"""
Answer fetch progam to receive structured question and get possible answers
"""
import logging
from .question_manager import StrucQ
from ...abstract import HDG

logger = logging.getLogger(__name__)

class AnswerFetcher(object):

	def __init__(self, struc_q: StrucQ, graph: HDG) -> None:
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

	def fetch_by_one(self) -> object:
		if not self.struc_q_check():
			return None
		rel_type = self.struc_q.relations[0]
		target_type = self.struc_q.target_type['target_type']
		fetch_type = self.struc_q.question_class['question_class']
		ent_id = self.struc_q.entities[0]['id']
		ent_type = self.struc_q.entities[0]['type']
		ent_to_fetch = self.graph.entities[target_type]
		rel_to_check = self.graph.relations[rel_type]

		# get column index for source entity and target entity ids in relation list 
		source_index = 0
		target_index = 0
		_from = self.graph.relation_attrs[rel_type]['from']
		_to = self.graph.relation_attrs[rel_type]['to']
		attrs = self.graph.relation_attrs[rel_type]['attrs']
		if self.struc_q.entities[0]['type'] in _from:
			source_index = attrs.index(_from[ent_type])
			target_index = attrs.index(list(_to.values())[0])
		elif self.struc_q.entities[0]['type'] in _to:
			source_index = attrs.index(_to[ent_type])
			target_index = attrs.index(list(_from.values())[0])
		else:
			logger.error("Relation type not match")
			return None

		# get id for target entity
		target_ids = []
		for rel in rel_to_check:
			if rel[source_index] == self.struc_q.entities[0]['id']:
				target_ids.append(rel[target_index])

		# get target entity record by its id
		target_cols = self.graph.entity_attrs[target_type]
		target_id_index = self.graph.entity_attrs[target_type].index('id')
		target_items = []
		for ent in ent_to_fetch:
			for target_id in target_ids:
				if ent[target_id_index] == target_id:
					target_items.append(ent)

		# compound to a complete entity as the answer
		res = []
		for item in target_items:
			tmp = {}
			for key, value in zip(target_cols, item):
				tmp[key] = value
			res.append(tmp)
		if fetch_type == 'entity':
			return res
		elif fetch_type == 'quantity':
			return len(res)


