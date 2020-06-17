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

	def fetch_by_one(self):
		ent_to_fetch = None
		rel_to_check = None
		fetch_type = self.struc_q.question_class['question_class']
		ent_id = self.struc_q.entities[0]['id']
		# get target entity list
		for entity_list in self.graph.entities:
			if entity_list['type'] == self.struc_q.target_type['target_type']:
				ent_to_fetch = entity_list['instances']
		# get relation list as a bridge
		for relation_list in self.graph.relations:
			if relation_list['type'] == self.struc_q.relations[0]:
				rel_to_check = relation_list['instances']

		# get column index for source entity and target entity in relation list 
		source_index = 0
		target_index = 0
		for rel_attr in self.graph.relation_attrs:
			if rel_attr['type'] == self.struc_q.relations[0]:
				if rel_attr['from'] == self.struc_q.entities[0]['type']:
					source_index = rel_attr['attrs'].index(rel_attr['from_attr'])
					target_index = rel_attr['attrs'].index(rel_attr['to_attr'])
				elif rel_attr['to'] == self.struc_q.entities[0]['type']:
					source_index = rel_attr['attrs'].index(rel_attr['to_attr'])
					target_index = rel_attr['attrs'].index(rel_attr['from_attr'])
				else:
					logger.error("Relation type not match")

		# get id for target entity
		target_ids = []
		for rel in rel_to_check:
			if rel[source_index] == self.struc_q.entities[0]['id']:
				target_ids.append(rel[target_index])

		# get target entity record by its id
		target_cols = []
		target_id_index = 0
		for ent_attr in self.graph.entity_attrs:
			if ent_attr['type'] == self.struc_q.target_type['target_type']:
				target_cols = ent_attr['attrs']
				target_id_index = ent_attr['attrs'].index('id')
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


