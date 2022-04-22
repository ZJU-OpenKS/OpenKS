# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

"""
Answer fetch progam to receive structured question and get possible answers
"""
import logging
from sklearn.metrics.pairwise import euclidean_distances
from typing import TypeVar
from .question_parser import StrucQ
from ...abstract.mtg import MTG

logger = logging.getLogger(__name__)
T = TypeVar('T')

class AnswerFetcher(object):

	def __init__(self, struc_q: StrucQ) -> None:
		self.struc_q = struc_q

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

	def fetch_by_matching(self, graph: MTG) -> T:
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
		for item in graph.schema:
			if item['type'] == 'relation' and item['concept'] == relation_type:
				if item['members'].index(entity_type) == 0:
					source_rel_col_index = 0
					target_rel_col_index = 2
				else:
					source_rel_col_index = 2
					target_rel_col_index = 0

		target_ids = []
		for rel in graph.triples:
			if rel[0][1] == relation_type:
				if rel[0][source_rel_col_index] == entity_id:
					target_ids.append(rel[0][target_rel_col_index])

		target_items = []
		for ent in graph.entities:
			if ent[1] == target_type:
				for tar_id in target_ids:
					if ent[0] == tar_id:
						target_items.append(ent)
		target_props = [item['properties'] for item in graph.schema if item['type'] == 'entity' and item['concept'] == target_type]
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

	def fetch_by_db_query(self, graph_db) -> T:
		""" fetch the answer through querying outside knowledge databases """
		final_answers = []
		for sql_ in self.struc_q.neo_sqls:
			question_type = sql_['type']
			queries = sql_['sql']
			answers = []
			for query in queries:
				ress = graph_db.run(query).data()
				answers += ress
			final_answers.append(answers)
		return final_answers

	def fetch_by_similarity(self, embeddings) -> T:
		""" 
		fetch the answer through calculating vector similarities 
		refer to paper: Knowledge Graph Embedding Based Question Answering, WSDM 2019

		"""
		if not self.struc_q_embed_check():
			return None

		else:
			# should calculate embedding similarities between question and graph nodes
			# match entity name in MTG graph to filter a smaller set of candidates
			matched_ids = entity_name_match(self.struc_q.entities, self.graph)
			# further calculate distances between question entity embeddings and graph embeddings
			similarities = euclidean_distances(self.struc_q.q_entity_embed * len(matched_ids), embeddings[matched_ids], squared=True).argsort(axis=1)
			# get the shortest distance entity id
			index_top = sort_with_index(similarities)[0]
			# use relation function to calculate the target entity embedding
			target_embed = relation_func(self.struc_q.q_entity_embed[index_top], self.struc_q.q_relation_embed)
			# find the closest target entity id
			closest_target_id = find_closest(target_embed, embeddings)
			# get the target entity object
			answer = [ent for ent in self.graph.entities if ent[0] == closest_target_id][0]
			return answer


def entity_name_match(entities, graph):
	return NotImplemented

def sort_with_index(value_array):
	return NotImplemented

def relation_func(head, relation):
	return NotImplemented

def find_closest(embed, embeddings):
	return NotImplemented


