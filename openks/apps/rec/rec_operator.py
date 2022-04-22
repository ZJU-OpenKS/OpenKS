# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

from ...abstract.mtg import MTG
from ...abstract.mmd import MMD

class RecOperator(object):

	def __init__(self, iteractions: MMD, graph: MTG):
		self.iteractions = iteractions
		self.graph = graph

	def rec_entity_embed(self, ent_id):
		return NotImplemented

	def rec_user_embed(self, user_id):
		return NotImplemented

	def rec_item_embed(self, item_id):
		return NotImplemented

	def rec_rate(self, user_ids, item_ids):
		return NotImplemented

