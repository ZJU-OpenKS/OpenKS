# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import torch
from .question_parser import QuestionParser, StrucQ
from ...models import *
from ...abstract.mmd import MMD

class ModelParser(QuestionParser):
	"""
	Embedding model based question parser
	"""
	def __init__(self, platform: str, model_type: str, model: str) -> None:
		self.platform = platform
		self.model_type = model_type
		self.model = model

	def train(self, dataset: MMD):
		model_type = OpenKSModel.get_module(self.platform, self.model_type)
		executor = model_type(dataset=dataset, model=OpenKSModel.get_module(self.platform, self.model), args=None)
		executor.run(dist=False)

	def entity_extract(self, model_path: str) -> None:
		model = torch.load(model_path, map_location=lambda storage, loc: storage)
		model.eval()
		entities = model(self.struc_q.text).cpu().data.numpy()
		self.struc_q.entities = entities
		return None

	def question_entity_embed(self, model_path: str) -> None:
		model = torch.load(model_path, map_location=lambda storage, loc: storage)
		model.eval()
		head_emb = model(self.struc_q.text).cpu().data.numpy()
		self.struc_q.q_entity_embed = head_emb
		return None

	def question_relation_embed(self, model_path: str) -> None:
		model = torch.load(model_path, map_location=lambda storage, loc: storage)
		model.eval()
		predicate_emb = model(self.struc_q.text).cpu().data.numpy()
		self.struc_q.q_relation_embed = head_emb
		return None
