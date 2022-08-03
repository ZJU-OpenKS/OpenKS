# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import torch
import torch.nn as nn
from ...model import TorchModel
from .nero_modules import SoftMatch

logger = logging.getLogger(__name__)

@TorchModel.register("relation-classification", "PyTorch")
class RelationClassification(TorchModel):
	def __init__(self, *args):
		super(RelationClassification, self).__init__()
		self.model = SoftMatch(*args)

	def forward(self, *args):
		return self.model(*args)
