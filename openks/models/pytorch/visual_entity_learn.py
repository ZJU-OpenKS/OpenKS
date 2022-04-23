# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from ..model import VisualConstructionModel


@VisualConstructionModel.register("VisualEntity", "PyTorch")
class VisualRelationTorch(VisualConstructionModel):
    def __init__(self, name: str, dataset=None, args=None):
        super().__init__(name=name, dataset=dataset, args=args)
    
    def parse_args(self):
        return super().parse_args()

    def data_reader(self, *args):
        return super().data_reader(*args)

    def evaluate(self, *args):
        return super().evaluate(*args)
    
    def train(self, *args):
        return super().train(*args)