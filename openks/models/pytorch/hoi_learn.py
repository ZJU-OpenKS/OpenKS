# Copyright (c) 2021 OpenKS Authors, Visual Computing Group, Beihang University.
# All rights reserved.

import logging
import argparse
import os
os.chdir("/data5/zhaolichen/codes/openks-wei")
import datetime
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

from ..model import VisualConstructionModel

from .mmd_modules.det_sgg.maskrcnn_benchmark.config import cfg
from .mmd_modules.det_sgg.relation_predictor.config import sg_cfg
from .mmd_modules.det_sgg.maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from .mmd_modules.det_sgg.maskrcnn_benchmark.data import make_data_loader
from .mmd_modules.det_sgg.maskrcnn_benchmark.solver import make_lr_scheduler
from .mmd_modules.det_sgg.maskrcnn_benchmark.solver import make_optimizer
from .mmd_modules.det_sgg.maskrcnn_benchmark.engine.inference import inference
from .mmd_modules.det_sgg.relation_predictor.relation_predictor import RelationPredictor
from .mmd_modules.det_sgg.relation_predictor.AttrRCNN import AttrRCNN
from .mmd_modules.det_sgg.maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from .mmd_modules.det_sgg.maskrcnn_benchmark.utils.collect_env import collect_env_info
from .mmd_modules.det_sgg.maskrcnn_benchmark.utils.comm import synchronize, get_rank, get_world_size
from .mmd_modules.det_sgg.maskrcnn_benchmark.utils.logger import setup_logger
from .mmd_modules.det_sgg.maskrcnn_benchmark.utils.metric_logger import MetricLogger
from .mmd_modules.det_sgg.maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config


@VisualConstructionModel.register("VisualRelation", "PyTorch")
class VisualRelationTorch(VisualConstructionModel):
    def __init__(self, name: str = 'pytorch-default', use_distributed: bool = False, args = None):
        self.name = name
        self.args = self.parse_args(args)
    
    def parse_args(self, args):
        parser = argparse.ArgumentParser(description="HOI Relation Extraction Model")
        
        args = parser.parse_args()
        return args

    def _load_model(self, output_dir):
        # load checkpoint
        pass

    def _save_model(self, checkpointer: DetectronCheckpointer, out_name: str = '', arguments: dict = {}):
        checkpointer.save(out_name, **arguments)

    def _reduce_loss_dict(self, loss_dict):
        """
        Reduce the loss dictionary from all processes so that process with rank
        0 has the averaged results. Returns a dict with the same fields as
        loss_dict, after reduction.

        """
        pass
    
    def evaluate(self):
        pass
    
    def train(self):
        pass
    
    def run(self, mode="train"):
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.evaluate()
        elif mode == "single":
            raise ValueError("UnImplemented mode!")
    