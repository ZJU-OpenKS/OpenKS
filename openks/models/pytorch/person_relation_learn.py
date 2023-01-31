# Copyright (c) 2021 OpenKS Authors, Visual Computing Group, Beihang University.
# All rights reserved.

import random
import torch
import numpy as np
import os
import argparse

from ..model import VisualConstructionModel

from .mmd_modules.Person_Relation.utils.logger import setup_logger
from .mmd_modules.Person_Relation.datasets import make_dataloader
from .mmd_modules.Person_Relation.model import make_model
from .mmd_modules.Person_Relation.solver import make_optimizer
from .mmd_modules.Person_Relation.solver.scheduler_factory import create_scheduler
from .mmd_modules.Person_Relation.loss import make_loss
from .mmd_modules.Person_Relation.processor import do_train, do_inference

# from timm.scheduler import create_scheduler
from .mmd_modules.Person_Relation.config import cfg


@VisualConstructionModel.register("Person_Relation", "PyTorch")
class VisualRelationTorch(VisualConstructionModel):
    # TODO distributed learning is not complete.
    def __init__(self, name: str = 'pytorch-person_relation', use_distributed: bool = False, args = {"Person_Relation": True}):
        self.name = name
        self.args = self.parse_args(args)
        print("args", self.args)
        print("############", self.args.config_file)

        if self.args.config_file != "":
            cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
        cfg.freeze()

        print("############", cfg.MODEL.DEVICE_ID)
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    
    def parse_args(self, args):
        parser = argparse.ArgumentParser(description="ReID Baseline Training")
        parser.add_argument(
            "--config_file", default="/data5/caidaigang/model/OpenKS-3djcg/openks/models/pytorch/mmd_modules/Person_Relation/configs/MSMT17/vit_transreid_stride.yml", help="path to config file", type=str
        )

        parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                            nargs=argparse.REMAINDER)
        parser.add_argument("--local_rank", default=0, type=int)
        args = parser.parse_args()

        return args

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def evaluate(self, args):
        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger = setup_logger("person_relation", output_dir, if_train=False)
        logger.info(args)

        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
            cfg)

        print("############cfg.TEST.WEIGHT", cfg.TEST.WEIGHT)
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        if cfg.TEST.WEIGHT != "":
            model.load_param(cfg.TEST.WEIGHT)
        else:
            model.load_param('/data5/caidaigang/model/OpenKS-3djcg/examples/logs/msmt17_vit_transreid_stride/transformer_120.pth')

        if cfg.DATASETS.NAMES == 'VehicleID':
            for trial in range(10):
                train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
                    cfg)
                rank_1, rank5 = do_inference(cfg,
                                             model,
                                             val_loader,
                                             num_query)
                if trial == 0:
                    all_rank_1 = rank_1
                    all_rank_5 = rank5
                else:
                    all_rank_1 = all_rank_1 + rank_1
                    all_rank_5 = all_rank_5 + rank5

                logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
            logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum() / 10.0, all_rank_5.sum() / 10.0))
        else:
            do_inference(cfg,
                         model,
                         val_loader,
                         num_query)

    def train(self, args):
        self.set_seed(cfg.SOLVER.SEED)

        if cfg.MODEL.DIST_TRAIN:
            torch.cuda.set_device(args.local_rank)

        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger = setup_logger("person_relation", output_dir, if_train=True)
        logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
        logger.info(args)

        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        if cfg.MODEL.DIST_TRAIN:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
            cfg)

        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

        loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

        optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

        scheduler = create_scheduler(cfg, optimizer)

        do_train(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query, args.local_rank
        )
    
    def run(self, mode="train"):
        if mode == "train":
            self.train(self.args)
        elif mode == "eval":
            self.evaluate(self.args)
        elif mode == "single":
            raise ValueError("UnImplemented mode!")
