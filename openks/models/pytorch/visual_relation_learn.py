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
from easydict import EasyDict

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
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.MODEL.DEVICE = self.args['MODEL.DEVICE']
        cfg.set_new_allowed(False)
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
        cfg.freeze()

        # Whether use multi gpus for model, initialize the process group.
        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = num_gpus > 1 if use_distributed else False
        if self.distributed:
            torch.cuda.set_device(self.args.local_rank)
            torch.distributed.init_process_group(
                backend=cfg.DISTRIBUTED_BACKEND, init_method="env://"
            )
            synchronize()

        save_dir = ""
        self.logger = setup_logger("visual relation extraction", save_dir, get_rank())
        self.logger.info("Using {} GPUs".format(num_gpus))
        self.logger.info(cfg)

        self.logger.info("Collecting env info (might take some time)")
        self.logger.info("\n" + collect_env_info())

        with open(self.args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            self.logger.info(config_str)
        self.logger.info("Running with config:\n{}".format(cfg))

        output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
        self.logger.info("Saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)

        if cfg.MODEL.META_ARCHITECTURE == "RelationPredictor":
            self.model = RelationPredictor(cfg)
        elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
            self.model = AttrRCNN(cfg)

        print('MODEL DEVICE', cfg.MODEL.DEVICE, flush=True)
        self.device = self.model.to(cfg.MODEL.DEVICE)
        self.optimizer = make_optimizer(cfg, self.model)
        self.scheduler = make_lr_scheduler(cfg, self.optimizer)

    
    def parse_args(self, args):
        args_ = {
            'MODEL.DEVICE': 'cuda',
            'opts': [],
            'ckpt': "",
            'mode': 'relation'
        }

        args_.update(args)
        args = EasyDict(args_)
        if args.mode == 'entity':
            args.config_file = 'openks/models/pytorch/mmd_modules/det_sgg/sgg_configs/vgattr/vinvl_x152c4.yaml'
        elif args.mode  == 'relation':
            args.config_file = 'openks/models/pytorch/mmd_modules/det_sgg/sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml'
        else:
            raise NotImplementedError(args)
        print('Training Use Args', args)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        return args

    def _load_model(self, output_dir):
        # load checkpoint
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=output_dir)
        ckpt = cfg.MODEL.WEIGHT if self.args.ckpt is None else self.args.ckpt
        _ = checkpointer.load(ckpt, use_latest=self.args.ckpt is None)

    def _save_model(self, checkpointer: DetectronCheckpointer, out_name: str = '', arguments: dict = {}):
        checkpointer.save(out_name, **arguments)

    def _reduce_loss_dict(self, loss_dict):
        """
        Reduce the loss dictionary from all processes so that process with rank
        0 has the averaged results. Returns a dict with the same fields as
        loss_dict, after reduction.

        """
        world_size = get_world_size()
        if world_size < 2:
            return loss_dict
        with torch.no_grad():
            loss_names = []
            all_losses = []
            for k in sorted(loss_dict.keys()):
                loss_names.append(k)
                all_losses.append(loss_dict[k])
            all_losses = torch.stack(all_losses, dim=0)
            dist.reduce(all_losses, dst=0)
            if dist.get_rank() == 0:
                # only main process gets accumulated, so only divide by
                # world_size in this case
                all_losses /= world_size
            reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
        return reduced_losses

    @torch.no_grad()
    def evaluate(self):
        self.load_model(cfg.OUTPUT_DIR)
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        output_folders = [None] * len(cfg.DATASETS.TEST)
        dataset_names = cfg.DATASETS.TEST
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=self.distributed)
        labelmap_file = config_dataset_file(cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            inference(
                self.model,
                cfg,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                skip_performance_eval=cfg.TEST.SKIP_PERFORMANCE_EVAL,
                labelmap_file=labelmap_file,
                save_predictions=cfg.TEST.SAVE_PREDICTIONS,
            )
            synchronize()
    
    def train(self):
        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
        else:
            model = self.model
        arguments = {}
        arguments["iteration"] = 0

        output_dir = cfg.OUTPUT_DIR

        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            cfg, model, self.optimizer, self.scheduler, output_dir, save_to_disk
        )
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
        arguments.update(extra_checkpoint_data)
        arguments["iteration"] = 0

        data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=self.distributed,
            start_iter=arguments["iteration"],
        )

        test_period = cfg.SOLVER.TEST_PERIOD
        if test_period > 0:
            data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=self.distributed, is_for_period=True)
        else:
            data_loader_val = None

        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        
        meters = MetricLogger(delimiter="  ")

        # save procedure is intergrated in this function
        # NOTE: Implement a training procedure here to flex the training process.
        logger = logging.getLogger("VisualRelationTraining")
        logger.info("Start training")
        max_iter = len(data_loader)
        start_iter = arguments["iteration"]
        model.train()
        start_training_time = time.time()
        end = time.time()

        iou_types = ("bbox",)

        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        dataset_names = cfg.DATASETS.TEST

        for iteration, data_batch in enumerate(data_loader, start_iter):

            images, targets, image_ids, scales = data_batch[0], data_batch[1], data_batch[2], data_batch[3:]
            
            if any(len(target) < 1 for target in targets):
                logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
                continue
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            images = images.to(self.device)
            # targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets)

            # take care of additional metric besides loss returned from model
            if type(loss_dict) == tuple:
                other_metric = loss_dict[1]
                meters.update_metrics(
                    {'other_metric': other_metric},
                )
                loss_dict = loss_dict[0]
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = self._reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 1 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=self.optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

            if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
                meters_val = MetricLogger(delimiter="  ")
                synchronize()
                _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                    model,
                    # The method changes the segmentation mask format in a data loader,
                    # so every time a new data loader is created:
                    make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                    dataset_name="[Validation]",
                    iou_types=iou_types,
                    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=None,
                )
                synchronize()
                model.train()
                with torch.no_grad():
                    # Should be one image for each GPU:
                    for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                        images_val = images_val.to(self.device)
                        targets_val = [target.to(self.device) for target in targets_val]
                        loss_dict = model(images_val, targets_val)
                        losses = sum(loss for loss in loss_dict.values())
                        loss_dict_reduced = self.reduce_loss_dict(loss_dict)
                        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                        meters_val.update(loss=losses_reduced, **loss_dict_reduced)
                synchronize()
                logger.info(
                    meters_val.delimiter.join(
                        [
                            "[Validation]: ",
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters_val),
                        lr=self.optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

            if iteration == max_iter:
                self.save_model(checkpointer, "model_final", arguments)
            elif iteration % checkpoint_period == 0:
                self.save_model(checkpointer, "model_{:07d}".format(iteration), arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )

        # return model
    
    def run(self, mode="train"):
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.evaluate()
        elif mode == "single":
            raise ValueError("UnImplemented mode!")
    
