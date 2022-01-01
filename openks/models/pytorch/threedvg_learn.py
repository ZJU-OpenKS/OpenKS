# Copyright (c) 2021 OpenKS Authors, Visual Computing Group, Beihang University.
# All rights reserved.

import os
import sys
import pickle
import argparse
import datetime
import time
import json
from tqdm import tqdm
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
from copy import deepcopy
from plyfile import PlyData, PlyElement

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from ..model import VisualConstructionModel

from .mmd_modules.ThreeDVG.data.scannet.model_util_scannet import ScannetDatasetConfig
from .mmd_modules.ThreeDVG.lib.dataset import ScannetReferenceDataset
from .mmd_modules.ThreeDVG.lib.solver import Solver
from .mmd_modules.ThreeDVG.lib.config import CONF
from .mmd_modules.ThreeDVG.models.refnet import RefNet
from .mmd_modules.ThreeDVG.scripts.utils.AdamW import AdamW
from .mmd_modules.ThreeDVG.scripts.utils.script_utils import set_params_lr_dict
from .mmd_modules.ThreeDVG.lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from .mmd_modules.ThreeDVG.lib.loss_helper import get_loss
from .mmd_modules.ThreeDVG.lib.eval_helper import get_eval
from .mmd_modules.ThreeDVG.utils.pc_utils import write_ply_rgb, write_oriented_bbox
from .mmd_modules.ThreeDVG.utils.box_util import get_3d_box, box3d_iou


SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
SCANNET_ROOT = "/data5/caidaigang/scanrefer/data/scannet/scans/"  #30
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply") # scene_id, scene_id
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt") # scene_id, scene_id

# constants
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DC = ScannetDatasetConfig()

@VisualConstructionModel.register("3DVG", "PyTorch")
class VisualGrounding3DVGTorch(VisualConstructionModel):
    # TODO distributed learning is not complete.
    def __init__(self, name: str = 'pytorch-3dvg', use_distributed: bool = False, args = {"3DVG": True}):
        self.name = name
        self.args = self.parse_args(args)
        print("args",self.args)
        #self.device = torch.device(self.args.device)

        # setting
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # reproducibility
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)
    
    def parse_args(self, args):
        parser = argparse.ArgumentParser(description="3DVG Visual Grounding Model")
        parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
        parser.add_argument("--gpu", type=str, help="gpu", default="2")
        parser.add_argument("--batch_size", type=int, help="batch size", default=10)
        parser.add_argument("--epoch", type=int, help="number of epochs", default=200)
        parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=50)
        parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000)
        parser.add_argument("--lr", type=float, help="learning rate", default=2e-3)
        parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
        parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
        parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
        parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
        parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
        parser.add_argument("--seed", type=int, default=42, help="random seed")
        parser.add_argument("--coslr", action='store_true', help="cosine learning rate")
        parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")
        parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
        parser.add_argument("--no_augment", action="store_true", help="Do NOT use augment on trainingset (not used)")
        parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
        parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
        parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
        parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
        parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
        parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
        parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
        parser.add_argument("--use_pretrained", type=str,
                            help="Specify the folder name containing the pretrained detection module.")
        parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")

        ###############################val##############################
        parser.add_argument("--folder", type=str, help="Folder containing the model")
        parser.add_argument("--force", action="store_true", help="enforce the generation of results")
        parser.add_argument("--no_nms", action="store_true",
                            help="do NOT use non-maximum suppression for post-processing.")
        parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
        parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
        parser.add_argument("--use_oracle", action="store_true", help="Use ground truth bounding boxes.")
        parser.add_argument("--use_cat_rand", action="store_true",
                            help="Use randomly selected bounding boxes from correct categories as outputs.")
        parser.add_argument("--use_best", action="store_true", help="Use best bounding boxes as outputs.")
        parser.add_argument("--reference", action="store_true", help="evaluate the reference localization results")
        parser.add_argument("--detection", action="store_true", help="evaluate the object detection results")

        ###############################visualize##############################
        parser.add_argument("--scene_id", type=str, help="scene id", default="")

        args = parser.parse_args(args)

        return args

    def get_dataloader(self, args, scanrefer, scanrefer_new, all_scene_list, split, config, augment, shuffle=True):
        dataset = ScannetReferenceDataset(
            scanrefer=scanrefer[split],
            scanrefer_new=scanrefer_new[split],
            scanrefer_all_scene=all_scene_list,
            split=split,
            num_points=args.num_points,
            use_height=(not args.no_height),
            use_color=args.use_color,
            use_normal=args.use_normal,
            use_multiview=args.use_multiview,
            lang_num_max=args.lang_num_max,
            augment=augment,
            shuffle=shuffle
        )
        # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4)

        return dataset, dataloader

    def get_eval_dataloader(self, args, scanrefer, scanrefer_new, all_scene_list, split, config):
        dataset = ScannetReferenceDataset(
            scanrefer=scanrefer,
            scanrefer_new=scanrefer_new,
            scanrefer_all_scene=all_scene_list,
            split=split,
            num_points=args.num_points,
            use_color=args.use_color,
            use_height=(not args.no_height),
            use_normal=args.use_normal,
            use_multiview=args.use_multiview,
            lang_num_max=1,#args.lang_num_max
        )
        print("evaluate on {} samples".format(len(dataset)))

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        return dataset, dataloader

    def get_model(self, args):
        # initiate model
        input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
            not args.no_height)
        model = RefNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            num_proposal=args.num_proposals,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir,
            no_reference=args.no_reference,
            dataset_config=DC
        )

        # trainable model
        if args.use_pretrained:
            # load model
            print("loading pretrained VoteNet...")
            pretrained_model = RefNet(
                num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=args.num_proposals,
                input_feature_dim=input_channels,
                use_bidir=args.use_bidir,
                no_reference=True,
                dataset_config=DC
            )

            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
            pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

            # mount
            model.backbone_net = pretrained_model.backbone_net
            model.vgen = pretrained_model.vgen
            model.proposal = pretrained_model.proposal

            if args.no_detection:
                # freeze pointnet++ backbone
                for param in model.backbone_net.parameters():
                    param.requires_grad = False

                # freeze voting
                for param in model.vgen.parameters():
                    param.requires_grad = False

                # freeze detector
                for param in model.proposal.parameters():
                    param.requires_grad = False

        # to CUDA
        model = model.cuda()

        return model

    def get_eval_model(self, args, config):
        # load model
        input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
            not args.no_height)
        model = RefNet(
            num_class=config.num_class,
            num_heading_bin=config.num_heading_bin,
            num_size_cluster=config.num_size_cluster,
            mean_size_arr=config.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir,
            dataset_config=config,
        ).cuda()

        model_name = "model_last.pth" if args.detection else "model.pth"
        path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
        model.load_state_dict(torch.load(path), strict=False)
        model.eval()

        return model

    def get_num_params(self, model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

        return num_params

    def get_solver(self, args, dataloader):
        model = self.get_model(args)
        # TODO
        weight_dict = {
            'detr': {'lr': 0.0001},
            'lang': {'lr': 0.0005},
            'match': {'lr': 0.0005},
        }
        params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
        # params = model.parameters()
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

        if args.use_checkpoint:
            print("loading checkpoint {}...".format(args.use_checkpoint))
            stamp = args.use_checkpoint
            root = os.path.join(CONF.PATH.OUTPUT, stamp)
            checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if args.tag: stamp += "_" + args.tag.upper()
            root = os.path.join(CONF.PATH.OUTPUT, stamp)
            os.makedirs(root, exist_ok=True)

        # scheduler parameters for training solely the detection pipeline
        LR_DECAY_STEP = [80, 120, 160] if args.no_reference else None
        if args.coslr:
            LR_DECAY_STEP = {
                'type': 'cosine',
                'T_max': args.epoch,
                'eta_min': 1e-5,
            }
        LR_DECAY_RATE = 0.1 if args.no_reference else None
        BN_DECAY_STEP = 20 if args.no_reference else None
        BN_DECAY_RATE = 0.5 if args.no_reference else None

        print('LR&BN_DECAY', LR_DECAY_STEP, LR_DECAY_RATE, BN_DECAY_STEP, BN_DECAY_RATE, flush=True)
        solver = Solver(
            model=model,
            config=DC,
            dataloader=dataloader,
            optimizer=optimizer,
            stamp=stamp,
            val_step=args.val_step,
            detection=not args.no_detection,
            reference=not args.no_reference,
            use_lang_classifier=not args.no_lang_cls,
            lr_decay_step=LR_DECAY_STEP,
            lr_decay_rate=LR_DECAY_RATE,
            bn_decay_step=BN_DECAY_STEP,
            bn_decay_rate=BN_DECAY_RATE
        )
        num_params = self.get_num_params(model)

        return solver, num_params, root

    def save_info(self, args, root, num_params, train_dataset, val_dataset):
        info = {}
        for key, value in vars(args).items():
            info[key] = value

        info["num_train"] = len(train_dataset)
        info["num_val"] = len(val_dataset)
        info["num_train_scenes"] = len(train_dataset.scene_list)
        info["num_val_scenes"] = len(val_dataset.scene_list)
        info["num_params"] = num_params

        with open(os.path.join(root, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

    def get_scannet_scene_list(self, split):
        scene_list = sorted(
            [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

        return scene_list

    def get_scanrefer(self, scanrefer_train, scanrefer_val, num_scenes, lang_num_max):
        if self.args.no_reference:
            train_scene_list = self.get_scannet_scene_list("train")
            new_scanrefer_train = []
            for scene_id in train_scene_list:
                data = deepcopy(SCANREFER_TRAIN[0])
                data["scene_id"] = scene_id
                new_scanrefer_train.append(data)

            val_scene_list = self.get_scannet_scene_list("val")
            new_scanrefer_val = []
            for scene_id in val_scene_list:
                data = deepcopy(SCANREFER_VAL[0])
                data["scene_id"] = scene_id
                new_scanrefer_val.append(data)
        else:
            # get initial scene list
            train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
            val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
            if num_scenes == -1:
                num_scenes = len(train_scene_list)
            else:
                assert len(train_scene_list) >= num_scenes

            # slice train_scene_list
            train_scene_list = train_scene_list[:num_scenes]

            # filter data in chosen scenes
            new_scanrefer_train = []
            scanrefer_train_new = []
            scanrefer_train_new_scene = []
            scene_id = ""
            for data in scanrefer_train:
                if data["scene_id"] in train_scene_list:
                    new_scanrefer_train.append(data)
                    if scene_id != data["scene_id"]:
                        scene_id = data["scene_id"]
                        if len(scanrefer_train_new_scene) > 0:
                            scanrefer_train_new.append(scanrefer_train_new_scene)
                        scanrefer_train_new_scene = []
                    if len(scanrefer_train_new_scene) >= lang_num_max:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                        scanrefer_train_new_scene = []
                    scanrefer_train_new_scene.append(data)
                    """
                    if data["scene_id"] not in scanrefer_train_new:
                        scanrefer_train_new[data["scene_id"]] = []
                    scanrefer_train_new[data["scene_id"]].append(data)
                    """
            scanrefer_train_new.append(scanrefer_train_new_scene)

            new_scanrefer_val = scanrefer_val
            scanrefer_val_new = []
            scanrefer_val_new_scene = []
            scene_id = ""
            for data in scanrefer_val:
                # if data["scene_id"] not in scanrefer_val_new:
                # scanrefer_val_new[data["scene_id"]] = []
                # scanrefer_val_new[data["scene_id"]].append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_val_new_scene) > 0:
                        scanrefer_val_new.append(scanrefer_val_new_scene)
                    scanrefer_val_new_scene = []
                if len(scanrefer_val_new_scene) >= lang_num_max:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                    scanrefer_val_new_scene = []
                scanrefer_val_new_scene.append(data)
            scanrefer_val_new.append(scanrefer_val_new_scene)

        print("scanrefer_train_new", len(scanrefer_train_new), len(scanrefer_val_new), len(scanrefer_train_new[0]))
        sum = 0
        for i in range(len(scanrefer_train_new)):
            sum += len(scanrefer_train_new[i])
            # print(len(scanrefer_train_new[i]))
        # for i in range(len(scanrefer_val_new)):
        #    print(len(scanrefer_val_new[i]))
        print("sum", sum)  # 1418 363
        # all scanrefer scene
        all_scene_list = train_scene_list + val_scene_list

        print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

        return new_scanrefer_train, new_scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new

    def get_eval_scanrefer(self, args):
        if args.detection:
            scene_list = self.get_scannet_scene_list("val")
            scanrefer = []
            for scene_id in scene_list:
                data = deepcopy(SCANREFER_TRAIN[0])
                data["scene_id"] = scene_id
                scanrefer.append(data)
        else:
            scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
            scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
            if args.num_scenes != -1:
                scene_list = scene_list[:args.num_scenes]

            scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

            new_scanrefer_val = scanrefer
            scanrefer_val_new = []
            scanrefer_val_new_scene = []
            scene_id = ""
            for data in scanrefer:
                # if data["scene_id"] not in scanrefer_val_new:
                # scanrefer_val_new[data["scene_id"]] = []
                # scanrefer_val_new[data["scene_id"]].append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_val_new_scene) > 0:
                        scanrefer_val_new.append(scanrefer_val_new_scene)
                    scanrefer_val_new_scene = []
                if len(scanrefer_val_new_scene) >= args.lang_num_max:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                    scanrefer_val_new_scene = []
                scanrefer_val_new_scene.append(data)
            if len(scanrefer_val_new_scene) > 0:
                scanrefer_val_new.append(scanrefer_val_new_scene)

        return scanrefer, scene_list, scanrefer_val_new

    def get_visualize_scanrefer(self, args):
        scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
        all_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
        if args.scene_id:
            assert args.scene_id in all_scene_list, "The scene_id is not found"
            scene_list = [args.scene_id]
        else:
            scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))

        scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

        new_scanrefer = []
        scanrefer_new = []
        scanrefer_new_scene = []
        scene_id = ""
        for data in scanrefer:
            new_scanrefer.append(data)
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_new_scene) > 0:
                    scanrefer_new.append(scanrefer_new_scene)
                scanrefer_new_scene = []
            if len(scanrefer_new_scene) >= 1:
                scanrefer_new.append(scanrefer_new_scene)
                scanrefer_new_scene = []
            scanrefer_new_scene.append(data)
        scanrefer_new.append(scanrefer_new_scene)
        return scanrefer, scene_list, scanrefer_new

    def eval_ref(self, args):
        print("evaluate localization...")
        # constant
        DC = ScannetDatasetConfig()

        # init training dataset
        print("preparing data...")
        scanrefer, scene_list, scanrefer_val_new = self.get_eval_scanrefer(args)

        # dataloader
        # _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)
        _, dataloader = self.get_eval_dataloader(args, scanrefer, scanrefer_val_new, scene_list, "val", DC)

        # model
        model = self.get_eval_model(args, DC)

        # config
        POST_DICT = {
            "remove_empty_box": True,
            "use_3d_nms": True,
            "nms_iou": 0.25,
            "use_old_type_nms": False,
            "cls_nms": True,
            "per_class_proposal": True,
            "conf_thresh": 0.05,
            "dataset_config": DC
        } if not args.no_nms else None

        # random seeds
        seeds = [args.seed] + [2 * i for i in range(args.repeat - 1)]

        # evaluate
        print("evaluating...")
        score_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "scores.p")
        pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "predictions.p")
        gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
        if gen_flag:
            ref_acc_all = []
            ious_all = []
            masks_all = []
            others_all = []
            lang_acc_all = []
            for seed in seeds:
                # reproducibility
                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                np.random.seed(seed)

                print("generating the scores for seed {}...".format(seed))
                ref_acc = []
                ious = []
                masks = []
                others = []
                lang_acc = []
                predictions = {}
                for data in tqdm(dataloader):
                    for key in data:
                        data[key] = data[key].cuda()

                    # feed
                    with torch.no_grad():
                        data = model(data)
                        _, data = get_loss(
                            data_dict=data,
                            config=DC,
                            detection=True,
                            reference=True,
                            use_lang_classifier=not args.no_lang_cls
                        )
                        data = get_eval(
                            data_dict=data,
                            config=DC,
                            reference=True,
                            use_lang_classifier=not args.no_lang_cls,
                            use_oracle=args.use_oracle,
                            use_cat_rand=args.use_cat_rand,
                            use_best=args.use_best,
                            post_processing=POST_DICT
                        )

                        ref_acc += data["ref_acc"]
                        ious += data["ref_iou"]
                        masks += data["ref_multiple_mask"]
                        others += data["ref_others_mask"]
                        lang_acc.append(data["lang_acc"].item())

                        # store predictions
                        ids = data["scan_idx"].detach().cpu().numpy()
                        for i in range(ids.shape[0]):
                            idx = ids[i]
                            scene_id = scanrefer[idx]["scene_id"]
                            object_id = scanrefer[idx]["object_id"]
                            ann_id = scanrefer[idx]["ann_id"]

                            if scene_id not in predictions:
                                predictions[scene_id] = {}

                            if object_id not in predictions[scene_id]:
                                predictions[scene_id][object_id] = {}

                            if ann_id not in predictions[scene_id][object_id]:
                                predictions[scene_id][object_id][ann_id] = {}

                            predictions[scene_id][object_id][ann_id]["pred_bbox"] = data["pred_bboxes"][i]
                            predictions[scene_id][object_id][ann_id]["gt_bbox"] = data["gt_bboxes"][i]
                            predictions[scene_id][object_id][ann_id]["iou"] = data["ref_iou"][i]

                # save the last predictions
                with open(pred_path, "wb") as f:
                    pickle.dump(predictions, f)

                # save to global
                ref_acc_all.append(ref_acc)
                ious_all.append(ious)
                masks_all.append(masks)
                others_all.append(others)
                lang_acc_all.append(lang_acc)

            # convert to numpy array
            ref_acc = np.array(ref_acc_all)
            ious = np.array(ious_all)
            masks = np.array(masks_all)
            others = np.array(others_all)
            lang_acc = np.array(lang_acc_all)

            # save the global scores
            with open(score_path, "wb") as f:
                scores = {
                    "ref_acc": ref_acc_all,
                    "ious": ious_all,
                    "masks": masks_all,
                    "others": others_all,
                    "lang_acc": lang_acc_all
                }
                pickle.dump(scores, f)

        else:
            print("loading the scores...")
            with open(score_path, "rb") as f:
                scores = pickle.load(f)

                # unpack
                ref_acc = np.array(scores["ref_acc"])
                ious = np.array(scores["ious"])
                masks = np.array(scores["masks"])
                others = np.array(scores["others"])
                lang_acc = np.array(scores["lang_acc"])

        multiple_dict = {
            "unique": 0,
            "multiple": 1
        }
        others_dict = {
            "not_in_others": 0,
            "in_others": 1
        }

        # evaluation stats
        stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
        stats["overall"] = masks[0].shape[0]
        stats = {}
        for k, v in multiple_dict.items():
            stats[k] = {}
            for k_o, v_o in others_dict.items():
                stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))

            stats[k]["overall"] = np.sum(masks[0] == v)

        stats["overall"] = {}
        for k_o, v_o in others_dict.items():
            stats["overall"][k_o] = np.sum(others[0] == v_o)

        stats["overall"]["overall"] = masks[0].shape[0]

        # aggregate scores
        scores = {}
        for k, v in multiple_dict.items():
            for k_o in others_dict.keys():
                ref_accs, acc_025ious, acc_05ious = [], [], []
                for i in range(masks.shape[0]):
                    running_ref_acc = np.mean(
                        ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                        if np.sum(
                        np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                    running_acc_025iou = ious[i][np.logical_and(
                        np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]),
                        ious[i] >= 0.25)].shape[0] \
                                         / ious[i][np.logical_and(masks[i] == multiple_dict[k],
                                                                  others[i] == others_dict[k_o])].shape[0] \
                        if np.sum(
                        np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                    running_acc_05iou = ious[i][np.logical_and(
                        np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]),
                        ious[i] >= 0.5)].shape[0] \
                                        / ious[i][np.logical_and(masks[i] == multiple_dict[k],
                                                                 others[i] == others_dict[k_o])].shape[0] \
                        if np.sum(
                        np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                    # store
                    ref_accs.append(running_ref_acc)
                    acc_025ious.append(running_acc_025iou)
                    acc_05ious.append(running_acc_05iou)

                if k not in scores:
                    scores[k] = {k_o: {} for k_o in others_dict.keys()}

                scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
                scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
                scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(
                    masks[i] == multiple_dict[k]) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                                     / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(
                    masks[i] == multiple_dict[k]) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                                    / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(
                    masks[i] == multiple_dict[k]) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            scores[k]["overall"] = {}
            scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
            scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

        scores["overall"] = {}
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(
                    others[i] == others_dict[k_o]) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                                     / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(
                    others[i] == others_dict[k_o]) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                                    / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(
                    others[i] == others_dict[k_o]) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            # aggregate
            scores["overall"][k_o] = {}
            scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
            scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i])
            running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
            running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"]["overall"] = {}
        scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

        # report
        print("\nstats:")
        for k_s in stats.keys():
            for k_o in stats[k_s].keys():
                print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

        for k_s in scores.keys():
            print("\n{}:".format(k_s))
            for k_m in scores[k_s].keys():
                for metric in scores[k_s][k_m].keys():
                    print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

        print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))

    def evaluate(self, args):
        print("evaluate...")
        assert args.lang_num_max == 1, 'lang max num == 1; avoid bugs'
        # evaluate
        if args.reference: self.eval_ref(args)
        if args.detection: raise ValueError("UnImplemented mode!")
    
    def train(self, args):
        # init training dataset
        print("preparing data...")
        scanrefer_train, scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new = self.get_scanrefer(
            SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes, args.lang_num_max)
        scanrefer = {
            "train": scanrefer_train,
            "val": scanrefer_val
        }
        scanrefer_new = {
            "train": scanrefer_train_new,
            "val": scanrefer_val_new
        }

        # dataloader
        train_dataset, train_dataloader = self.get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "train", DC,
                                                         augment=True)
        val_dataset, val_dataloader = self.get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "val", DC,
                                                     augment=False)
        dataloader = {
            "train": train_dataloader,
            "val": val_dataloader
        }

        print("initializing...")
        solver, num_params, root = self.get_solver(args, dataloader)

        print("Start training...\n")
        self.save_info(args, root, num_params, train_dataset, val_dataset)
        solver(args.epoch, args.verbose)

    def write_ply(self, verts, colors, indices, output_file):
        if colors is None:
            colors = np.zeros_like(verts)
        if indices is None:
            indices = []

        file = open(output_file, 'w')
        file.write('ply \n')
        file.write('format ascii 1.0\n')
        file.write('element vertex {:d}\n'.format(len(verts)))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('element face {:d}\n'.format(len(indices)))
        file.write('property list uchar uint vertex_indices\n')
        file.write('end_header\n')
        for vert, color in zip(verts, colors):
            file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], int(color[0] * 255),
                                                                int(color[1] * 255), int(color[2] * 255)))
        for ind in indices:
            file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
        file.close()

    def write_bbox(self, bbox, mode, output_file):
        """
        bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
        output_file: string

        """

        def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

            import math

            def compute_length_vec3(vec3):
                return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

            def rotation(axis, angle):
                rot = np.eye(4)
                c = np.cos(-angle)
                s = np.sin(-angle)
                t = 1.0 - c
                axis /= compute_length_vec3(axis)
                x = axis[0]
                y = axis[1]
                z = axis[2]
                rot[0, 0] = 1 + t * (x * x - 1)
                rot[0, 1] = z * s + t * x * y
                rot[0, 2] = -y * s + t * x * z
                rot[1, 0] = -z * s + t * x * y
                rot[1, 1] = 1 + t * (y * y - 1)
                rot[1, 2] = x * s + t * y * z
                rot[2, 0] = y * s + t * x * z
                rot[2, 1] = -x * s + t * y * z
                rot[2, 2] = 1 + t * (z * z - 1)
                return rot

            verts = []
            indices = []
            diff = (p1 - p0).astype(np.float32)
            height = compute_length_vec3(diff)
            for i in range(stacks + 1):
                for i2 in range(slices):
                    theta = i2 * 2.0 * math.pi / slices
                    pos = np.array([radius * math.cos(theta), radius * math.sin(theta), height * i / stacks])
                    verts.append(pos)
            for i in range(stacks):
                for i2 in range(slices):
                    i2p1 = math.fmod(i2 + 1, slices)
                    indices.append(
                        np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
                    indices.append(
                        np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32))
            transform = np.eye(4)
            va = np.array([0, 0, 1], dtype=np.float32)
            vb = diff
            vb /= compute_length_vec3(vb)
            axis = np.cross(vb, va)
            angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
            if angle != 0:
                if compute_length_vec3(axis) == 0:
                    dotx = va[0]
                    if (math.fabs(dotx) != 1.0):
                        axis = np.array([1, 0, 0]) - dotx * va
                    else:
                        axis = np.array([0, 1, 0]) - va[1] * va
                    axis /= compute_length_vec3(axis)
                transform = rotation(axis, -angle)
            transform[:3, 3] += p0
            verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
            verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

            return verts, indices

        def get_bbox_edges(bbox_min, bbox_max):
            def get_bbox_verts(bbox_min, bbox_max):
                verts = [
                    np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                    np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                    np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                    np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                    np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                    np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                    np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                    np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
                ]
                return verts

            box_verts = get_bbox_verts(bbox_min, bbox_max)
            edges = [
                (box_verts[0], box_verts[1]),
                (box_verts[1], box_verts[2]),
                (box_verts[2], box_verts[3]),
                (box_verts[3], box_verts[0]),

                (box_verts[4], box_verts[5]),
                (box_verts[5], box_verts[6]),
                (box_verts[6], box_verts[7]),
                (box_verts[7], box_verts[4]),

                (box_verts[0], box_verts[4]),
                (box_verts[1], box_verts[5]),
                (box_verts[2], box_verts[6]),
                (box_verts[3], box_verts[7])
            ]
            return edges

        def get_bbox_corners(bbox):
            centers, lengths = bbox[:3], bbox[3:6]
            xmin, xmax = centers[0] - lengths[0] / 2, centers[0] + lengths[0] / 2
            ymin, ymax = centers[1] - lengths[1] / 2, centers[1] + lengths[1] / 2
            zmin, zmax = centers[2] - lengths[2] / 2, centers[2] + lengths[2] / 2
            corners = []
            corners.append(np.array([xmax, ymax, zmax]).reshape(1, 3))
            corners.append(np.array([xmax, ymax, zmin]).reshape(1, 3))
            corners.append(np.array([xmin, ymax, zmin]).reshape(1, 3))
            corners.append(np.array([xmin, ymax, zmax]).reshape(1, 3))
            corners.append(np.array([xmax, ymin, zmax]).reshape(1, 3))
            corners.append(np.array([xmax, ymin, zmin]).reshape(1, 3))
            corners.append(np.array([xmin, ymin, zmin]).reshape(1, 3))
            corners.append(np.array([xmin, ymin, zmax]).reshape(1, 3))
            corners = np.concatenate(corners, axis=0)  # 8 x 3

            return corners

        radius = 0.01
        offset = [0, 0, 0]
        verts = []
        indices = []
        colors = []
        corners = get_bbox_corners(bbox)

        box_min = np.min(corners, axis=0)
        box_max = np.max(corners, axis=0)
        palette = {
            0: [0, 255, 0],  # gt
            1: [0, 0, 255]  # pred
        }
        chosen_color = palette[mode]
        edges = get_bbox_edges(box_min, box_max)
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cur_num_verts = len(verts)
            cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
            cyl_verts = [x + offset for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            verts.extend(cyl_verts)
            indices.extend(cyl_ind)
            colors.extend(cyl_color)

        self.write_ply(verts, colors, indices, output_file)

    def read_mesh(self, filename):
        """ read XYZ for each vertex.
        """

        assert os.path.isfile(filename)
        with open(filename, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
            vertices[:, 0] = plydata['vertex'].data['x']
            vertices[:, 1] = plydata['vertex'].data['y']
            vertices[:, 2] = plydata['vertex'].data['z']
            vertices[:, 3] = plydata['vertex'].data['red']
            vertices[:, 4] = plydata['vertex'].data['green']
            vertices[:, 5] = plydata['vertex'].data['blue']

        return vertices, plydata['face']

    def export_mesh(self, vertices, faces):
        new_vertices = []
        for i in range(vertices.shape[0]):
            new_vertices.append(
                (
                    vertices[i][0],
                    vertices[i][1],
                    vertices[i][2],
                    vertices[i][3],
                    vertices[i][4],
                    vertices[i][5],
                )
            )

        vertices = np.array(
            new_vertices,
            dtype=[
                ("x", np.dtype("float32")),
                ("y", np.dtype("float32")),
                ("z", np.dtype("float32")),
                ("red", np.dtype("uint8")),
                ("green", np.dtype("uint8")),
                ("blue", np.dtype("uint8"))
            ]
        )

        vertices = PlyElement.describe(vertices, "vertex")

        return PlyData([vertices, faces])

    def align_mesh(self, scene_id):
        vertices, faces = self.read_mesh(SCANNET_MESH.format(scene_id, scene_id))
        for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
            if 'axisAlignment' in line:
                axis_align_matrix = np.array(
                    [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
                break

        # align
        pts = np.ones((vertices.shape[0], 4))
        pts[:, :3] = vertices[:, :3]
        pts = np.dot(pts, axis_align_matrix.T)
        vertices[:, :3] = pts[:, :3]

        mesh = self.export_mesh(vertices, faces)

        return mesh

    def dump_results(self, args, scanrefer, data, config):
        dump_dir = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis")
        os.makedirs(dump_dir, exist_ok=True)

        # from inputs
        ids = data['scan_idx'].detach().cpu().numpy()
        point_clouds = data['point_clouds'].cpu().numpy()
        batch_size = point_clouds.shape[0]

        pcl_color = data["pcl_color"].detach().cpu().numpy()
        if args.use_color:
            pcl_color = (pcl_color * 256 + MEAN_COLOR_RGB).astype(np.int64)

        # from network outputs
        # detection
        pred_objectness = torch.argmax(data['objectness_scores'], 2).float().detach().cpu().numpy()
        pred_center = data['center'].detach().cpu().numpy()  # (B,K,3)
        pred_heading_class = torch.argmax(data['heading_scores'], -1)  # B,num_proposal
        pred_heading_residual = torch.gather(data['heading_residuals'], 2,
                                             pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
        pred_size_class = torch.argmax(data['size_scores'], -1)  # B,num_proposal
        pred_size_residual = torch.gather(data['size_residuals'], 2,
                                          pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                             3))  # B,num_proposal,1,3
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
        # reference
        pred_ref_scores = data["cluster_ref"].detach().cpu().numpy()
        # pred_ref_scores_softmax = F.softmax(data["cluster_ref"] * torch.argmax(data['objectness_scores'], 2).float() * data['pred_mask'], dim=1).detach().cpu().numpy()
        pred_ref_scores_softmax = F.softmax(data["cluster_ref"], dim=1).detach().cpu().numpy()
        # post-processing
        #nms_masks = data['pred_mask'].detach().cpu().numpy()  # B,num_proposal

        # ground truth
        gt_center = data['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
        gt_heading_class = data['heading_class_label'].cpu().numpy()  # B,K2
        gt_heading_residual = data['heading_residual_label'].cpu().numpy()  # B,K2
        gt_size_class = data['size_class_label'].cpu().numpy()  # B,K2
        gt_size_residual = data['size_residual_label'].cpu().numpy()  # B,K2,3
        # reference
        # gt_ref_labels = data["ref_box_label"].detach().cpu().numpy()
        gt_ref_labels_list = data["ref_box_label_list"].detach().cpu().numpy()

        for i in range(batch_size):
            # basic info
            idx = ids[i]
            scene_id = scanrefer[idx]["scene_id"]
            object_id = scanrefer[idx]["object_id"]
            object_name = scanrefer[idx]["object_name"]
            ann_id = scanrefer[idx]["ann_id"]

            # scene_output
            scene_dump_dir = os.path.join(dump_dir, scene_id)
            if not os.path.exists(scene_dump_dir):
                os.mkdir(scene_dump_dir)

                # # Dump the original scene point clouds
                mesh = self.align_mesh(scene_id)
                mesh.write(os.path.join(scene_dump_dir, 'mesh.ply'))

                write_ply_rgb(point_clouds[i], pcl_color[i], os.path.join(scene_dump_dir, 'pc.ply'))

            # filter out the valid ground truth reference box
            #assert gt_ref_labels[i].shape[0] == gt_center[i].shape[0]
            #gt_ref_idx = np.argmax(gt_ref_labels[i], 0)
            assert gt_ref_labels_list[i][0].shape[0] == gt_center[i].shape[0]
            gt_ref_idx = np.argmax(gt_ref_labels_list[i][0], 0)

            # visualize the gt reference box
            # NOTE: for each object there should be only one gt reference box
            object_dump_dir = os.path.join(dump_dir, scene_id, "gt_{}_{}.ply".format(object_id, object_name))
            gt_obb = config.param2obb(gt_center[i, gt_ref_idx, 0:3], gt_heading_class[i, gt_ref_idx],
                                      gt_heading_residual[i, gt_ref_idx],
                                      gt_size_class[i, gt_ref_idx], gt_size_residual[i, gt_ref_idx])
            gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])

            if not os.path.exists(object_dump_dir):
                self.write_bbox(gt_obb, 0, os.path.join(scene_dump_dir, 'gt_{}_{}.ply'.format(object_id, object_name)))

            # find the valid reference prediction
            #pred_masks = nms_masks[i] * pred_objectness[i] == 1
            assert pred_ref_scores[i].shape[0] == pred_center[i].shape[0]
            # pred_ref_idx = np.argmax(pred_ref_scores[i] * pred_masks, 0)
            pred_ref_idx = np.argmax(pred_ref_scores[i], 0)
            #assigned_gt = torch.gather(data["ref_box_label"], 1, data["object_assignment"]).detach().cpu().numpy()

            # visualize the predicted reference box
            pred_obb = config.param2obb(pred_center[i, pred_ref_idx, 0:3], pred_heading_class[i, pred_ref_idx],
                                        pred_heading_residual[i, pred_ref_idx],
                                        pred_size_class[i, pred_ref_idx], pred_size_residual[i, pred_ref_idx])
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
            iou = box3d_iou(gt_bbox, pred_bbox)

            self.write_bbox(pred_obb, 1, os.path.join(scene_dump_dir,
                                                 'pred_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(object_id, object_name,
                                                                                          ann_id,
                                                                                          pred_ref_scores_softmax[
                                                                                              i, pred_ref_idx], iou)))

    def visualize(self, args):
        # init training dataset
        print("preparing data...")
        scanrefer, scene_list, new_scanrefer = self.get_visualize_scanrefer(args)

        # dataloader
        dataset, dataloader = self.get_eval_dataloader(args, scanrefer, new_scanrefer, scene_list, "val", DC)

        # model
        model = self.get_eval_model(args, DC)

        # evaluate
        print("visualizing...")
        for data in tqdm(dataloader):
            for key in data:
                data[key] = data[key].cuda()

            # feed
            data["epoch"] = 0
            data = model(data)
            _, data = get_loss(
                data_dict=data,
                config=DC,
                detection=True,
                reference=True,
                use_lang_classifier=not args.no_lang_cls
            )

            # visualize
            self.dump_results(args, scanrefer, data, DC)

        print("done!")

    def run(self, mode="train"):
        if mode == "train":
            self.train(self.args)
        elif mode == "eval":
            #self.args.lang_num_max = 1
            #self.args.folder = "2021-12-23_13-42-46"
            #self.args.repeat = 1
            #self.args.batch_size = 20
            self.evaluate(self.args)
        elif mode == "visualize":
            #self.args.lang_num_max = 1
            #self.args.folder = "2021-12-23_13-42-46"
            #self.args.batch_size = 20
            self.visualize(self.args)
    