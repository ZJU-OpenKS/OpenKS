import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.visual_question_answering.dataset import ScanVQADataset
from lib.visual_question_answering.solver_3dgqa import Solver
from lib.configs.config_vqa import CONF
from models.vqanet.vqanet import VqaNet
from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict
import crash_on_ipy

# TODO val_seen, val_unseen
SCANVQA_TRAIN = []
SCANVQA_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_train.json")))
# # SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_train.json")))
# # SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_train.json")))
# # SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_train.json")))
# TODO more dataset
# SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_generated.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanRefer_filtered_generated.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/nr3d_generated.json")))
# #
# # # # TODO template-based; dataset-size x 2
# SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_train.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_generated.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanRefer_filtered_generated_masked.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/nr3d_generated_masked.json")))

SCANVQA_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_train.json")))  # UNSEEN
# SCANVQA_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_val.json")))  # UNSEEN
# SCANVQA_VAL_SEEN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA_val.json")))

SCANVQA_ANSWER_LIST = []
SCANVQA_ANSWER_LIST += [data["answer"] for data in SCANVQA_TRAIN]
SCANVQA_ANSWER_LIST += [data["answer"] for data in SCANVQA_VAL]
SCANVQA_ANSWER_LIST = sorted(list(set(SCANVQA_ANSWER_LIST)))

# constants
DC = ScannetDatasetConfig()


def get_dataloader(args, scanvqa, scene_list, split, config, augment, shuffle=True):
    dataset = ScanVQADataset(
        scanvqa_data=scanvqa[split],
        scanvqa_all_scene=scene_list,
        answer_type=SCANVQA_ANSWER_LIST,
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
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, collate_fn=dataset.collate_fn)

    return dataset, dataloader

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = VqaNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        no_reference=args.no_reference,
        dataset_config=DC
    )

    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained VoteNet...")

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        load_result = model.load_state_dict(torch.load(pretrained_path), strict=False)
        print(load_result, flush=True)

        # mount
        # model.backbone_net = pretrained_model.backbone_net
        # model.vgen = pretrained_model.vgen
        # model.proposal = pretrained_model.proposal

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


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    weight_dict = {
        # 'encoder': {'lr': 0.000001},
        # 'decoder': {'lr': 0.000001},
        'lang': {'lr': 0.0001},
        'relation': {'lr': 0.0001},
        'crossmodal': {'lr': 0.0001},
    }
    params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
    # params = model.parameters()
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    CONF.PATH.OUTPUT = os.path.join(CONF.PATH.OUTPUT, 'visual_question_answering')
    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    if args.coslr:
        lr_args = {
            'type': 'cosine',
            'T_max': args.epoch,
            'eta_min': 1e-5,
        }
    else:
        lr_args = None
    if args.no_reference:
        bn_args = {
            'step': 20,
            'rate': 0.5
        }
    else:
        bn_args = None

    print('learning rate and batch norm args', lr_args, bn_args, flush=True)
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
        lr_args=lr_args,
        bn_args=bn_args,
    )
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, train_dataset, val_dataset):
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


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

# def get_splited_data(scanvqa_data, scene_list, lang_num_max):
#     # filter data in chosen scenes
#     new_scanvqa = []
#     scanvqa = []
#     scene_id = ""
#     for data in scanvqa_data:
#         if data["scene_id"] in scene_list:
#             if scene_id != data["scene_id"]:
#                 scene_id = data["scene_id"]
#                 if len(new_scanvqa) > 0:
#                     scanvqa.append(new_scanvqa)
#                     new_scanvqa = []
#             if len(new_scanvqa) == lang_num_max:
#                 scanvqa.append(new_scanvqa)
#                 new_scanvqa = []
#             new_scanvqa.append(data)
#     if len(new_scanvqa) > 0:
#         scanvqa.append(new_scanvqa)
#     return scanvqa

def get_scanvqa(scanvqa_train, scanvqa_val, num_scenes, lang_num_max):
    # get initial scene list
    train_scene_list = get_scannet_scene_list("train")
    val_scene_list = get_scannet_scene_list("val")
    # train_scene_list = sorted(list(set([data["scene_id"] for data in scanvqa_train])))
    # val_scene_list = sorted(list(set([data["scene_id"] for data in scanvqa_val])))
    if num_scenes == -1:
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes

    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]
    all_scene_list = train_scene_list + val_scene_list

    scanvqa_train = [value for value in scanvqa_train if value["scene_id"] in train_scene_list]
    scanvqa_val = [value for value in scanvqa_val if value["scene_id"] in val_scene_list]
    # scanvqa_train = get_splited_data(scanvqa_train, train_scene_list, lang_num_max)
    # scanvqa_val = get_splited_data(scanvqa_val, val_scene_list, lang_num_max)

    # print("scanvqa_iter_number", len(scanvqa_train), len(scanvqa_val), 'lang_per_iter', lang_num_max)
    # # sum = 0
    # # for i in range(len(scanvqa_train)):
    # #     sum += len(scanvqa_train[i])
    # #     # print(len(scanvqa_train_new[i]))
    # # # for i in range(len(scanvqa_val_new)):
    # # #    print(len(scanvqa_val_new[i]))
    # # print("train data number", sum)  # 1418 363
    # # all scanvqa scene
    # all_scene_list = train_scene_list + val_scene_list

    # print(sum([len(data) for data in scanvqa_train]))
    # print("train on {} samples and val on {} samples".format( \
    #     sum([len(data) for data in scanvqa_train]),
    #     sum([len(data) for data in scanvqa_val])
    # ))

    print("train on {} samples and val on {} samples".format(len(scanvqa_train), len(scanvqa_val)))
    return scanvqa_train, scanvqa_val, train_scene_list, val_scene_list, all_scene_list


def train(args):
    # init training dataset
    print("preparing data...")
    scanvqa_train, scanvqa_val, all_scene_list, train_scene_list, val_scene_list = \
        get_scanvqa(SCANVQA_TRAIN, SCANVQA_VAL, args.num_scenes, args.lang_num_max)

    scanvqa = {
        "train": scanvqa_train,
        "val": scanvqa_val
    }

    # dataloade
    train_dataset, train_dataloader = get_dataloader(args, scanvqa, train_scene_list, "train", DC, augment=True, shuffle=True)
    val_dataset, val_dataloader = get_dataloader(args, scanvqa, val_scene_list, "val", DC, augment=False, shuffle=False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    # parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-3)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=32)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--coslr", action='store_true', help="cosine learning rate")
    parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")

    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_pretrained", type=str,
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)

