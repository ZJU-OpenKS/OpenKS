import os
import sys
import json
import h5py
import pickle
import argparse
import importlib
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile
from plyfile import PlyData, PlyElement

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from utils.pc_utils import write_ply_rgb, write_oriented_bbox
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch, box3d_iou_batch_tensor
from lib.ap_helper4 import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper.loss_vqa import get_loss
from lib.visual_question_answering.eval_helper import get_eval
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.visual_question_answering.dataset import ScanVQADataset
from lib.visual_question_answering.solver_v0 import Solver
from lib.config_vqa import CONF
from models.vqanet.vqanet_v6 import VqaNet
from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict
import crash_on_ipy

print('Import Done', flush=True)

SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_test.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanvqa, scene_list, split, config):
    dataset = ScanVQADataset(
        scanvqa_data=scanvqa,
        scanvqa_all_scene=scene_list,
        answer_type=[],
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max
    )
    print("test on {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

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
    ).cuda()

    model_name = "model_last.pth" if args.last_ckpt else "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):
    scanrefer = SCANREFER_VAL
    scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]
    # TODO Change The Data Type To VQA
    new_data = []
    prefix = args.dataset
    for value in scanrefer:
        current_label = {
            'source': f'{prefix} dataset based',
            'scene_id': value['scene_id'],
            'question_type': 'grounding',
            'question': value['description'],
            'answer': ' '.join(value['object_name'].split('_')),
            'related_object(type 1)': [int(value['object_id'])],  # todo
            'related_object(type 3)': [],  # todo
            'rank(filter)': 'A',
            'issue(filter)': 'template based',
            'ann_id': value['ann_id'],
            'object_id': value['object_id'],
            'object_name': value['object_name']
        }
        new_data.append(current_label)
    scanrefer = new_data
    return scanrefer, scene_list


def predict_ref(args):
    print("evaluate localization...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "test", DC)

    # model
    model = get_model(args)

    # random seeds
    seed = args.seed

    # evaluate
    print("generating bboxes...")
    pred_bboxes = []

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
    for idx, data in enumerate(tqdm(dataloader)):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()

        # feed
        with torch.no_grad():
            data["epoch"] = 0
            data = model(data)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data = get_loss(
                data_dict=data,
                config=DC,
                detection=True,
                qa=True,
                use_lang_classifier=True
            )
            # config
            objectness_preds_batch = torch.argmax(data['objectness_scores'], 2).long()
            # objectness_labels_batch = data_dict['objectness_label'].long()  # GT

            # config
            post_processing = {
                "remove_empty_box": True,
                "use_3d_nms": True,
                "nms_iou": 0.25,
                "use_old_type_nms": False,
                "cls_nms": True,
                "per_class_proposal": True,
                "conf_thresh": 0.05,
                "dataset_config": DC
            }

            # if post_processing:  # Must Have NMS Mask; otherwise the iou precision and recall will be not right
            _ = parse_predictions(data, post_processing)
            nms_masks = torch.LongTensor(data['pred_mask']).cuda()
            # construct valid mask
            pred_masks = (nms_masks * objectness_preds_batch == 1).float()
            # label_masks = (objectness_labels_batch == 1).float()
            pred_masks = pred_masks.detach()
            # Predict Answer
            pred_answer = data['vqa_pred_answer'].detach().cpu().numpy()
            pred_related_object_confidence = data['vqa_pred_related_object_confidence'].sigmoid().detach()

            # predicted bbox
            pred_heading = data['pred_heading'].detach().cpu().numpy() # B,num_proposal
            pred_center = data['pred_center'].detach().cpu().numpy() # (B, num_proposal)
            pred_box_size = data['pred_size'].detach().cpu().numpy() # (B, num_proposal, 3)

            #print("pred_ref", pred_ref.shape, gt_ref.shape)
            batch_size = pred_heading.shape[0]
            for i in range(batch_size):
                # compute the iou
                pred_bbox_batch = get_3d_box_batch(pred_box_size[i], pred_heading[i], pred_center[i])
                pred_bbox_batch = torch.from_numpy(pred_bbox_batch).float().to(pred_masks.device)
                for j in range(args.lang_num_max):
                    x = 0
                    pred_related_batch = pred_related_object_confidence[i, j, :, x]
                    pred_bbox_id = torch.argmax(pred_related_batch * pred_masks[i, :], -1)
                    pred_bbox = pred_bbox_batch[pred_bbox_id].cpu().numpy()

                    # construct the multiple mask
                    multiple = data["unique_multiple_list"][i][j]
                    # construct the others mask
                    others = 1 if data["object_cat_list"][i][j] == 17 else 0
                    # store data
                    scanrefer_idx = data["scan_idx"][i].item()
                    pred_data = {
                        "scene_id": scanrefer[scanrefer_idx]["scene_id"],
                        "object_id": scanrefer[scanrefer_idx]["object_id"],
                        "ann_id": scanrefer[scanrefer_idx]["ann_id"],
                        "bbox": pred_bbox.tolist(),
                        "unique_multiple": multiple,
                        "others": others
                    }
                    pred_bboxes.append(pred_data)

    # dump
    print("dumping...")
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred.json")
    with open(pred_path, "w") as f:
        json.dump(pred_bboxes, f, indent=4)

    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer_filtered of nr3d", default="ScanRefer_filtered")
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=1)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--force", action="store_true", help="enforce the generation of results")  # Not Useful
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_oracle", action="store_true", help="Use ground truth bounding boxes.")
    parser.add_argument("--use_cat_rand", action="store_true", help="Use randomly selected bounding boxes from correct categories as outputs.")
    parser.add_argument("--use_best", action="store_true", help="Use best bounding boxes as outputs.")
    parser.add_argument("--last_ckpt", action="store_true", help="evaluate the last_ckpt results")

    args = parser.parse_args()

    assert args.lang_num_max == 1, 'lang max num == 1; avoid bugs'
    # # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # evaluate
    predict_ref(args)

