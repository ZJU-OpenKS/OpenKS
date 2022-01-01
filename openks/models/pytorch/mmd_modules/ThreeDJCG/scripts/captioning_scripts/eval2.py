# HACK ignore warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import torch
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset2 import ScannetReferenceDataset
from lib.config import CONF
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper2 import get_scene_cap_loss
from models.capnet2 import CapNet
from lib.eval_helper2 import eval_cap

# SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
# SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_new=scanrefer_new,
        scanrefer_all_scene=all_scene_list,  
        split="val",
        name=args.dataset,
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max,
        augment=False
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args, dataset, device, root=CONF.PATH.OUTPUT, eval_pretrained=False):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = CapNet(
        num_class=DC.num_class,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        no_caption=not args.eval_caption,
        use_topdown=args.use_topdown,
        num_locals=args.num_locals,
        query_mode=args.query_mode,
        graph_mode=args.graph_mode,
        num_graph_steps=args.num_graph_steps,
        use_relation=args.use_relation,
        dataset_config=DC
    )

    if eval_pretrained:
        # load pretrained model
        print("loading pretrained VoteNet...")
        pretrained_model = CapNet(
            num_class=DC.num_class,
            vocabulary=dataset.vocabulary,
            embeddings=dataset.glove,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            no_caption=True
        )

        pretrained_name = "PRETRAIN_VOTENET_XYZ"
        if args.use_color: pretrained_name += "_COLOR"
        if args.use_multiview: pretrained_name += "_MULTIVIEW"
        if args.use_normal: pretrained_name += "_NORMAL"

        pretrained_path = os.path.join(CONF.PATH.PRETRAINED, pretrained_name, "model.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal
    else:
        # load
        model_name = "model_last.pth" if args.use_last else "model.pth"
        model_path = os.path.join(root, args.folder, model_name)
        model.load_state_dict(torch.load(model_path), strict=False)
        # model.load_state_dict(torch.load(model_path))

    # multi-GPU
    if torch.cuda.device_count() > 1:
        print("using {} GPUs...".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    
    # to device
    model.to(device)

    # set mode
    model.eval()

    return model

def get_scannet_scene_list(data):
    # scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.txt".format(split)))])
    scene_list = sorted(list(set([d["scene_id"] for d in data])))

    return scene_list

def get_eval_data(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "ReferIt3D":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    eval_scene_list = get_scannet_scene_list(scanrefer_train) if args.use_train else get_scannet_scene_list(scanrefer_val)
    scanrefer_eval = []
    scanrefer_eval_new = []
    for scene_id in eval_scene_list:
        data = deepcopy(scanrefer_train[0]) if args.use_train else deepcopy(scanrefer_val[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)
        scanrefer_eval_new_scene = []
        for i in range(args.lang_num_max):
            scanrefer_eval_new_scene.append(data)
        scanrefer_eval_new.append(scanrefer_eval_new_scene)

    print("eval on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list, scanrefer_eval_new

def eval_caption(args):
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get eval data
    scanrefer_eval, eval_scene_list, scanrefer_eval_new = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval, scanrefer_eval_new, eval_scene_list, DC)

    # get model
    model = get_model(args, dataset, device)

    # evaluate
    bleu, cider, rouge, meteor = eval_cap(model, device, dataset, dataloader, "val", args.folder, args.use_tf, 
        force=args.force, save_interm=args.save_interm, min_iou=args.min_iou)

    # report
    print("\n----------------------Evaluation-----------------------")
    print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    print("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1])))
    print()

def eval_detection(args):
    print("evaluate detection...")
    # constant
    DC = ScannetDatasetConfig()
    
    # init training dataset
    print("preparing data...")
    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval, eval_scene_list, DC)

    # model
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # root = CONF.PATH.PRETRAINED if args.eval_pretrained else CONF.PATH.OUTPUT
    model = get_model(args, dataset, device, eval_pretrained=args.eval_pretrained)

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
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()

        # feed
        with torch.no_grad():
            data = model(data, False, True)
            data = get_scene_cap_loss(data, device, DC, weights=dataset.weights, detection=True, caption=False)

        batch_pred_map_cls = parse_predictions(data, POST_DICT) 
        batch_gt_map_cls = parse_groundtruths(data, POST_DICT) 
        for ap_calculator in AP_CALCULATOR_LIST:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate object detection results and report
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    # parser.add_argument("--gpu", type=str, help="gpu", default=["0"], nargs="+")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--num_locals", type=int, default=-1, help="Number of local objects [default: -1]")
    parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
    
    parser.add_argument("--query_mode", type=str, default="corner", help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument("--graph_mode", type=str, default="edge_conv", help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr", type=str, default="add", help="Mode for aggregating features, [choices: add, mean, max]")
    
    parser.add_argument("--min_iou", type=float, default=0.25, help="Min IoU threshold for evaluation")
    
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    
    parser.add_argument("--use_tf", action="store_true", help="Enable teacher forcing")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_last", action="store_true", help="Use the last model")
    parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
    parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
    
    parser.add_argument("--eval_caption", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--eval_detection", action="store_true", help="evaluate the object detection results")
    parser.add_argument("--eval_pretrained", action="store_true", help="evaluate the pretrained object detection results")
    
    parser.add_argument("--force", action="store_true", help="generate the results by force")
    parser.add_argument("--save_interm", action="store_true", help="Save the intermediate results")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # evaluate
    if args.eval_caption: eval_caption(args)
    if args.eval_detection: eval_detection(args)

