import os
import sys
import json
import torch
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from numpy.linalg import inv

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import openks.models.pytorch.mmd_modules.ThreeDJCG.lib.capeval.bleu.bleu as capblue
import openks.models.pytorch.mmd_modules.ThreeDJCG.lib.capeval.cider.cider as capcider
import openks.models.pytorch.mmd_modules.ThreeDJCG.lib.capeval.rouge.rouge as caprouge
import openks.models.pytorch.mmd_modules.ThreeDJCG.lib.capeval.meteor.meteor as capmeteor

from openks.models.pytorch.mmd_modules.ThreeDJCG.data.scannet.model_util_scannet import ScannetDatasetConfig
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.config_captioning import CONF
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.ap_helper import parse_predictions
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.loss_helper.loss_captioning import get_scene_cap_loss, get_object_cap_loss
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.box_util import box3d_iou_batch_tensor

# constants
DC = ScannetDatasetConfig()

SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json")))

def prepare_corpus(raw_data, max_len=CONF.TRAIN.MAX_DES_LEN):
    corpus = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        object_id = data["object_id"]
        object_name = data["object_name"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)
        # key = "{}|{}".format(scene_id, object_id)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus

def decode_caption(raw_caption, idx2word):
    decoded = ["sos"]
    for token_idx in raw_caption:
        token_idx = token_idx.item()
        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded

def check_candidates(corpus, candidates):
    placeholder = "sos eos"
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates

def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates


def feed_scene_cap(model, device, dataset, dataloader, phase, folder, 
    is_eval=True, max_len=CONF.TRAIN.MAX_DES_LEN, save_interm=False, min_iou=CONF.EVAL.MIN_IOU_THRESHOLD, organized=SCANREFER_ORGANIZED):
    candidates = {}
    intermediates = {}
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():
            data_dict = model(data_dict, is_eval)
            data_dict = get_scene_cap_loss(data_dict, device, DC, weights=dataset.weights, detection=True, caption=False)

        # unpack
        captions = data_dict["lang_cap"].argmax(-1) # batch_size, num_proposals, max_len - 1
        dataset_ids = data_dict["dataset_idx"]
        batch_size, num_proposals, _ = captions.shape

        # post-process
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

        # nms mask
        _ = parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.LongTensor(data_dict["pred_mask"]).cuda()

        # objectness mask
        obj_masks = torch.argmax(data_dict["objectness_scores"], 2).long()

        # final mask
        nms_masks = nms_masks * obj_masks

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1, data_dict["object_assignment"])

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            data_dict["gt_box_corner_label"].float(), 
            1, 
            data_dict["object_assignment"].view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
        ) # batch_size, num_proposals, 8, 3
        detected_bbox_corners = data_dict["pred_bbox_corner"] # batch_size, num_proposals, 8, 3
        
        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3), # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3) # batch_size * num_proposals, 8, 3
        ).view(batch_size, num_proposals)
        
        # find good boxes (IoU > threshold)
        good_bbox_masks = ious > min_iou # batch_size, num_proposals

        # dump generated captions
        object_attn_masks = {}
        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            object_attn_masks[scene_id] = np.zeros((num_proposals, num_proposals))
            for prop_id in range(num_proposals):
                if nms_masks[batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
                    object_id = str(detected_object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], dataset.vocabulary["idx2word"])

                    # print(scene_id, object_id)
                    try:
                        ann_list = list(organized[scene_id][object_id].keys())
                        object_name = organized[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        key = "{}|{}|{}".format(scene_id, object_id, object_name)
                        # key = "{}|{}".format(scene_id, object_id)
                        candidates[key] = [caption_decoded]

                        if save_interm:
                            if scene_id not in intermediates: intermediates[scene_id] = {}
                            if object_id not in intermediates[scene_id]: intermediates[scene_id][object_id] = {}

                            intermediates[scene_id][object_id]["object_name"] = object_name
                            intermediates[scene_id][object_id]["box_corner"] = detected_bbox_corners[batch_id, prop_id].cpu().numpy().tolist()
                            intermediates[scene_id][object_id]["description"] = caption_decoded
                            intermediates[scene_id][object_id]["token"] = caption_decoded.split(" ")

                            # attention context
                            # extract attention masks for each object
                            object_attn_weights = data_dict["topdown_attn"][:, :, :num_proposals] # NOTE only consider attention on objects
                            valid_context_masks = data_dict["valid_masks"][:, :, :num_proposals] # NOTE only consider attention on objects

                            cur_valid_context_masks = valid_context_masks[batch_id, prop_id] # num_proposals
                            cur_context_box_corners = detected_bbox_corners[batch_id, cur_valid_context_masks == 1] # X, 8, 3
                            cur_object_attn_weights = object_attn_weights[batch_id, prop_id, cur_valid_context_masks == 1] # X

                            intermediates[scene_id][object_id]["object_attn_weight"] = cur_object_attn_weights.cpu().numpy().T.tolist()
                            intermediates[scene_id][object_id]["object_attn_context"] = cur_context_box_corners.cpu().numpy().tolist()

                        # cache
                        object_attn_masks[scene_id][prop_id, prop_id] = 1
                    except KeyError:
                        continue

    # detected boxes
    if save_interm:
        print("saving intermediate results...")
        interm_path = os.path.join(CONF.PATH.OUTPUT, folder, "interm.json")
        with open(interm_path, "w") as f:
            json.dump(intermediates, f, indent=4)

    return candidates


def update_interm(interm, candidates, bleu, cider, rouge, meteor):
    for i, (key, value) in enumerate(candidates.items()):
        scene_id, object_id, object_name = key.split("|")
        if scene_id in interm:
            if object_id in interm[scene_id]:
                interm[scene_id][object_id]["bleu_1"] = bleu[1][0][i]
                interm[scene_id][object_id]["bleu_2"] = bleu[1][1][i]
                interm[scene_id][object_id]["bleu_3"] = bleu[1][2][i]
                interm[scene_id][object_id]["bleu_4"] = bleu[1][3][i]

                interm[scene_id][object_id]["cider"] = cider[1][i]

                interm[scene_id][object_id]["rouge"] = rouge[1][i]

                interm[scene_id][object_id]["meteor"] = meteor[1][i]

    return interm

def eval_cap(model, device, dataset, dataloader, phase, folder, 
    is_eval=True, max_len=CONF.TRAIN.MAX_DES_LEN, force=False,
    mode="scene", save_interm=False, no_caption=False, no_classify=False, min_iou=CONF.EVAL.MIN_IOU_THRESHOLD):
    if no_caption:
        bleu = 0
        cider = 0
        rouge = 0
        meteor = 0

        if no_classify:
            cls_acc = 0
        else:
            print("evaluating classification accuracy...")
            cls_acc = []
            for data_dict in tqdm(dataloader):
                # move to cuda
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(device)

                with torch.no_grad():
                    data_dict = model(data_dict, is_eval)
                
                # unpack
                preds = data_dict["enc_preds"] # (B, num_cls)
                targets = data_dict["object_cat"] # (B,)
                
                # classification acc
                preds = preds.argmax(-1) # (B,)
                acc = (preds == targets).sum().float() / targets.shape[0]
                
                # dump
                cls_acc.append(acc.item())

            cls_acc = np.mean(cls_acc)
    else:
        # corpus
        corpus_path = os.path.join(CONF.PATH.OUTPUT, folder, "corpus_{}.json".format(phase))
        if not os.path.exists(corpus_path) or force:
            print("preparing corpus...")
            if dataset.name == "ScanRefer":
                raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.json".format(phase))))
            elif dataset.name == "ReferIt3D":
                raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_{}.json".format(phase))))
            else:
                raise ValueError("Invalid dataset.")

            corpus = prepare_corpus(raw_data, max_len)
            with open(corpus_path, "w") as f:
                json.dump(corpus, f, indent=4)
        else:
            print("loading corpus...")
            with open(corpus_path) as f:
                corpus = json.load(f)

        if dataset.name == "ScanRefer":
            organized = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json")))
        elif dataset.name == "ReferIt3D":
            organized = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_organized.json")))
        else:
            raise ValueError("Invalid dataset.")

        pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}.json".format(phase))
        # if not os.path.exists(pred_path) or force:
        # generate results
        print("generating descriptions...")
        if mode == "scene":
            candidates = feed_scene_cap(model, device, dataset, dataloader, phase, folder, is_eval, max_len, save_interm, min_iou, organized=organized)
        elif mode == "object":
            candidates, cls_acc = feed_object_cap(model, device, dataset, dataloader, phase, folder, is_eval, max_len)
        elif mode == "oracle":
            candidates = feed_oracle_cap(model, device, dataset, dataloader, phase, folder, is_eval, max_len)
        else:
            raise ValueError("invalid mode: {}".format(mode))

        # check candidates
        # NOTE: make up the captions for the undetected object by "sos eos"
        candidates = check_candidates(corpus, candidates)

        candidates = organize_candidates(corpus, candidates)

        with open(pred_path, "w") as f:
            json.dump(candidates, f, indent=4)
        # else:
        #     print("loading descriptions...")
        #     with open(pred_path) as f:
        #         candidates = json.load(f)

        # compute scores
        print("computing scores...")
        bleu = capblue.Bleu(4).compute_score(corpus, candidates)
        cider = capcider.Cider().compute_score(corpus, candidates)
        rouge = caprouge.Rouge().compute_score(corpus, candidates)
        meteor = capmeteor.Meteor().compute_score(corpus, candidates)
        #meteor = 0.

        # # save scores
        # print("saving scores...")
        # score_path = os.path.join(CONF.PATH.OUTPUT, folder, "score_{}.json".format(phase))
        # with open(score_path, "w") as f:
        #     scores = {
        #         "bleu-1": [float(s) for s in bleu[1][0]],
        #         "bleu-2": [float(s) for s in bleu[1][1]],
        #         "bleu-3": [float(s) for s in bleu[1][2]],
        #         "bleu-4": [float(s) for s in bleu[1][3]],
        #         "cider": [float(s) for s in cider[1]],
        #         "rouge": [float(s) for s in rouge[1]],
        #         "meteor": [float(s) for s in meteor[1]],
        #     }
        #     json.dump(scores, f, indent=4)

        # update intermediates
        if save_interm:
            print("updating intermediate results...")
            interm_path = os.path.join(CONF.PATH.OUTPUT, folder, "interm.json")
            with open(interm_path) as f:
                interm = json.load(f)

            interm = update_interm(interm, candidates, bleu, cider, rouge, meteor)

            with open(interm_path, "w") as f:
                json.dump(interm, f, indent=4)

    if mode == "scene" or mode == "oracle":
        return bleu, cider, rouge, meteor
    else:
        return bleu, cider, rouge, meteor, cls_acc

