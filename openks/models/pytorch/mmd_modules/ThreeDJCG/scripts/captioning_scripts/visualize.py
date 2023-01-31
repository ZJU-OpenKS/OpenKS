# HACK ignore warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import h5py
import torch
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from plyfile import PlyData, PlyElement
from shutil import copyfile

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_scene_cap_loss
from models.capnet import CapNet
from lib.eval_helper import eval_cap
from scripts.colors import COLORS

SCANNET_MESH = os.path.join(CONF.PATH.AXIS_ALIGNED_MESH, "{}", "axis_aligned_scene.ply")
SCANNET_AGGR = os.path.join(CONF.PATH.SCANNET_SCANS, "{}/{}_vh_clean.aggregation.json") # scene_id, scene_id

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json")))

VOTENET_DATABASE = h5py.File(os.path.join(CONF.PATH.VOTENET_FEATURES, "val.hdf5"), "r", libver="latest")

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list,  
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=False
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args, dataset, root=CONF.PATH.OUTPUT):
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
        no_caption=False,
        use_topdown=args.use_topdown,
        num_locals=args.num_locals,
        query_mode=args.query_mode,
        graph_mode=args.graph_mode,
        num_graph_steps=args.num_graph_steps,
        use_relation=args.use_relation
    )

    # load
    model_name = "model_last.pth" if args.use_last else "model.pth"
    model_path = os.path.join(root, args.folder, model_name)
    model.load_state_dict(torch.load(model_path), strict=False)
    # model.load_state_dict(torch.load(model_path))

    # to device
    model.cuda()

    # set mode
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.txt".format(split)))])

    return scene_list

def get_eval_data(args):
    eval_scene_list = get_scannet_scene_list("val") if args.scene_id == "-1" else [args.scene_id]
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(SCANREFER_VAL[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("eval on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list

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

def write_ply(verts, colors, indices, output_file):
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
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def write_bbox(corners, color, output_file):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string
    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        
        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
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
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
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

    radius = 0.03
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)

def visualize(args):
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval, eval_scene_list, DC)

    # get model
    model = get_model(args, dataset)

    object_id_to_object_name = {}
    for scene_id in eval_scene_list:
        object_id_to_object_name[scene_id] = {}

        aggr_file = json.load(open(SCANNET_AGGR.format(scene_id, scene_id)))
        for entry in aggr_file["segGroups"]:
            object_id = str(entry["objectId"])
            object_name = entry["label"]
            if len(object_name.split(" ")) > 1: object_name = "_".join(object_name.split(" "))

            object_id_to_object_name[scene_id][object_id] = object_name

    # forward
    print("visualizing...")
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():
            data_dict = model(data_dict, use_tf=False, is_eval=True)
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
        detected_bbox_corners = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
        detected_bbox_centers = data_dict["center"] # batch_size, num_proposals, 3

        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]

            scene_root = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis/{}".format(scene_id))
            os.makedirs(scene_root, exist_ok=True)
            mesh_path = os.path.join(scene_root, "{}.ply".format(scene_id))
            copyfile(SCANNET_MESH.format(scene_id), mesh_path)

            candidates = {}
            for prop_id in range(num_proposals):
                if nms_masks[batch_id, prop_id] == 1:
                    object_id = str(detected_object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], dataset.vocabulary["idx2word"])
                    detected_bbox_corner = detected_bbox_corners[batch_id, prop_id].detach().cpu().numpy()

                    # print(scene_id, object_id)
                    try:
                        ann_list = list(SCANREFER_ORGANIZED[scene_id][object_id].keys())
                        object_name = SCANREFER_ORGANIZED[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        candidates[object_id] = {
                            "object_name": object_name,
                            "description": caption_decoded
                        }

                        ply_name = "pred-{}-{}.ply".format(object_id, object_name)
                        ply_path = os.path.join(scene_root, ply_name)

                        palette_idx = int(object_id) % len(COLORS)
                        color = COLORS[palette_idx]
                        write_bbox(detected_bbox_corner, color, ply_path)
                        
                    except KeyError:
                        continue

            # save predictions for the scene
            pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis/{}/predictions.json".format(scene_id))
            with open(pred_path, "w") as f:
                json.dump(candidates, f, indent=4)

            gt_object_ids = VOTENET_DATABASE["0|{}_gt_ids".format(scene_id)]
            gt_object_ids = np.array(gt_object_ids)

            gt_bbox_corners = VOTENET_DATABASE["0|{}_gt_corners".format(scene_id)]
            gt_bbox_corners = np.array(gt_bbox_corners)

            for i, object_id in enumerate(gt_object_ids):
                object_id = str(int(object_id))
                object_name = object_id_to_object_name[scene_id][object_id]

                ply_name = "gt-{}-{}.ply".format(object_id, object_name)
                ply_path = os.path.join(scene_root, ply_name)

                palette_idx = int(object_id) % len(COLORS)
                color = COLORS[palette_idx]
                write_bbox(gt_bbox_corners[i], color, ply_path)

    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--scene_id", type=str, help="scene id", default="-1")

    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--num_locals", type=int, default=-1, help="Number of local objects [default: -1]")
    parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
    parser.add_argument("--query_mode", type=str, default="corner", help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument("--graph_mode", type=str, default="edge_conv", help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr", type=str, default="add", help="Mode for aggregating features, [choices: add, mean, max]")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    
    parser.add_argument("--use_tf", action="store_true", help="Enable teacher forcing")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_last", action="store_true", help="Use the last model")
    parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
    parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
    args = parser.parse_args()

    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    visualize(args)

