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
import torch.nn.functional as F
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
SCANNET_ROOT = os.path.join(CONF.PATH.DATA, 'scannet/scans')
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply") # scene_id, scene_id
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt") # scene_id, scene_id 

# SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_test.json")))
# TODO ScanVQA Train And Val
SCANVQA_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_train.json")))
# TODO more dataset
SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_generated.json")))
SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanRefer_filtered_generated.json")))
SCANVQA_TRAIN += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/nr3d_generated.json")))

# SCANVQA_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_train.json")))  # UNSEEN
SCANVQA_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_generated.json")))
# SCANVQA_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanRefer_filtered_generated.json")))
# SCANVQA_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/nr3d_generated_masked.json")))

SCANVQA_ANSWER_LIST = []
SCANVQA_ANSWER_LIST += [data["answer"] for data in SCANVQA_TRAIN]
SCANVQA_ANSWER_LIST += [data["answer"] for data in SCANVQA_VAL]
SCANVQA_ANSWER_LIST = sorted(list(set(SCANVQA_ANSWER_LIST)))

# constants
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DC = ScannetDatasetConfig()

def get_dataloader(args, scanvqa, scene_list, split, config):
    dataset = ScanVQADataset(
        scanvqa_data=scanvqa,
        scanvqa_all_scene=scene_list,
        answer_type=SCANVQA_ANSWER_LIST,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=1,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

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
        dataset_config=DC
    ).cuda()

    path = os.path.join(CONF.PATH.OUTPUT, args.folder, "model.pth")
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanvqa(args):
    if args.detection:
        scene_list = get_scannet_scene_list("val")
        scanvqa = []
        for scene_id in scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            scanvqa.append(data)
    else:
        if args.dataset == 'ScanRefer_filtered':
            SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
            SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
        elif args.dataset == 'nr3d':
            SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
            SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d.json")))
        else:
            raise NotImplementedError()

        scanvqa = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
        scene_list = sorted(list(set([data["scene_id"] for data in scanvqa])))
        scanvqa = [data for data in scanvqa if data["scene_id"] in scene_list]
        # TODO Change The Data Type To VQA
        new_data = []
        prefix = args.dataset
        for value in scanvqa:
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
            # current_label = {
            #     'source': f'{prefix} dataset based',
            #     'scene_id': value['scene_id'],
            #     'question_type': 'grounding',
            #     'question': value['description'].replace(value['object_name'], '[mask]'),
            #     'answer': value['object_name'],
            #     'related_object(type 1)': [],  # todo
            #     'related_object(type 3)': [value['object_id']],  # todo
            #     'rank(filter)': 'A',
            #     'issue(filter)': 'template based'
            # }
            # new_data.append(current_label)
        scanvqa = new_data
    return scanvqa, scene_list

def get_scanvqa(args):
    # get initial scene list
    train_scene_list = get_scannet_scene_list("train")
    val_scene_list = get_scannet_scene_list("val")
    # train_scene_list = sorted(list(set([data["scene_id"] for data in scanvqa_train])))
    # val_scene_list = sorted(list(set([data["scene_id"] for data in scanvqa_val])))
    # slice train_scene_list
    all_scene_list = train_scene_list + val_scene_list
    scanvqa_train, scanvqa_val = SCANVQA_TRAIN, SCANVQA_VAL
    scanvqa_train = [value for value in scanvqa_train if value["scene_id"] in train_scene_list]
    scanvqa_val = [value for value in scanvqa_val if value["scene_id"] in val_scene_list]
    print("train on {} samples and val on {} samples".format(len(scanvqa_train), len(scanvqa_val)))
    return scanvqa_train, scanvqa_val, train_scene_list, val_scene_list, all_scene_list


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

def write_bbox(bbox, mode, output_file):
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
        corners = np.concatenate(corners, axis=0) # 8 x 3

        return corners


    radius = 0.04
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    corners = get_bbox_corners(bbox)

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    palette = {
        0: [0, 255, 0], # gt, green
        1: [0, 0, 255], # pred, blue
        2: [255, 0, 0], # red
        3: [0, 255, 255], # cyan
        4: [255, 255, 0],
        5: [0, 255, 255],
        6: [210, 128, 210] # purple
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

    # TODO change it to multiple-object
    if output_file is not None:
        write_ply(verts, colors, indices, output_file)
    else:
        return verts, colors, indices  # list


def read_mesh(filename):
    """ read XYZ for each vertex.
    """

    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

    return vertices, plydata['face']

def export_mesh(vertices, faces):
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

def align_mesh(scene_id):
    vertices, faces = read_mesh(SCANNET_MESH.format(scene_id, scene_id))
    for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
            break
    
    # align
    pts = np.ones((vertices.shape[0], 4))
    pts[:, :3] = vertices[:, :3]
    pts = np.dot(pts, axis_align_matrix.T)
    vertices[:, :3] = pts[:, :3]

    mesh = export_mesh(vertices, faces)

    return mesh

def write_bboxes(bboxes, output_file):
    verts, colors, indices = [], [], []
    vert_start = 0
    # for i, x in enumerate([0, 6, 3]):
    for i, x in enumerate([0, 2, 3]):
        for k in bboxes[i]:
            new_verts, new_colors, new_indices = write_bbox(k, x, output_file=None)
            neww_indices = []  # filter; the indices will be added
            for face in new_indices:
                neww_indices.append(face + vert_start)
            vert_start = vert_start + len(new_verts)
            verts = verts + new_verts
            colors = colors + new_colors
            indices = indices + neww_indices
            # if len(new_verts) == 0:
            #     continue
        # import ipdb; ipdb.set_trace()
    # print('write bboxes', bboxes, output_file)
    write_ply(verts, colors, indices, output_file)


def dump_results(args, scanvqa, data, config):
    dump_dir = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis")
    os.makedirs(dump_dir, exist_ok=True)

    # from inputs
    ids = data['scan_idx'].detach().cpu().numpy()
    point_clouds = data['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    pcl_color = data["pcl_color"].detach().cpu().numpy()
    if args.use_color:
        pcl_color = (pcl_color * 256 + MEAN_COLOR_RGB).astype(np.int64)

    # # from network outputs
    # # detection
    # # predicted bbox
    # pred_heading = data['pred_heading'].detach().cpu().numpy() # B,num_proposal
    # pred_center = data['pred_center'].detach().cpu().numpy() # (B, num_proposal)
    # pred_box_size = data['pred_size'].detach().cpu().numpy() # (B, num_proposal, 3)
    # # reference
    # pred_ref_scores = data["cluster_ref"].detach().cpu().numpy()
    # #pred_ref_scores_softmax = F.softmax(data["cluster_ref"] * torch.argmax(data['objectness_scores'], 2).float() * data['pred_mask'], dim=1).detach().cpu().numpy()
    # pred_ref_scores_softmax = F.softmax(data["cluster_ref"], dim=1).detach().cpu().numpy()
    # # post-processing
    # #nms_masks = data['pred_mask'].detach().cpu().numpy() # B,num_proposal
    # # ground truth
    # gt_center = data['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    # gt_heading_class = data['heading_class_label'].cpu().numpy() # B,K2
    # gt_heading_residual = data['heading_residual_label'].cpu().numpy() # B,K2
    # gt_size_class = data['size_class_label'].cpu().numpy() # B,K2
    # gt_size_residual = data['size_residual_label'].cpu().numpy() # B,K2,3
    # # reference
    # #gt_ref_labels = data["ref_box_label"].detach().cpu().numpy()
    # gt_ref_labels_list = data["ref_box_label_list"].detach().cpu().numpy()
    # objectness_preds_batch = torch.argmax(data['objectness_scores'], 2).long()
    # # objectness_labels_batch = data['objectness_label'].long()  # GT

    # TODO from EVAL
    Cal25 = APCalculator(ap_iou_thresh=0.25)
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

    objectness_preds_batch = torch.argmax(data['objectness_scores'], 2).long()
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

    gt_center_list = data['center_label'].cpu().numpy()
    gt_heading_class_list = data['heading_class_label'].cpu().numpy()
    gt_heading_residual_list = data['heading_residual_label'].cpu().numpy()
    gt_size_class_list = data['size_class_label'].cpu().numpy()
    gt_size_residual_list = data['size_residual_label'].cpu().numpy()

    gt_related_object = data['vqa_related_object_id']
    gt_answer = data['vqa_answer_id']
    test_type = data["test_type_id"]

    batch_size, lang_num_max, num_proposals = pred_related_object_confidence.shape[:3]
    lang_num = data["lang_num"]

    for i in range(batch_size):
        gt_obb_batch = config.param2obb_batch(gt_center_list[i][:, 0:3], gt_heading_class_list[i],
                                              gt_heading_residual_list[i],
                                              gt_size_class_list[i], gt_size_residual_list[i])
        pred_obb_batch = np.concatenate([pred_center, pred_box_size, pred_heading[:, :, None]], axis=-1)[i]

        gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
        pred_bbox_batch = get_3d_box_batch(pred_box_size[i], pred_heading[i], pred_center[i])

        gt_bbox_batch = torch.from_numpy(gt_bbox_batch).float().to(pred_masks.device)
        pred_bbox_batch = torch.from_numpy(pred_bbox_batch).float().to(pred_masks.device)

        # basic info
        idx = ids[i]
        scene_id = scanvqa[idx]["scene_id"]
        question = scanvqa[idx]['question'].replace(' ', '_').replace('?', '_')
        answer = scanvqa[idx]['answer'].replace(' ', '_')
        gt_answer_id = data['vqa_answer_id']
        pred_answer_id = data['vqa_pred_answer'][:, 0, :].max(-1)[1]
        pred_answer = SCANVQA_ANSWER_LIST[pred_answer_id[i]]

        assert lang_num[i] == 1, 'lang num == 1 (not support other value)'
        gt_ref_idx = scanvqa[idx]['related_object_id']

        thres = 0.5  # choose

        pred_related_batch = pred_related_object_confidence[i, 0]
        positive_mask = (pred_related_batch>thres) * pred_masks[i, :, None]  # pred_masks!

        pred_ref_idx = []
        # positive_mask = (positive_mask > 0)
        for _ in range(3):
            pred_ref_idx.append([x for x in range(256) if positive_mask[x, _]>0])
        pred_ref_idx.append(None)

        batch_gt_map_cls = []
        batch_pred_map_cls = []
        for x in range(3):
            if gt_ref_idx[x] is None:
                continue
            for y in gt_ref_idx[x]:
                batch_gt_map_cls.append((x, gt_bbox_batch[y].detach().cpu().numpy()))
            batch_pred_map_cls.extend([(x, pred_bbox_batch[i].detach().cpu().numpy(), pred_related_batch[i, x].detach().cpu().numpy())
                                       for i in range(positive_mask.shape[0]) if positive_mask[i, x]])
        Cal25.step([batch_pred_map_cls], [batch_gt_map_cls])
        mAP25_metrics = Cal25.compute_metrics()
        Cal25.reset()
        mAP = mAP25_metrics['mAP']

        # TODO CONTINUE
        if sum([len(x) for x in gt_ref_idx[:3] if x is not None]) == 0:  # gt no object
            continue
        # if answer != pred_answer:
        #     continue
        # if mAP < 0.4:
        #     continue

        # scene_output
        scene_dump_dir = os.path.join(dump_dir, scene_id)
        if not os.path.exists(scene_dump_dir):
            os.mkdir(scene_dump_dir)
            # # Dump the original scene point clouds
            mesh = align_mesh(scene_id)
            mesh.write(os.path.join(scene_dump_dir, 'mesh.ply'))
            # write_ply_rgb(point_clouds[i], pcl_color[i], os.path.join(scene_dump_dir, 'pc.ply'))  # do not use this

        # filter out the valid ground truth reference box
        #assert gt_ref_labels[i].shape[0] == gt_center[i].shape[0]
        #gt_ref_idx = np.argmax(gt_ref_labels[i], 0)
        # visualize the gt reference box
        # NOTE: for each object there should be only one gt reference box

        # print(mAP25_metrics, flush=True)

        if len(question) > 100:
            question = question[:100]

        object_dump_dir = os.path.join(dump_dir, scene_id, "gt_{}_{}_{}.ply".format(mAP, question, answer))
        print('dump object path', object_dump_dir, gt_ref_idx, flush=True)

        gt_obbs, pred_obbs = [], []
        for y in range(3):
            if gt_ref_idx[y] is not None:
                current_gt_obbs = [gt_obb_batch[x, :] for x in gt_ref_idx[y]]
                gt_obbs.append(current_gt_obbs)
            else:
                gt_obbs.append([])
            current_pred_obbs = [pred_obb_batch[x, :] for x in pred_ref_idx[y]]
            pred_obbs.append(current_pred_obbs)

        write_bboxes(gt_obbs, os.path.join(scene_dump_dir, 'gt_{:.2f}_{}_{}.ply'.format(mAP, question, answer)))
        write_bboxes(pred_obbs, os.path.join(scene_dump_dir, 'pred_{:.2f}_{}_{}.ply'.format(mAP, question, pred_answer)))
        # write_bbox(pred_obb, 1, os.path.join(scene_dump_dir, 'pred_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(object_id, object_name, ann_id, pred_ref_scores_softmax[i, pred_ref_idx], iou)))

def visualize(args):
    # init training dataset
    print("preparing data...")
    scanvqa_train, scanvqa_val, train_scene_list, val_scene_list, all_scene_list = get_scanvqa(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanvqa_val, val_scene_list, "val", DC)

    # model
    model = get_model(args)

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

    # evaluate
    print("visualizing...")
    for data in tqdm(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
        # feed
        with torch.no_grad():
            data = model(data)
            data = get_loss(
                data_dict=data,
                config=DC,
                detection=True,
                qa=True,
                use_lang_classifier=True
            )
            # data = get_eval(
            #     data=data, 
            #     config=DC, 
            #     reference=False,
            #     post_processing=POST_DICT
            # )
            dump_results(args, scanvqa_val, data, DC)

    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer_filtered of nr3d", default="ScanRefer_filtered")
    parser.add_argument("--folder", type=str, help="Folder containing the model", required=True)
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--scene_id", type=str, help="scene id", default="")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument('--num_points', type=int, default=40000, help='Point Number [default: 40000]')
    parser.add_argument('--num_proposals', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--num_scenes', type=int, default=-1, help='Number of scenes [default: -1]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--no_nms', action='store_true', help='do NOT use non-maximum suppression for post-processing.')
    parser.add_argument('--use_train', action='store_true', help='Use the training set.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_normal', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true', help='Use multiview images.')
    args = parser.parse_args()

    # # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    visualize(args)
