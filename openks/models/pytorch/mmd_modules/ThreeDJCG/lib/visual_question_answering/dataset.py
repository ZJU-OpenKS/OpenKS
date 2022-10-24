'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
Changed: Lichen Zhao (zlc1114@buaa.edu.cn)
'''

import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset
import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.config_vqa import CONF
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.pc_utils import random_sampling, rotx, roty, rotz
from openks.models.pytorch.mmd_modules.ThreeDJCG.data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis
import random

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class ScanVQADataset(Dataset):

    def __init__(self, scanvqa_data, scanvqa_all_scene,
        answer_type=None,
        split="train",
        num_points=40000,
        lang_num_max=32,
        use_height=False,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        augment=False,
        shuffle=False):

        self.answer_type = answer_type
        self.scanvqa = scanvqa_data
        self.scanvqa_all_scene = scanvqa_all_scene # all scene_ids in scanvqa
        self.split = split
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.augment = augment
        self.lang_num_max = lang_num_max

        # load data
        self._load_data()
        self.multiview_data = {}
        self.should_shuffle = shuffle
        self.scanvqa_new_len = -1
        self.shuffle_data()

    def __len__(self):
        return self.scanvqa_new_len


    def split_scene_new(self, scanvqa_data):
        scanvqa_train_new = []
        scanvqa_train_new_scene, scanvqa_train_scene = [], []
        scene_id = ''
        lang_num_max = self.lang_num_max
        for data in scanvqa_data:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanvqa_train_scene) > 0:
                    if self.should_shuffle:
                        random.shuffle(scanvqa_train_scene)
                    # print("scanvqa_train_scene", len(scanvqa_train_scene))
                    for new_data in scanvqa_train_scene:
                        if len(scanvqa_train_new_scene) >= lang_num_max:
                            scanvqa_train_new.append(scanvqa_train_new_scene)
                            scanvqa_train_new_scene = []
                        scanvqa_train_new_scene.append(new_data)
                    if len(scanvqa_train_new_scene) > 0:
                        scanvqa_train_new.append(scanvqa_train_new_scene)
                        scanvqa_train_new_scene = []
                    scanvqa_train_scene = []
            scanvqa_train_scene.append(data)
        if len(scanvqa_train_scene) > 0:
            if self.should_shuffle:
                random.shuffle(scanvqa_train_scene)
            # print("scanvqa_train_scene", len(scanvqa_train_scene))
            for new_data in scanvqa_train_scene:
                if len(scanvqa_train_new_scene) >= lang_num_max:
                    scanvqa_train_new.append(scanvqa_train_new_scene)
                    scanvqa_train_new_scene = []
                scanvqa_train_new_scene.append(new_data)
            if len(scanvqa_train_new_scene) > 0:
                scanvqa_train_new.append(scanvqa_train_new_scene)
                scanvqa_train_new_scene = []
        return scanvqa_train_new


    def shuffle_data(self):
        print(self.split, ': re-build dataset data(lang), should_shuffle=', self.should_shuffle, flush=True)
        self.scanvqa_new = self.split_scene_new(self.scanvqa)
        if self.should_shuffle:
            random.shuffle(self.scanvqa_new)
        if self.scanvqa_new_len == -1:
            self.scanvqa_new_len = len(self.scanvqa_new)
        assert len(self.scanvqa_new) == self.scanvqa_new_len, 'assert scanvqa length right'
        print(self.split, ': build dataset done', flush=True)


    def __getitem__(self, idx):
        start = time.time()

        # split the dict
        lang_num = len(self.scanvqa_new[idx])
        scene_id = self.scanvqa_new[idx][0]["scene_id"]

        # scanvqa dataset info
        related_object_id_list = []  # fc
        related_object_sem_id_list = []
        question_id_list = []
        question_list = []
        question_embedding_list = []
        question_embedding_length_list = []
        answer_id_list = []
        answer_list = []
        test_type_id_list = []
        unique_multiple_list = []
        object_cat_list = []

        for i in range(self.lang_num_max):
            if i < lang_num:
                related_object_id = self.scanvqa_new[idx][i]['related_object_id']
                related_object_sem_id = self.scanvqa_new[idx][i]['related_object_sem_id']
                question_type_id = self.scanvqa_new[idx][i]['question_type_id']
                question_embedding = self.scanvqa_new[idx][i]['question_embedding']
                question_embedding_len = self.scanvqa_new[idx][i]['question_embedding_len']
                question = self.scanvqa_new[idx][i]['question']
                test_type_id = self.scanvqa_new[idx][i]['test_type_id']
                # add question prefix
                question_prefix = self.scanvqa_new[idx][i]['question_type']
                if question_prefix not in ['grounding']:
                    question_prefix = 'visual question answering'
                question = question_prefix + ': ' + question
                answer_id = self.scanvqa_new[idx][i]['answer_id']
                answer = self.scanvqa_new[idx][i]['answer']
                unique_multiple_id = self.scanvqa_new[idx][i].get('unique_multiple', None)
                object_cat = self.scanvqa_new[idx][i].get('object_cat', None)

            related_object_id_list.append(related_object_id)
            related_object_sem_id_list.append(related_object_sem_id)
            question_id_list.append(question_type_id)
            question_embedding_list.append(question_embedding)
            question_embedding_length_list.append(question_embedding_len)
            question_list.append(question)
            answer_id_list.append(answer_id)
            answer_list.append(answer)
            test_type_id_list.append(test_type_id)
            unique_multiple_list.append(unique_multiple_id)
            object_cat_list.append(object_cat)
        question_id_list = torch.stack([torch.tensor(v) for v in question_id_list])
        question_embedding_list = torch.stack([torch.from_numpy(v) for v in question_embedding_list]).float()
        question_embedding_length_list = torch.from_numpy(np.array(question_embedding_length_list))

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6]
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]

        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            pid = 'pid'
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")
            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------
            if self.augment:  # and not self.debug: # shape not changed;
                if np.random.random() < 0.3:
                    # Flipping along the YZ plane
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if np.random.random() < 0.3:
                    # Flipping along the XZ plane
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                    # Rotation along X-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

                # Rotation along Y-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

                # print('Warning! Dont Use Extra Augmentation!(votenet didnot use it)', flush=True)
                # NEW: scale from 0.8 to 1.2
                # print(rot_mat.shape, point_cloud.shape, flush=True)
                scale = np.random.uniform(-0.1, 0.1, (3, 3))
                scale = np.exp(scale)
                # print(scale, '<<< scale', flush=True)
                scale = scale * np.eye(3)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], scale)
                if self.use_height:
                    point_cloud[:, 3] = point_cloud[:, 3] * float(scale[2, 2])
                target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], scale)
                target_bboxes[:, 3:6] = np.dot(target_bboxes[:, 3:6], scale)

                # Translation
                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            for i_instance in np.unique(instance_labels):
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical

            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]

            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

            # # construct the reference target label for each bbox
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            print('[Dataset] Warning: target bboxes sem_cls Not Right', flush=True)
            pass

        istrain = 1 if self.split == "train" else 0

        # return data_dict
        data_dict = {}

        data_dict["istrain"] = istrain
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features

        # votenet data
        data_dict["scan_idx"] = idx
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)

        # data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color

        data_dict["load_time"] = time.time() - start

        # related_object_id_list = []  # fc
        # related_object_sem_id_list = []
        # question_id_list = []
        # question_list = []
        # answer_id_list = []
        # answer_list = []
        data_dict["vqa_related_object_id"] = related_object_id_list
        data_dict["vqa_related_object_sem_id"] = related_object_sem_id_list
        data_dict["vqa_question_id"] = question_id_list
        data_dict["vqa_question_embedding"] = question_embedding_list
        data_dict["vqa_question_embedding_length"] = question_embedding_length_list
        data_dict["vqa_question"] = question_list
        data_dict["vqa_answer_id"] = answer_id_list
        data_dict["vqa_answer"] = answer_list
        data_dict["test_type_id"] = test_type_id_list

        data_dict["lang_num"] = np.array(lang_num).astype(np.int64)

        # # unique_multiple_list: number of same-class objects
        # unique_multiple_list = []
        # for i in range(self.lang_num_max):
        #     object_id = object_id_list[i]
        #     ann_id = ann_id_list[i]
        #     unique_multiple = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]
        #     unique_multiple_list.append(unique_multiple)
        data_dict["unique_multiple_list"] = unique_multiple_list
        data_dict['object_cat_list'] = object_cat_list

        return data_dict

    def _get_raw2label(self):
        # maepping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]
        raw2label["shower_curtain"] = 13

        return raw2label

    def _scanvqa_preprocess(self):
        question_type = []
        question_number = []
        test_num = np.zeros((4), dtype=np.int32)
        answer_type = self.answer_type
        answer_number = [0 for i in answer_type]
        # answer_type = []  # should be initialized individually
        # answer_number = []
        related_keys = ['related_object(type 1)', 'related_object(type 2)', 'related_object(type 3)', 'related_object(type 4)']
        # TODO Some data does not have related_object(type...)
        for id_q, data in enumerate(self.scanvqa):
            scene_id = data["scene_id"]
            related_object = []
            for key in related_keys:
                related_object.append(data.get(key, None))
            self.scanvqa[id_q]['related_object'] = related_object
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            related_object_id = []
            related_object_sem_id = []
            for objs in related_object:
                if objs is None:
                    related_object_id.append(None)
                    related_object_sem_id.append(None)
                    continue
                object_id, object_sem_id = [], []
                for id_b, bbox in enumerate(instance_bboxes):
                    if int(bbox[-1]) in objs:
                        object_id.append(id_b)
                        object_sem_id.append(DC.nyu40id2class[int(bbox[-2])])  # nyu40id to scannetid
                related_object_id.append(object_id)
                related_object_sem_id.append(object_sem_id)
            self.scanvqa[id_q]['related_object_id'] = related_object_id
            self.scanvqa[id_q]['related_object_sem_id'] = related_object_sem_id
            if data['question_type'] not in question_type:
                question_type.append(data['question_type'])
                question_number.append(0)
            question_type_id = question_type.index(data['question_type'])
            self.scanvqa[id_q]['question_type_id'] = question_type_id
            question_number[question_type_id] += 1

            if data['answer'] not in answer_type:
                answer_type.append(data['answer'])
                answer_number.append(0)
            answer_id = answer_type.index(data['answer'])
            self.scanvqa[id_q]['answer_id'] = answer_id
            answer_number[answer_id] += 1

            if data['question_type'] == "how many":
                self.scanvqa[id_q]['test_type_id'] = 0
                test_num[0] += 1
                #print("how many", data['answer'])
            elif data['question_type'] == "what color":
                self.scanvqa[id_q]['test_type_id'] = 1
                test_num[1] += 1
                #print("what color", data['answer'])
            elif data['answer'] == "yes" or data['answer'] == "no":
                self.scanvqa[id_q]['test_type_id'] = 2
                test_num[2] += 1
                #print("yes/no", data['answer'])
            else:
                test_num[3] += 1
                self.scanvqa[id_q]['test_type_id'] = 3

            # get scanrefer unique-multiple-label
            if 'object_name' in data.keys():
                scene_id = data["scene_id"]
                object_id = data["object_id"]
                ann_id = data["ann_id"]
                object_name = ' '.join(data['object_name'].split('_'))
                self.scanvqa[id_q]["object_cat"]=self.raw2label[object_name] if object_name in self.raw2label else 17
                self.scanvqa[id_q]["unique_multiple"] = self.unique_multiple_lookup[scene_id][object_id][ann_id]

        # remove the QA pairs in the initial dataset
        for id_q in range(len(self.scanvqa)):
            for val in related_keys:
                if val in self.scanvqa[id_q].keys():
                    del self.scanvqa[id_q][val]

        self.question_type = question_type
        self.question_number = question_number
        print(len(question_type), 'vqa question types:', question_type, question_number)
        self.answer_type = answer_type
        self.answer_number = answer_number
        print(len(answer_type), 'vqa answer types:', answer_type, answer_number)

        print("test_num:", test_num)

    def _tokenize_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)
        for id_q, data in enumerate(self.scanvqa):
            # tokenize the description
            tokens = data["question"].split(' ')
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
            for token_id in range(CONF.TRAIN.MAX_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove["pad"]
            # store
            self.scanvqa[id_q]['question_embedding'] = embeddings
            self.scanvqa[id_q]['question_embedding_len'] = len(tokens)

    def _load_data(self):
        print("loading data...")

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanvqa])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            # instance_bboxes: [center(x/y/z), distance(dx/dy/dz), sem_cls, obj_id]
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()

        # for scanrefer use
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()  # for ScanRefer use

        # dataset utils
        self._scanvqa_preprocess()

        # load language features
        self._tokenize_des()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]

        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox

    def collate_fn(self, batch):
        res = {}
        for key in batch[0].keys():
            value = [b[key] for b in batch]
            if isinstance(value[0], torch.Tensor):
                res[key] = torch.stack(value)
            elif isinstance(value[0], np.ndarray):
                res[key] = torch.stack([torch.from_numpy(v) for v in value])
            elif isinstance(value[0], (int, float)):
                res[key] = torch.from_numpy(np.array(value))
            elif isinstance(value[0], str):
                res[key] = value
            elif isinstance(value[0], list) and isinstance(value[0][0], str):  # question & answer
                res[key] = value
            elif isinstance(value[0], list):  # related objects
                res[key] = value
            else:
                raise NotImplementedError(key, value)
        return res


    def _get_unique_multiple_lookup(self):  # this function is only for scanrefer use
        all_sem_labels = {}
        cache = {}
        for data in self.scanvqa:
            scene_id = data["scene_id"]
            if 'object_name' not in data.keys():
                continue
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]
            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []
            if scene_id not in cache:
                cache[scene_id] = {}
            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)
        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanvqa:
            scene_id = data["scene_id"]
            if 'object_name' not in data.keys():
                continue
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]
            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17
            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1
            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}
            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}
            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None
            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

