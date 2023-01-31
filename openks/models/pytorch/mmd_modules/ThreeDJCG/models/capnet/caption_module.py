import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
#sys.path.append(os.path.join(os.getcwd(), os.pardir, "openks/models/pytorch/mmd_modules/ThreeDJCG")) # HACK add the lib folder
from openks.models.pytorch.mmd_modules.ThreeDJCG.data.scannet.model_util_scannet import ScannetDatasetConfig
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.config_captioning import CONF
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.box_util import box3d_iou_batch_tensor
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.transformer.attention import MultiHeadAttention

# constants
DC = ScannetDatasetConfig()

def select_target(data_dict):
    # predicted bbox
    pred_bbox = data_dict["pred_bbox_corner"] # batch_size, num_proposals, 8, 3
    batch_size, num_proposals, _, _ = pred_bbox.shape

    # ground truth bbox
    gt_bbox = data_dict["ref_box_corner_label"].float() # batch_size, 8, 3

    target_ids = []
    target_ious = []
    for i in range(batch_size):
        # convert the bbox parameters to bbox corners
        pred_bbox_batch = pred_bbox[i] # num_proposals, 8, 3
        gt_bbox_batch = gt_bbox[i].unsqueeze(0).repeat(num_proposals, 1, 1) # num_proposals, 8, 3
        ious = box3d_iou_batch_tensor(pred_bbox_batch, gt_bbox_batch)
        target_id = ious.argmax().item() # 0 ~ num_proposals - 1
        target_ids.append(target_id)
        target_ious.append(ious[target_id])

    target_ids = torch.LongTensor(target_ids).cuda() # batch_size
    target_ious = torch.FloatTensor(target_ious).cuda() # batch_size

    return target_ids, target_ious

def select_multi_target(data_dict):
    # predicted bbox
    pred_bbox = data_dict["pred_bbox_corner"] # batch_size, num_proposals, 8, 3
    batch_size, num_proposals, box_size, _ = pred_bbox.shape

    # ground truth bbox
    gt_bbox = data_dict["ref_box_corner_label_list"].float() # batch_size, lang_num_max, 8, 3
    batch_size, len_nun_max, box_size = gt_bbox.shape[:3]
    # print("word_embs", word_embs[0][0][0][:5], word_embs[0][1][0][:5])
    gt_bbox = gt_bbox.reshape(batch_size * len_nun_max, box_size, -1) # batch_size * lang_num_max, 8, 3
    pred_bbox = pred_bbox[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1)\
        .reshape(batch_size * len_nun_max, num_proposals, box_size, -1) # batch_size * lang_num_max, num_proposals, 8, 3
    target_ids = []
    target_ious = []
    for i in range(batch_size * len_nun_max):
        # convert the bbox parameters to bbox corners
        pred_bbox_batch = pred_bbox[i] # num_proposals, 8, 3
        gt_bbox_batch = gt_bbox[i].unsqueeze(0).repeat(num_proposals, 1, 1) # num_proposals, 8, 3
        ious = box3d_iou_batch_tensor(pred_bbox_batch, gt_bbox_batch)

        target_id = ious.argmax().item()
        target_ious.append(ious[target_id].clone())
        """
        # 随机选取iou最大的五个物体中的一个
        num = random.randint(0, 4)
        for j in range(num):
           target_id = ious.argmax().item()
           if ious[target_id] >= 0.25:
               ious[target_id] = 0.
        target_id = ious.argmax().item()  # 0 ~ num_proposals - 1
        """
        target_ids.append(target_id)

    target_ids = torch.LongTensor(target_ids).cuda() # batch_size * lang_num_max
    target_ious = torch.FloatTensor(target_ious).cuda() # batch_size * lang_num_max

    return target_ids, target_ious


class SceneCaptionModule(nn.Module):
    def __init__(self, vocabulary, embeddings, emb_size=300, feat_size=128, hidden_size=512, num_proposals=256,
        num_locals=-1, query_mode="corner", use_oracle=False, head=4, depth=2):
        super().__init__()
        self.use_box_embedding = True
        self.use_dist_weight_matrix = True

        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])

        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_proposals = num_proposals
        self.depth = depth - 1

        self.num_locals = num_locals
        self.query_mode = query_mode

        self.use_oracle = use_oracle

        self.bbox_embedding = nn.Linear(27, 128) #12 27
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=feat_size, d_k=feat_size // head, d_v=feat_size // head, h=head) for i in
            range(depth))

        self.map_previous = nn.Sequential(
            nn.Linear(hidden_size + feat_size + emb_size, 128),  # emb_size hidden_size
            nn.ReLU()
        )

        # top-down attention module
        self.map_feat = nn.Linear(feat_size, hidden_size, bias=False)
        self.map_hidd = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attend = nn.Linear(hidden_size, 1, bias=False)

        # language recurrent module
        self.map_lang = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_size, self.num_vocabs)
        self.self_fc = nn.Sequential(  # 4 128 256 4(head)
            nn.Linear(4, 128),
            nn.Dropout(0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, 4)
        )

        self.attention_size = 128  # hidden_size
        self.dec_att2 = MultiHeadAttention(d_model=self.attention_size, d_k=self.attention_size // head,
                                           d_v=self.attention_size // head, h=head)
        self.obj_fc = nn.Linear(feat_size, self.attention_size)
        self.obj_dropout = nn.Dropout(p=.1)
        self.obj_layer_norm = nn.LayerNorm(self.attention_size)

    def _step(self, step_input, target_feat, obj_feats, hidden, object_masks):
        '''
            recurrent step

            Args:
                step_input: current word embedding, (batch_size, emb_size)
                target_feat: object feature of the target object, (batch_size, feat_size)
                obj_feats: object features of all detected objects, (batch_size, num_proposals, feat_size)
                hidden_1: hidden state of top-down recurrent unit, (batch_size, hidden_size)
                hidden_2: hidden state of language recurrent unit, (batch_size, hidden_size)

            Returns:
                hidden_1: hidden state of top-down recurrent unit, (batch_size, hidden_size)
                hidden_2: hidden state of language recurrent unit, (batch_size, hidden_size)
                masks: attention masks on proposals, (batch_size, num_proposals, 1)
        '''

        # fuse inputs (Previous Input)
        step_input = torch.cat([step_input, hidden, target_feat], dim=-1)
        step_input = self.map_previous(step_input)  # FC & ReLU

        # this fc could be taken out
        proposal_feats = F.relu(self.obj_fc(obj_feats))  # batch_size, n, hidden_size
        proposal_feats = self.obj_dropout(proposal_feats)
        proposal_feats = self.obj_layer_norm(proposal_feats)

        combined = self.map_feat(obj_feats)  # batch_size, num_proposals, hidden_size
        combined = torch.tanh(combined)
        # print("combined", combined.shape)

        # fuse inputs for language module
        lang_input = step_input  # single word
        # Attention
        if len(step_input.shape) == 2:
            lang_input = self.dec_att2(lang_input.unsqueeze(1), proposal_feats, proposal_feats).squeeze(1)
        else:
            lang_input = self.dec_att2(lang_input, proposal_feats, proposal_feats)

        lang_input = self.map_lang(lang_input)
        # The Recurrent Cell is No Use
        hidden = lang_input

        # Mask is no-use
        scores = self.attend(combined)  # batch_size, num_proposals, 1
        masks = F.softmax(scores, dim=1)  # batch_size, num_proposals, 1

        return hidden, masks

    def _nn_distance(self, pc1, pc2):
        """
        Input:
            pc1: (B,N,C) torch tensor
            pc2: (B,M,C) torch tensor

        Output:
            dist1: (B,N) torch float32 tensor
            idx1: (B,N) torch int64 tensor
            dist2: (B,M) torch float32 tensor
            idx2: (B,M) torch int64 tensor
        """

        N = pc1.shape[1]
        M = pc2.shape[1]
        pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
        pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
        pc_diff = pc1_expand_tile - pc2_expand_tile
        pc_dist = torch.sqrt(torch.sum(pc_diff**2, dim=-1) + 1e-8) # (B,N,M)

        return pc_dist
    
    def _get_bbox_centers(self, corners):
        coord_min = torch.min(corners, dim=2)[0] # batch_size, num_proposals, 3
        coord_max = torch.max(corners, dim=2)[0] # batch_size, num_proposals, 3

        return (coord_min + coord_max) / 2

    def _query_locals(self, data_dict, target_ids, object_masks, include_self=True, overlay_threshold=CONF.TRAIN.OVERLAID_THRESHOLD):
        corners = data_dict["pred_bbox_corner"] # batch_size, num_proposals, 8, 3
        centers = self._get_bbox_centers(corners) # batch_size, num_proposals, 3
        batch_size, _, _ = centers.shape

        # decode target box info
        target_centers = torch.gather(centers, 1, target_ids.view(-1, 1, 1).repeat(1, 1, 3)) # batch_size, 1, 3
        target_corners = torch.gather(corners, 1, target_ids.view(-1, 1, 1, 1).repeat(1, 1, 8, 3)) # batch_size, 1, 8, 3

        # get the distance
        if self.query_mode == "center":
            pc_dist = self._nn_distance(target_centers, centers).squeeze(1) # batch_size, num_proposals
        elif self.query_mode == "corner":
            pc_dist = self._nn_distance(target_corners.squeeze(1), centers) # batch_size, 8, num_proposals
            pc_dist, _ = torch.min(pc_dist, dim=1) # batch_size, num_proposals
        else:
            raise ValueError("invalid distance mode, choice: [\"center\", \"corner\"]")

        # mask out invalid objects
        pc_dist.masked_fill_(object_masks == 0, float('1e30')) # distance to invalid objects: infinity

        # exclude overlaid boxes
        tar2neigbor_iou = box3d_iou_batch_tensor(
            target_corners.repeat(1, self.num_proposals, 1, 1).view(-1, 8, 3), corners.view(-1, 8, 3)).view(batch_size, self.num_proposals) # batch_size, num_proposals
        overlaid_masks = tar2neigbor_iou >= overlay_threshold
        pc_dist.masked_fill_(overlaid_masks, float('1e30')) # distance to overlaid objects: infinity

        # include the target objects themselves
        self_dist = 0 if include_self else float('1e30')
        self_masks = torch.zeros(batch_size, self.num_proposals).cuda()
        self_masks.scatter_(1, target_ids.view(-1, 1), 1)
        pc_dist.masked_fill_(self_masks == 1, self_dist) # distance to themselves: 0 or infinity

        # get the top-k object ids
        _, topk_ids = torch.topk(pc_dist, self.num_locals, largest=False, dim=1) # batch_size, num_locals

        # construct masks for the local context
        local_masks = torch.zeros(batch_size, self.num_proposals).cuda()
        local_masks.scatter_(1, topk_ids, 1)

        return local_masks

    def _query_multi_locals(self, data_dict, target_ids, object_masks, include_self=True, overlay_threshold=CONF.TRAIN.OVERLAID_THRESHOLD):
        corners = data_dict["pred_bbox_corner"] # batch_size, num_proposals, 8, 3
        centers = self._get_bbox_centers(corners) # batch_size, num_proposals, 3
        batch_size, num_proposal, _ = centers.shape
        batch_size, len_nun_max = data_dict["lang_feat_list"].shape[:2]

        corners = corners[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1)\
            .reshape(batch_size * len_nun_max, num_proposal, 8, -1) # batch_size*len_nun_max, num_proposals, 8, 3
        centers = centers[:, None, :, :].repeat(1, len_nun_max, 1, 1)\
            .reshape(batch_size * len_nun_max, num_proposal, -1) # batch_size*len_nun_max, num_proposals, 3

        # decode target box info
        target_centers = torch.gather(centers, 1, target_ids.view(-1, 1, 1).repeat(1, 1, 3)) # batch_size*len_nun_max, 1, 3
        target_corners = torch.gather(corners, 1, target_ids.view(-1, 1, 1, 1).repeat(1, 1, 8, 3)) # batch_size*len_nun_max, 1, 8, 3

        # get the distance
        if self.query_mode == "center":
            pc_dist = self._nn_distance(target_centers, centers).squeeze(1) # batch_size*len_nun_max, num_proposals
        elif self.query_mode == "corner":
            pc_dist = self._nn_distance(target_corners.squeeze(1), centers) # batch_size*len_nun_max, 8, num_proposals
            pc_dist, _ = torch.min(pc_dist, dim=1) # batch_size*len_nun_max, num_proposals
        else:
            raise ValueError("invalid distance mode, choice: [\"center\", \"corner\"]")

        # mask out invalid objects
        pc_dist.masked_fill_(object_masks == 0, float('1e30')) # distance to invalid objects: infinity

        # exclude overlaid boxes
        tar2neigbor_iou = box3d_iou_batch_tensor(
            target_corners.repeat(1, self.num_proposals, 1, 1).view(-1, 8, 3), corners.view(-1, 8, 3)).view(-1, self.num_proposals) # batch_size*len_nun_max, num_proposals
        overlaid_masks = tar2neigbor_iou >= overlay_threshold
        pc_dist.masked_fill_(overlaid_masks, float('1e30')) # distance to overlaid objects: infinity

        # include the target objects themselves
        self_dist = 0 if include_self else float('1e30')
        self_masks = torch.zeros(batch_size*len_nun_max, self.num_proposals).cuda()
        self_masks.scatter_(1, target_ids.view(-1, 1), 1)
        pc_dist.masked_fill_(self_masks == 1, self_dist) # distance to themselves: 0 or infinity

        # get the top-k object ids
        _, topk_ids = torch.topk(pc_dist, self.num_locals, largest=False, dim=1) # batch_size, num_locals

        # construct masks for the local context
        local_masks = torch.zeros(batch_size*len_nun_max, self.num_proposals).cuda()
        local_masks.scatter_(1, topk_ids, 1)

        return local_masks

    def forward(self, data_dict, is_eval=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        if not is_eval:
            data_dict = self._forward_sample_batch(data_dict, max_len)
        else:
            data_dict = self._forward_scene_batch(data_dict, max_len)

        return data_dict

    def _create_adjacent_mat(self, data_dict, object_masks):
        batch_size, num_objects = object_masks.shape
        adjacent_mat = torch.zeros(batch_size, num_objects, num_objects).cuda()

        for obj_id in range(num_objects):
            target_ids = torch.LongTensor([obj_id for _ in range(batch_size)]).cuda()
            adjacent_entry = self._query_locals(data_dict, target_ids, object_masks, include_self=False) # batch_size, num_objects
            adjacent_mat[:, obj_id] = adjacent_entry

        return adjacent_mat

    def _get_valid_object_masks(self, data_dict, target_ids, object_masks):
        if self.num_locals == -1:
            valid_masks = object_masks
        else:
            adjacent_mat = data_dict["adjacent_mat"]
            batch_size, _, _ = adjacent_mat.shape
            valid_masks = torch.gather(
                adjacent_mat, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, self.num_proposals)).squeeze(1) # batch_size, num_proposals

        return valid_masks

    def _add_relation_feat(self, data_dict, obj_feats, target_ids):
        rel_feats = data_dict["edge_feature"] # batch_size, num_proposals, num_locals, feat_size
        batch_size = rel_feats.shape[0]

        rel_feats = torch.gather(rel_feats, 1, 
            target_ids.view(batch_size, 1, 1, 1).repeat(1, 1, self.num_locals, self.feat_size)).squeeze(1) # batch_size, num_locals, feat_size

        # new_obj_feats = torch.cat([obj_feats, rel_feats], dim=1) # batch_size, num_proposals + num_locals, feat_size

        # scatter the relation features to objects
        adjacent_mat = data_dict["adjacent_mat"] # batch_size, num_proposals, num_proposals
        rel_indices = torch.gather(adjacent_mat, 1, 
            target_ids.view(batch_size, 1, 1).repeat(1, 1, self.num_proposals)).squeeze(1) # batch_size, num_proposals
        rel_masks = rel_indices.unsqueeze(-1).repeat(1, 1, self.feat_size) == 1 # batch_size, num_proposals, feat_size
        scattered_rel_feats = torch.zeros(obj_feats.shape).cuda().masked_scatter(rel_masks, rel_feats) # batch_size, num_proposals, feat_size

        new_obj_feats = obj_feats + scattered_rel_feats
        # new_obj_feats = torch.cat([obj_feats, scattered_rel_feats], dim=-1)
        # new_obj_feats = self.map_rel(new_obj_feats)

        return new_obj_feats

    def _expand_object_mask(self, data_dict, object_masks, num_extra):
        batch_size, num_objects = object_masks.shape
        exp_masks = torch.zeros(batch_size, num_extra).cuda()

        num_edge_targets = data_dict["num_edge_target"]
        for batch_id in range(batch_size):
            exp_masks[batch_id, :num_edge_targets[batch_id]] = 1

        object_masks = torch.cat([object_masks, exp_masks], dim=1) # batch_size, num_objects + num_extra

        return object_masks

    def get_local_feat(self, data_dict, obj_feats, object_masks):
        batch_size, num_proposal, feature_size = obj_feats.shape
        obj_masks = object_masks.bool().view(batch_size * num_proposal)  # batch_size * num_proposals
        obj_masks = torch.where(obj_masks[:] == True)[0]
        """
        if data_dict["istrain"][0] == 1:
            obj_masks = obj_masks.view(batch_size, self.num_locals)
            num = self.num_locals//2 - 1
            for i in range(batch_size):
                rand = random.random()
                if rand < 0.2:
                    for j in range(num):
                        obj_masks[i, j*2] = num_proposal - obj_masks[i, j*2] - 1
                elif rand < 0.4:
                    for j in range(num):
                        obj_masks[i, j*2+1] = num_proposal - obj_masks[i, j*2+1] - 1
            obj_masks = obj_masks.view(batch_size*self.num_locals)
        """
        obj_features = obj_feats.view(batch_size * num_proposal, feature_size)
        obj_features = obj_features[obj_masks, :]  # batch_size * num_local, feature_size
        obj_features = obj_features.view(batch_size, self.num_locals, feature_size)

        return obj_features

    def target_feat_aug(self, data_dict, target_ids, target_feats, obj_feats):
        B, num_proposal, obj_feature_size = obj_feats.shape  # batch_size * len_nun_max, num_proposal, 128
        len_nun_max = data_dict["lang_feat_list"].shape[1]
        batch_size = B // len_nun_max

        obj_feats = obj_feats.view(batch_size, len_nun_max, num_proposal, obj_feature_size)
        pred_cluster_labels = target_ids.view(batch_size, len_nun_max, 1)  # B len_nun_max 1
        """
        sem_cls_scores = data_dict['sem_cls_scores']  # batch_size, num_proposal, 18
        sem_cls_scores = sem_cls_scores[:, None, :, :].repeat(1, len_nun_max, 1, 1)  # batch_size, len_nun_max, num_proposal, 18
        pred_obj_cls_scores = torch.gather(sem_cls_scores, 2, pred_cluster_labels.unsqueeze(3).repeat(1, 1, 1, self.num_class)).squeeze(2)  # B len_nun_max 18
        pred_obj_cls = torch.argmax(pred_obj_cls_scores, 2).unsqueeze(2)  # batch_size, len_nun_max 1

        select_sem_cls_scores = torch.gather(sem_cls_scores, 3, pred_obj_cls.unsqueeze(3).repeat(1, 1, num_proposal, 1)).squeeze(3)  # B len_nun_max num_proposal
        _, select_cls_scores_idx = torch.topk(select_sem_cls_scores, k=self.k, dim=-1, largest=True, sorted=False)  # B len_nun_max k
        new_object_feats = torch.gather(obj_feats, 2, select_cls_scores_idx.unsqueeze(3).repeat(1, 1, 1, obj_feature_size))  # B len_nun_max k 128
        new_object_feats = new_object_feats.view(batch_size * len_nun_max, self.k, obj_feature_size)
        """
        center = data_dict['center']  # batch_size, num_proposal, 3
        center = center[:, None, :, :].repeat(1, len_nun_max, 1, 1)  # batch_size, len_nun_max, num_proposal, 3
        pred_obj_center = torch.gather(center, 2, pred_cluster_labels.unsqueeze(3).repeat(1, 1, 1, 3)).view(batch_size * len_nun_max, 1, -1)  # B*len_nun_max 1 3
        center = center.view(batch_size * len_nun_max, num_proposal, -1)
        dist, ind = knn_distance(center[:, :, 0:3], pred_obj_center[:, :, 0:3], k=self.k)  # ind: B*len_nun_max 1 k
        ind = ind.permute(0, 2, 1).contiguous()  # B*len_nun_max k

        obj_features_reshape = obj_feats.view(batch_size * len_nun_max, num_proposal, -1)
        obj_knn = torch.gather(obj_features_reshape.unsqueeze(2).repeat(1, 1, self.k, 1), 1,
                               ind.unsqueeze(3).repeat(1, 1, 1, obj_feature_size))
        new_object_feats = obj_knn.view(batch_size * len_nun_max, self.k, -1)

        T = 0.7
        for i in range(batch_size * len_nun_max):
            num = random.randint(0, 9)
            if num < self.k:
                target_feats[i] = new_object_feats[i, num, :]
                #target_feats[i] = target_feats[i]*T + new_object_feats[i, num, :]*(1-T)
                """
                rand = random.random()
                if rand < 0.5:
                    target_feats[i, :obj_feature_size // 2] = new_object_feats[i, num, :obj_feature_size // 2]
                else:
                    target_feats[i, obj_feature_size // 2:] = new_object_feats[i, num, obj_feature_size // 2:]
                """

        return data_dict, target_feats

    def _forward_sample_batch(self, data_dict, max_len=CONF.TRAIN.MAX_DES_LEN, min_iou=CONF.TRAIN.MIN_IOU_THRESHOLD):
        """
            generate descriptions based on input tokens and object features
        """

        # unpack
        #word_embs = data_dict["lang_feat"] # batch_size, max_len, max_len
        #des_lens = data_dict["lang_len"] # batch_size

        word_embs = data_dict["lang_feat_list"]  # batch_size, lang_num_max, max_len, max_len
        des_lens = data_dict["lang_len_list"]
        batch_size, len_nun_max, max_des_len = word_embs.shape[:3]
        word_embs = word_embs.reshape(batch_size * len_nun_max, max_des_len, -1) # batch_size * lang_num_max, max_len, max_len
        des_lens = des_lens.reshape(batch_size * len_nun_max)
        first_obj = data_dict["first_obj_list"].reshape(batch_size * len_nun_max)
        if data_dict["istrain"][0] == 1 and random.random() < 0.5:
            for i in range(word_embs.shape[0]):
                word_embs[i, first_obj] = data_dict["unk"][0]
        """
                len = des_lens[i]
                for j in range(int(len/5)):
                    num = random.randint(0, len-1)
                    word_embs[i, num] = data_dict["unk"][0]
        elif data_dict["istrain"][0] == 1:
            for i in range(word_embs.shape[0]):
                len = des_lens[i]
                for j in range(int(len/5)):
                    num = random.randint(0, len-1)
                    word_embs[i, num] = data_dict["unk"][0]
        """

        #obj_feats = data_dict["pred_bbox_feature"] # batch_size, num_proposals, feat_size
        object_masks = data_dict["pred_bbox_mask"] # batch_size, num_proposals  

        obj_feats = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size
        dist_weights = data_dict['dist_weights']
        attention_matrix_way = data_dict['attention_matrix_way']

        features = obj_feats

        batch_size, num_proposal = features.shape[:2]

        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1

        len_nun_max = data_dict["lang_feat_list"].shape[1]

        # objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
        data_dict["random"] = random.random()

        feature1 = features[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size * len_nun_max, num_proposal,
                                                                                -1)
        if dist_weights is not None:
            dist_weights = dist_weights[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1).reshape(
                batch_size * len_nun_max, dist_weights.shape[1], num_proposal, num_proposal)

        obj_feats = feature1
        object_masks = object_masks[:, None, :].repeat(1, len_nun_max, 1).reshape(batch_size * len_nun_max, num_proposal)
        #################################caption########################################
        num_words = des_lens.max()
        # batch_size = des_lens.shape[0]

        # find the target object ids
        if self.use_oracle:
            target_ids = data_dict["bbox_idx"]  # batch_size
            target_ious = torch.ones(batch_size).cuda()
        else:
            # target_ids, target_ious = select_target(data_dict)
            target_ids, target_ious = select_multi_target(data_dict)  # batch_size * len_nun_max
        # print("obj_feats1", obj_feats.shape)
        # print("target_ids1", target_ids.shape)
        # print("target_ids2", target_ids.view(batch_size*len_nun_max, 1, 1).repeat(1, 1, self.feat_size).shape)
        # select object features
        target_feats = torch.gather(obj_feats, 1, target_ids.view(batch_size*len_nun_max, 1, 1).repeat(1, 1, self.feat_size)).squeeze(1) # batch_size * len_nun_max, emb_size

        # valid object proposal masks
        #valid_masks = object_masks if self.num_locals == -1 else self._query_locals(data_dict, target_ids, object_masks)
        valid_masks = object_masks if self.num_locals == -1 else self._query_multi_locals(data_dict, target_ids, object_masks)
        #print("valid_masks", valid_masks.shape, valid_masks[0], valid_masks[0].sum())
        #print("object_masks", object_masks.shape, object_masks[0], object_masks[0].sum())

        # Todo attention caption

        obj_feats = self.get_local_feat(data_dict, obj_feats, valid_masks)
        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []
        hidden = torch.zeros(batch_size * len_nun_max, self.hidden_size).cuda()  # batch_size*len_nun_max, hidden_size
        step_id = 0
        step_input = word_embs[:, step_id]  # batch_size*len_nun_max, emb_size

        # output_rand, step_input_rand = [], step_input[:, None, :].repeat(self.search_length)
        while True:
            # feed; The Top-Module is not useful
            hidden, step_mask = self._step(step_input, target_feats, obj_feats, hidden,
                                                       valid_masks.unsqueeze(-1))
            step_output = self.classifier(hidden)  # batch_size*len_nun_max, num_vocabs

            # store
            step_output = step_output.unsqueeze(1) # batch_size*len_nun_max, 1, num_vocabs
            outputs.append(step_output)
            masks.append(step_mask) # batch_size*len_nun_max, num_proposals, 1

            # next step
            step_id += 1
            if step_id == num_words - 1: break  # exit for train mode

            # step_input = word_embs[:, step_id]
            rand2 = random.random()
            unk = data_dict["unk"][0]
            #  predword3
            if (data_dict["epoch"] < 20 and rand2 < 0.1) or (data_dict["epoch"] >= 20 and rand2 < 0.2):
                # predicted word
                step_preds = []
                for batch_id in range(batch_size * len_nun_max):
                    idx = step_output[batch_id].argmax()  # 0 ~ num_vocabs
                    word = self.vocabulary["idx2word"][str(idx.item())]
                    if word in self.embeddings.keys():
                        emb = torch.FloatTensor(self.embeddings[word]).cuda()
                    else: # special token
                        emb = torch.zeros(self.embeddings['none'].shape).cuda()
                    # emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda()  # 1, emb_size
                    step_preds.append(emb[None, :])
                step_preds = torch.cat(step_preds, dim=0)  # batch_size*len_nun_max, emb_size
                step_input = step_preds
            elif (data_dict["epoch"] < 20 and rand2 < 0.3) or (data_dict["epoch"] >= 20 and rand2 < 0.4):
                step_input = unk.unsqueeze(0).repeat(batch_size * len_nun_max, 1)
            else:
                step_input = word_embs[:, step_id]  # batch_size*len_nun_max, emb_size

            # RL Policy Gradient
            # TODO

        outputs = torch.cat(outputs, dim=1)  # batch_size*len_nun_max, num_words - 1/max_len, num_vocabs
        masks = torch.cat(masks, dim=-1)  # batch_size*len_nun_max, num_proposals, num_words - 1/max_len

        # NOTE when the IoU of best matching predicted boxes and the GT boxes
        # are smaller than the threshold, the corresponding predicted captions
        # should be filtered out in case the model learns wrong things
        good_bbox_masks = target_ious > min_iou # batch_size*len_nun_max

        num_good_bboxes = good_bbox_masks.sum()
        mean_target_ious = target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()
        #print("outputs", outputs.shape)
        #print("mean_target_ious", mean_target_ious.shape)
        #print("masks", masks.shape)
        #print("valid_masks", valid_masks.shape)
        #print("good_bbox_masks", good_bbox_masks.shape)
        # store
        data_dict["lang_cap"] = outputs
        data_dict["pred_ious"] = mean_target_ious
        data_dict["topdown_attn"] = masks
        data_dict["valid_masks"] = valid_masks
        data_dict["good_bbox_masks"] = good_bbox_masks

        target_ious = target_ious.reshape(batch_size, len_nun_max)
        lang_num = data_dict["lang_num"]
        max_iou_rate_25 = 0
        max_iou_rate_5 = 0
        for i in range(batch_size):
            for j in range(len_nun_max):
                if j < lang_num[i]:
                    if target_ious[i, j] >= 0.25:
                        max_iou_rate_25 += 1
                    if target_ious[i, j] >= 0.5:
                        max_iou_rate_5 += 1
        data_dict['max_iou_rate_0.25'] = max_iou_rate_25 / sum(lang_num.cpu().numpy())
        data_dict['max_iou_rate_0.5'] = max_iou_rate_5 / sum(lang_num.cpu().numpy())

        return data_dict

    def _forward_scene_batch(self, data_dict, max_len=CONF.TRAIN.MAX_DES_LEN):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"] # batch_size, emb_size

        # valid object proposal masks
        #obj_feats = data_dict["pred_bbox_feature"]  # batch_size, num_proposals, feat_size
        object_masks = data_dict["pred_bbox_mask"]  # batch_size, num_proposals

        obj_feats = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size
        dist_weights = data_dict['dist_weights']
        attention_matrix_way = data_dict['attention_matrix_way']

        features = obj_feats
        batch_size, num_proposal = features.shape[:2]

        # print("feature1", feature1.shape)
        # print("object_masks", object_masks.shape)
        obj_feats = features
        ###############################caption#######################################
        # # create adjacency matrices
        # if self.num_locals != -1 and "adjacent_mat" not in data_dict:
        #     adjacent_mat = self._create_adjacent_mat(data_dict, object_masks)
        #     data_dict["adjacent_mat"] = adjacent_mat

        # # include self to adjacency matrices
        # identity = torch.eye(self.num_proposals).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        # data_dict["adjacent_mat"] += identity # include self

        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []
        valid_masks = []
        for prop_id in range(self.num_proposals):
            # select object features
            target_feats = obj_feats[:, prop_id] # batch_size, emb_size
            target_ids = torch.zeros(batch_size).fill_(prop_id).long().cuda()

            prop_obj_feats = obj_feats.clone()
            # valid_prop_masks = self._get_valid_object_masks(data_dict, target_ids, object_masks)
            valid_prop_masks = object_masks if self.num_locals == -1 else self._query_locals(data_dict, target_ids, object_masks)

            valid_masks.append(valid_prop_masks.unsqueeze(1))

            # start recurrence
            prop_outputs = []
            prop_masks = []
            hidden = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
            step_id = 0
            # step_input = word_embs[:, step_id] # batch_size, emb_size
            step_input = word_embs[:, 0] # batch_size, emb_size
            while True:
                # feed
                hidden, step_mask = self._step(step_input, target_feats, prop_obj_feats, hidden,
                                                           valid_prop_masks.unsqueeze(-1))
                step_output = self.classifier(hidden)  # batch_size, num_vocabs

                # predicted word
                step_preds = []
                for batch_id in range(batch_size):
                    idx = step_output[batch_id].argmax() # 0 ~ num_vocabs
                    word = self.vocabulary["idx2word"][str(idx.item())]
                    emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda() # 1, emb_size
                    step_preds.append(emb)

                step_preds = torch.cat(step_preds, dim=0) # batch_size, emb_size

                # store
                step_output = step_output.unsqueeze(1) # batch_size, 1, num_vocabs 
                prop_outputs.append(step_output)
                prop_masks.append(step_mask)

                # next step
                step_id += 1
                if step_id == max_len - 1: break # exit for eval mode
                step_input = step_preds # batch_size, emb_size

            prop_outputs = torch.cat(prop_outputs, dim=1).unsqueeze(1) # batch_size, 1, num_words - 1/max_len, num_vocabs
            prop_masks = torch.cat(prop_masks, dim=-1).unsqueeze(1) # batch_size, 1, num_proposals, num_words - 1/max_len
            outputs.append(prop_outputs)
            masks.append(prop_masks)

        outputs = torch.cat(outputs, dim=1) # batch_size, num_proposals, num_words - 1/max_len, num_vocabs
        masks = torch.cat(masks, dim=1) # batch_size, num_proposals, num_proposals, num_words - 1/max_len
        valid_masks = torch.cat(valid_masks, dim=1) # batch_size, num_proposals, num_proposals

        # store
        data_dict["lang_cap"] = outputs
        data_dict["topdown_attn"] = masks
        data_dict["valid_masks"] = valid_masks
        data_dict['max_iou_rate_0.25'] = 0.
        data_dict['max_iou_rate_0.5'] = 0.
        return data_dict
