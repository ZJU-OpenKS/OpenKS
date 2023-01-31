# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
# sys.path.append(BASE_DIR)  # DETR

from ..lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from easydict import EasyDict
from ..models.detr.detr3d import DETR3D
from ..models.detr.transformer3D import decode_scores_boxes
from ..utils.box_util import get_3d_box_batch


def decode_scores_classes(output_dict, end_points, num_class):
    pred_logits = output_dict['pred_logits']
    assert pred_logits.shape[-1] == 2+num_class, 'pred_logits.shape wrong'
    objectness_scores = pred_logits[:,:,0:2]  # TODO CHANGE IT; JUST SOFTMAXd
    end_points['objectness_scores'] = objectness_scores
    sem_cls_scores = pred_logits[:,:,2:2+num_class] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


def decode_dataset_config(data_dict, dataset_config):
    if dataset_config is not None:
        # print('decode_dataset_config', flush=True)
        pred_center = data_dict['center'].detach().cpu().numpy()  # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
                                             pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
        pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
        pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
                                          pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
        batch_size = pred_center.shape[0]
        pred_obbs, pred_bboxes = [], []
        for i in range(batch_size):
            pred_obb_batch = dataset_config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i],
                                                            pred_heading_residual[i],
                                                            pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            pred_obbs.append(torch.from_numpy(pred_obb_batch))
            pred_bboxes.append(torch.from_numpy(pred_bbox_batch))
            # print(pred_obb_batch.shape, pred_bbox_batch.shape)
        data_dict['pred_obbs'] = torch.stack(pred_obbs, dim=0).cuda()
        data_dict['pred_bboxes'] = torch.stack(pred_bboxes, dim=0).cuda()
    return data_dict


# TODO: You Should Check It!
def decode_scores(output_dict, end_points,  num_class, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias=False, quality_channel=False, dataset_config=None):
    end_points = decode_scores_classes(output_dict, end_points, num_class)
    end_points = decode_scores_boxes(output_dict, end_points, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias, quality_channel)
    end_points = decode_dataset_config(end_points, dataset_config)
    return end_points

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, config_transformer=None, quality_channel=False, dataset_config=None):
        super().__init__()
        if config_transformer is None:
            raise NotImplementedError('You should input a config')
            config_transformer = {
                'mask': 'near_5',
                'weighted_input': True,
                'transformer_type': 'deformable',
                'deformable_type': 'interpolation',
                'position_embedding': 'none',
                'input_dim': 0,
                'enc_layers': 0,
                'dec_layers': 4,
                'dim_feedforward': 2048,
                'hidden_dim': 288,
                'dropout': 0.1,
                'nheads': 8,
                'pre_norm': False
            }
        config_transformer = EasyDict(config_transformer)
        print(config_transformer, '<< config transformer ')

        transformer_type = config_transformer.get('transformer_type', 'enc_dec')
        position_type = config_transformer.get('position_type', 'vote')
        self.transformer_type = transformer_type
        self.position_type = position_type
        self.center_with_bias = 'dec' not in transformer_type

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.quality_channel = quality_channel
        self.dataset_config = dataset_config

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        # JUST FOR

        if self.position_type == 'seed_attention':
            self.seed_feature_trans = torch.nn.Sequential(
                torch.nn.Conv1d(256, 128, 1),
                torch.nn.BatchNorm1d(128),
                torch.nn.PReLU(128)
            )

        self.detr = DETR3D(config_transformer, input_channels=128, class_output_shape=2+num_class, bbox_output_shape=3+num_heading_bin*2+num_size_cluster*4+int(quality_channel))

    def forward(self, xyz, features, end_points):  # initial_xyz and xyz(voted): just for encoding
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        seed_xyz, seed_features = end_points['seed_xyz'], features
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        else:
            raise NotImplementedError('Unknown sampling strategy: %s. Exiting!' % (self.sampling))

        end_points['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        end_points['aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ----------  TODO PROPOSAL GENERATION AND CHANGE LOSS GENERATION
        # print(features.mean(), features.std(), ' << first,votenet forward features mean and std', flush=True) # TODO CHECK IT
        features = F.relu(self.bn1(self.conv1(features)))
        features = F.relu(self.bn2(self.conv2(features)))

        # print(features.mean(), features.std(), ' << votenet forward features mean and std', flush=True) # TODO CHECK IT

        # _xyz = torch.gather(initial_xyz, 1, sample_inds.long().unsqueeze(-1).repeat(1,1,3))
        # print(initial_xyz.shape, xyz.shape, sample_inds.shape, _xyz.shape, '<< sample xyz shape', flush=True)
        features = features.permute(0, 2, 1)
        # print(xyz.shape, features.shape, '<< detr input feature dim')
        if self.position_type == 'vote':
            output_dict = self.detr(xyz, features, end_points)
            end_points['detr_features'] = output_dict['detr_features']
        elif self.position_type == 'seed_attention':
            decode_vars = {
                'num_class': self.num_class, 
                'num_heading_bin': self.num_heading_bin,
                'num_size_cluster': self.num_size_cluster, 
                'mean_size_arr': self.mean_size_arr,
                'aggregated_vote_xyz': xyz
            }
            seed_features = self.seed_feature_trans(seed_features)
            seed_features = seed_features.permute(0, 2, 1).contiguous()
            output_dict = self.detr(xyz, features, end_points, seed_xyz=seed_xyz, seed_features=seed_features, decode_vars=decode_vars)
        else:
            raise NotImplementedError(self.position_type)
        # output_dict = self.detr(xyz, features, end_points)
        end_points = decode_scores(output_dict, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr,
                                   self.center_with_bias, quality_channel=self.quality_channel, dataset_config=self.dataset_config)

        return end_points


if __name__ == "__main__":
    pass
    # from easydict import EasyDict
    # from model_util_scannet import ScannetDatasetConfig
    # DATASET_CONFIG = ScannetDatasetConfig()
    # config = {
    #     'num_target': 10,
    # }
    # config = EasyDict(config)
    # model = ScannetDatasetConfig(num_class=DATASET_CONFIG.num_class,
    #                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
    #                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
    #                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
    #                              num_proposal=config.num_target,
    #                              sampling="vote_fps")
    # initial_xyz = torch.randn(3, 128, 3)
    # xyz = torch.randn(3, 128, 3)
    # features = torch.randn(3, 128, 128)
    # end_points = model(initial_xyz, xyz, features, {})
    # print(end_points)
