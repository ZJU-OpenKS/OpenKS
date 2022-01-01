import torch
import torch.nn as nn
import numpy as np
import sys
import os
#sys.path.append(os.path.join(os.getcwd(), os.pardir, "openks/models/pytorch/mmd_modules/ThreeDJCG")) # HACK add the lib folder
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.base_module.backbone_module import Pointnet2Backbone
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.base_module.voting_module import VotingModule
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.proposal_module.proposal_module import ProposalModule
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.proposal_module.relation_module import RelationModule
from .caption_module import SceneCaptionModule


class CapNet(nn.Module):
    def __init__(self, num_class, vocabulary, embeddings, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=256, num_locals=-1, vote_factor=1, sampling="vote_fps",
    no_caption=False, query_mode="corner", num_graph_steps=0,
    emb_size=300, hidden_size=512, dataset_config=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.no_caption = no_caption
        self.num_graph_steps = num_graph_steps
        self.dataset_config = dataset_config

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        #self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal,
        #                               sampling, config_transformer=config_transformer, dataset_config=dataset_config)

        # Caption generation
        if not no_caption:
            self.relation = RelationModule(num_proposals=num_proposal, det_channel=128)  # bef 256
            self.caption = SceneCaptionModule(vocabulary, embeddings, emb_size, 128,
                    hidden_size, num_proposal, num_locals, query_mode)

    def forward(self, data_dict, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)

        #######################################
        #                                     #
        #           GRAPH ENHANCEMENT         #
        #                                     #
        #######################################

        if self.num_graph_steps > 0: data_dict = self.graph(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.no_caption:
            data_dict = self.relation(data_dict)
            data_dict = self.caption(data_dict, is_eval)
        else:
            data_dict['max_iou_rate_0.25'] = 0.
            data_dict['max_iou_rate_0.5'] = 0.

        return data_dict
