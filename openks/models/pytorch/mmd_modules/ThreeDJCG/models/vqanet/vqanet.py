import torch
import torch.nn as nn
import numpy as np
import sys
import os

from openks.models.pytorch.mmd_modules.ThreeDJCG.models.base_module.backbone_module import Pointnet2Backbone
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.base_module.voting_module import VotingModule

from openks.models.pytorch.mmd_modules.ThreeDJCG.models.proposal_module.proposal_module_fcos import ProposalModule
from openks.models.pytorch.mmd_modules.ThreeDJCG.models.proposal_module.relation_module import RelationModule

from .lang_module_t5 import LangModule
from .crossmodal_module import CrossmodalModule


class VqaNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, no_reference=False,
                 hidden_size=256, dataset_config=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.no_reference = no_reference
        self.dataset_config = dataset_config

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.language = LangModule(100, use_lang_classifier=use_lang_classifier)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.relation = RelationModule(num_proposals=num_proposal, det_channel=128)  # bef 256
            self.crossmodal = CrossmodalModule(num_proposals=num_proposal, det_channel=128, hidden_size=768)  # bef 256


    def forward(self, data_dict):
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

        if not self.no_reference:
            #######################################
            #           LANGUAGE BRANCH           #
            #           PROPOSAL BRANCH           #
            #######################################
            data_dict = self.language.encode(data_dict)
            data_dict = self.relation(data_dict)

            #######################################
            #        CROSSMODAL INTERACTION       #
            #######################################
            # config for bbox_embedding
            data_dict = self.crossmodal(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################
            data_dict = self.language.decode(data_dict)

        return data_dict
