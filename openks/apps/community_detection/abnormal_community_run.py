# -*- coding: utf-8 -*-
#
# Copyright 2022 HangZhou Hikvision Digital Technology Co., Ltd. All Right Reserved.
# Modified from rozemberczki2019gemsec "GEMSEC: Graph Embedding with Self Clustering"
# * author={Rozemberczki, Benedek and Davies, Ryan and Sarkar, Rik and Sutton, Charles}
# * booktitle={Proceedings of the 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2019}
# * pages={65-72}
# * year={2019}
# * organization={ACM}
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import yaml
import argparse
import pandas as pd

from .abnormal_community_model import *


class AbnormalCommunityPaddle(object):

    def __init__(self, config_file):
        """
        initialization
        :param config_file: configuration file
        """

        self.config_file = config_file
        self.config = yaml.safe_load(open(self.config_file))

        self.input = self.config['input_path']
        self.node_info = self.config['node_info']

        self.embedding_output = self.config['embedding_output']
        self.cluster_mean_output = self.config['cluster_mean_output']
        self.log_output = self.config['log_output']
        self.assignment_output = self.config['assignment_output']
        self.abnormal_group_score_output = self.config['abnormal_group_score_output']

        self.dump_matrices = self.config['dump_matrices']
        self.model = self.config['model']

        self.P = self.config['P']
        self.Q = self.config['Q']
        self.walker = self.config['walker']

        self.dimensions = self.config['dimensions']

        self.random_walk_length = self.config['random_walk_length']
        self.num_of_walks = self.config['num_of_walks']
        self.window_size = self.config['window_size']
        self.distortion = self.config['distortion']
        self.negative_sample_number = self.config['negative_sample_number']

        self.initial_learning_rate = self.config['initial_learning_rate']
        self.minimal_learning_rate = self.config['minimal_learning_rate']
        self.annealing_factor = self.config['annealing_factor']
        self.initial_gamma = self.config['initial_gamma']
        self.lambd = self.config['lambd']
        self.cluster_number = self.config['cluster_number']
        self.overlap_weighting = self.config['overlap_weighting']
        self.regularization_noise = self.config['regularization_noise']

        if not os.path.exists(self.input):
            os.mkdir(self.input)

        if not os.path.exists(self.embedding_output):
            os.mkdir(self.embedding_output)

        if not os.path.exists(self.cluster_mean_output):
            os.mkdir(self.cluster_mean_output)

        if not os.path.exists(self.log_output):
            os.mkdir(self.log_output)

        if not os.path.exists(self.assignment_output):
            os.mkdir(self.assignment_output)

    def graph_reader(self):
        """
        get relationship network
        :return: graph
        """

        edges = pd.read_csv(self.input)
        self.graph = nx.from_edgelist(edges.values.tolist())

    def train(self):
        """
        train
        :return: model
        """

        if self.model == "GEMSECWithRegularization":
            model = GEMSECWithRegularization(self.config, self.graph)
        elif self.model == "GEMSEC":
            model = GEMSEC(self.config, self.graph)
        elif self.model == "DeepWalkWithRegularization":
            model = DeepWalkWithRegularization(self.config, self.graph)
        else:
            model = DeepWalk(self.config, self.graph)
        model.train()

    def GroupScore(self):
        """
        get abnormal score for every community
        :param assignments: community dictionary
        :return: GroupAbnormalScore: abnormal score for every community
        """

        group_dict = json_read(self.config["assignment_output"])

        nodefeatures = pd.read_csv(self.node_info)

        AbnormalGroup = dict()
        for node in group_dict.keys():
            if group_dict[node] in AbnormalGroup.keys():
                AbnormalGroup[group_dict[node]].append(int(node))
            else:
                AbnormalGroup[group_dict[node]] = [int(node)]

        def calculateScore(abnormalgroup, nodefeatures):
            abnormalnodeOccupation = sum([nodefeatures['type'][ID] for ID in abnormalgroup]) / len(abnormalgroup)
            AverageabnormalRecord = sum([nodefeatures['abnormal_record'][ID] for ID in abnormalgroup]) / len(
                abnormalgroup)
            maleOccupation = sum([nodefeatures['sex'][ID] for ID in abnormalgroup]) / len(abnormalgroup)
            AverageAge = sum([nodefeatures['age'][ID] for ID in abnormalgroup]) / len(abnormalgroup)
            return abnormalnodeOccupation + AverageabnormalRecord + maleOccupation + AverageAge

        AbnormalGroupScore = dict()
        for group in AbnormalGroup.keys():
            AbnormalGroupScore[group] = calculateScore(AbnormalGroup[group], nodefeatures)
        print(AbnormalGroupScore)
        json_dumper(AbnormalGroupScore, self.abnormal_group_score_output)
        return AbnormalGroupScore


def get_abnormal_communities(config_file):
    """
    abnormal communities identify
    :param config_file: configuration file
    :return:
    """

    abnormalCommunitypaddle = AbnormalCommunityPaddle(config_file=config_file)

    abnormalCommunitypaddle.graph_reader()
    abnormalCommunitypaddle.train()
    abnormalCommunitypaddle.GroupScore()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='abnormal_action')
    parser.add_argument('-config_file', type=str, default='config.yml', help='the path of config file')

    args = parser.parse_args()

    config_file = args.config_file

    get_abnormal_communities(config_file)
