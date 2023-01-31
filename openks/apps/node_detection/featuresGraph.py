# -*- coding: utf-8 -*-
#
# Copyright 2022 HangZhou Hikvision Digital Technology Co., Ltd. All Right Reserved.
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

import numpy as np
import pandas as pd


def calculate_label_percent(data, n_class):
    """calculate the percent of label"""
    label_count = np.zeros(n_class + 1)
    for index, value in zip(data.value_counts().index, data.value_counts().values):
        label_count[index + 1] = value
    label_percent = np.zeros(n_class + 1)
    neighbors_num = np.sum(label_count)
    if neighbors_num != 0:
        label_percent = label_count / neighbors_num
    return np.concatenate([label_count, label_percent])


def agg_features(edge_file, features, in_or_out):
    """get aggregator features"""
    features_df = pd.DataFrame(features).reset_index().rename(columns={'index': 'node_index'})

    if in_or_out == 'in':
        df = pd.merge(features_df, edge_file, left_on='node_index', right_on='dst_idx', how='left')
        df = df[['node_index', 'src_idx', 'dst_idx']]

        df = pd.merge(df, features_df, left_on='src_idx', right_on='node_index', how='left')
        df = df.fillna(0)
        df = df.drop(columns=['src_idx', 'dst_idx', 'node_index_y'])
        in_agg_features = df.groupby('node_index_x').agg(['sum', 'max', 'min', 'mean']).sort_index()
        return in_agg_features.values

    if in_or_out == 'out':
        # out
        df = pd.merge(features_df, edge_file, left_on='node_index', right_on='src_idx', how='left')
        df = df[['node_index', 'src_idx', 'dst_idx']]
        df = pd.merge(df, features_df, left_on='dst_idx', right_on='node_index', how='left')
        df = df.fillna(0)
        df = df.drop(columns=['src_idx', 'dst_idx', 'node_index_y'])
        out_agg_features = df.groupby('node_index_x').agg(['sum', 'max', 'min', 'mean']).sort_index()

        return out_agg_features.values


def is_directed_graph(edge_file, train_indices, test_indices):
    """whether it is a directed graph"""
    node_num = len(train_indices) + len(test_indices)

    out_1st_degree_temp = edge_file.groupby('src_idx')['dst_idx'].agg(lambda x: len(x))
    out_1st_degree = np.zeros(node_num)
    for index in out_1st_degree_temp.index:
        out_1st_degree[index] = out_1st_degree_temp[index]

    in_1st_degree_temp = edge_file.groupby('dst_idx')['src_idx'].agg(lambda x: len(x))
    in_1st_degree = np.zeros(node_num)
    for index in in_1st_degree_temp.index:
        in_1st_degree[index] = in_1st_degree_temp[index]

    for i in range(len(in_1st_degree)):
        if in_1st_degree[i] != out_1st_degree[i]:
            return True
    return False


def origin_feature_agg(edge_file, train_indices, test_indices, fea_table):
    """get aggregator feature table"""
    directed_flag = is_directed_graph(edge_file, train_indices, test_indices)
    if directed_flag:
        agg_fea_table_in = agg_features(edge_file, fea_table.values, 'in')
        agg_fea_table_out = agg_features(edge_file, fea_table.values, 'out')
        agg_fea_table = np.hstack((agg_fea_table_in, agg_fea_table_out))
    else:
        agg_fea_table_in = agg_features(edge_file, fea_table.values, 'in')
        agg_fea_table = agg_fea_table_in

    return agg_fea_table


def extract_1st_features(edge_file, train_indices, test_indices, train_label, n_class):
    """get the first order features"""
    node_num = len(train_indices) + len(test_indices)

    out_1st_degree_temp = edge_file.groupby('src_idx')['dst_idx'].agg(lambda x: len(x))
    out_1st_degree = np.zeros(node_num)
    for index in out_1st_degree_temp.index:
        out_1st_degree[index] = out_1st_degree_temp[index]

    in_1st_degree_temp = edge_file.groupby('dst_idx')['src_idx'].agg(lambda x: len(x))
    in_1st_degree = np.zeros(node_num)
    for index in in_1st_degree_temp.index:
        in_1st_degree[index] = in_1st_degree_temp[index]

    is_directed = False
    for i in range(len(in_1st_degree)):
        if in_1st_degree[i] != out_1st_degree[i]:
            is_directed = True
            break

    in_1st_edge_weight_sum_temp = edge_file.groupby('dst_idx')['edge_weight'].agg(np.sum)
    in_1st_edge_weight_sum = np.zeros(node_num)
    for index in in_1st_edge_weight_sum_temp.index:
        in_1st_edge_weight_sum[index] = in_1st_edge_weight_sum_temp[index]

    # Proportion of labels of first-order neighbors
    df = pd.merge(edge_file, train_label, left_on='src_idx', right_on='node_index', how='left')
    df.rename(columns={'label': 'src_label'}, inplace=True)
    df = pd.merge(df, train_label, left_on='dst_idx', right_on='node_index', how='left')
    df.rename(columns={'label': 'dst_label'}, inplace=True)
    df.fillna(-1, inplace=True)
    df = df.astype(int)

    in_1st_neighbor_label_features_temp = df.groupby('dst_idx')['src_label'].apply(calculate_label_percent, n_class)
    in_1st_neighbor_label_features = pd.Series([np.zeros(2 * n_class + 2)] * node_num)
    for index in in_1st_neighbor_label_features_temp.index:
        in_1st_neighbor_label_features[index] = in_1st_neighbor_label_features_temp[index]
    in_1st_neighbor_label_features = np.concatenate(in_1st_neighbor_label_features.values).reshape(-1,
                                                                                                   2 * n_class + 2)

    # add aggregator features
    agg_in_1st_degree = agg_features(edge_file, in_1st_degree.reshape(-1, 1), 'in')
    agg_in_1st_edge_weight_sum = agg_features(edge_file, in_1st_edge_weight_sum.reshape(-1, 1), 'in')

    if is_directed:
        out_1st_edge_weight_sum_temp = edge_file.groupby('src_idx')['edge_weight'].agg(np.sum)
        out_1st_edge_weight_sum = np.zeros(node_num)
        for index in out_1st_edge_weight_sum_temp.index:
            out_1st_edge_weight_sum[index] = out_1st_edge_weight_sum_temp[index]

        out_1st_neighbor_label_features_temp = df.groupby('src_idx')['dst_label'].apply(calculate_label_percent,
                                                                                        n_class)
        out_1st_neighbor_label_features = pd.Series([np.zeros(2 * n_class + 2)] * node_num)
        for index in out_1st_neighbor_label_features_temp.index:
            out_1st_neighbor_label_features[index] = out_1st_neighbor_label_features_temp[index]

        out_1st_neighbor_label_features = np.concatenate(out_1st_neighbor_label_features.values).reshape(-1,
                                                                                                         2 * n_class + 2)

        agg_out_1st_degree = agg_features(edge_file, out_1st_degree.reshape(-1, 1), 'out')
        agg_out_1st_edge_weight_sum = agg_features(edge_file, out_1st_edge_weight_sum.reshape(-1, 1), 'out')

        return np.hstack((in_1st_degree.reshape(-1, 1), in_1st_edge_weight_sum.reshape(-1, 1),
                          in_1st_neighbor_label_features, agg_in_1st_degree, agg_in_1st_edge_weight_sum,
                          out_1st_degree.reshape(-1, 1), out_1st_edge_weight_sum.reshape(-1, 1),
                          out_1st_neighbor_label_features, agg_out_1st_degree, agg_out_1st_edge_weight_sum))
        # return  np.hstack((in_1st_degree.reshape(-1, 1),out_1st_degree.reshape(-1, 1)))

    # return np.hstack((in_1st_degree.reshape(-1, 1), in_1st_edge_weight_sum.reshape(-1, 1),
    #                 in_1st_neighbor_label_features, agg_in_1st_degree, agg_in_1st_edge_weight_sum))
    return in_1st_degree.reshape(-1, 1)


def extract_2nd_features(edge_file, train_indices, test_indices):
    """get the second order features"""
    node_num = len(train_indices) + len(test_indices)
    out_1st_degree_temp = edge_file.groupby('src_idx')['dst_idx'].agg(lambda x: len(x))
    out_1st_degree = pd.Series(np.zeros(node_num))
    for index in out_1st_degree_temp.index:
        out_1st_degree[index] = out_1st_degree_temp[index]
    in_1st_degree_temp = edge_file.groupby('dst_idx')['src_idx'].agg(lambda x: len(x))
    in_1st_degree = pd.Series(np.zeros(node_num))
    for index in in_1st_degree_temp.index:
        in_1st_degree[index] = in_1st_degree_temp[index]

    is_directed = False
    for i in range(len(in_1st_degree)):
        if in_1st_degree[i] != out_1st_degree[i]:
            is_directed = True
            break

    in_1st_degree_df = in_1st_degree.to_frame(name='in_degree').reset_index().rename(columns={'index': 'node_index'})
    df = pd.merge(edge_file, in_1st_degree_df, left_on='src_idx', right_on='node_index', how='left')
    in_degree_2nd_temp = df.groupby('dst_idx')['in_degree'].agg(np.sum)
    in_degree_2nd = np.zeros(node_num)
    for index in in_degree_2nd_temp.index:
        in_degree_2nd[index] = in_degree_2nd_temp[index]

    # add aggregator features
    agg_in_2nd_degree = agg_features(edge_file, in_degree_2nd.reshape(-1, 1), 'in')

    if is_directed:
        out_1st_degree_df = out_1st_degree.to_frame(name='out_degree').reset_index().rename(
            columns={'index': 'node_index'})
        df = pd.merge(edge_file, out_1st_degree_df, left_on='dst_idx', right_on='node_index', how='left')
        out_degree_2nd_temp = df.groupby('src_idx')['out_degree'].agg(np.sum)
        out_degree_2nd = np.zeros(node_num)

        for index in out_degree_2nd_temp.index:
            out_degree_2nd[index] = out_degree_2nd_temp[index]

        agg_out_2nd_degree = agg_features(edge_file, out_degree_2nd.reshape(-1, 1), 'out')

        return np.hstack((in_degree_2nd.reshape(-1, 1), out_degree_2nd.reshape(-1, 1)))
        # return np.hstack(
        #     (in_degree_2nd.reshape(-1, 1), agg_in_2nd_degree, out_degree_2nd.reshape(-1, 1), agg_out_2nd_degree))
    return np.hstack((in_degree_2nd.reshape(-1, 1), agg_in_2nd_degree))
    # return in_degree_2nd.reshape(-1, 1)
