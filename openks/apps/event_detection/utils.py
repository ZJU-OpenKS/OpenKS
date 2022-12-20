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

import pandas as pd
import numpy as np
import json


def json_read(schema_path):
    """
    read json file
    :param schema_path: the path of json file
    :return:
    """
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return schema


def get_name(schema):
    """
    get numerical features' name
    """
    filter_names = ["user_id", "type"] + ["fea1", "fea2", "fea3", "fea4", "fea5", "fea6", "fea7", "fea8", "fea9"]
    names = [x['name'] for x in schema[0]['properties']]
    names = [name for name in names if name not in filter_names]
    return names

def label_encoder(dataframe):
    """
    process encoding
    :param dataframe: data needs encoding
    :return: dictionary index and total dictionary
    """
    total_dic = {}
    dic_idx = 0

    total_dic["pad_unk"] = dic_idx
    dic_idx += 1

    cols = dataframe.columns.tolist()
    for col in cols:
        dic_values = dataframe[col].drop_duplicates()

        total_dic[col] = dic_idx
        dic_idx += 1

        for val in dic_values:
            total_dic[(col, val)] = dic_idx
            dic_idx += 1
    return dic_idx, total_dic


def evaluate_val_res(pred_label, actual_label):
    """
    evaluate validation results
    :param pred_label: predicted label
    :param actual_label: actual label
    :return: Top N validation results
    """
    val_res = pd.DataFrame()
    val_res["pred"] = np.exp(pred_label)
    val_res["label"] = actual_label
    val_res = val_res.sort_values("pred", ascending=False)
    valTop100 = val_res.iloc[:100]["label"].sum()
    valTop500 = val_res.iloc[:500]["label"].sum()
    valTop2000 = val_res.iloc[:2000]["label"].sum()
    valTop5000 = val_res.iloc[:5000]["label"].sum()
    valTop10000 = val_res.iloc[:10000]["label"].sum()
    print("Val Top:", "Top100", valTop100, "Top500", valTop500, "Top2000", valTop2000, "Top5000", valTop5000,
          "Top10000", valTop10000)
