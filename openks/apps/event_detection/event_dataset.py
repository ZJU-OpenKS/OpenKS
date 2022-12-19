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

import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from .utils import *

from openks.models import TorchDataset


class CatEmbdConcatData(TorchDataset):
    """
    get data for category column embedding pre-mode
    data: user_id event_time cat1~catN  dense1~denseN
    labels: user_id label
    """

    def __init__(self, datas, config, labels=None, isStandScaler=True, dropTime=True):
        self.config = config
        self.seq_len = config["seq_len"]
        self.dense_cols = config["dense_cols"]
        self.dense_num = config["dense_num"]
        self.cat_cols = config["cat_cols"]
        self.cat_num = config["cat_num"]
        self.time_num = 0 if dropTime else 1
        vocab_dic = config["vocab_dic"]
        ss = config["standardScaler"]
        if labels is not None:
            self.labels = labels.sort_values('user_id').reset_index(drop=True)
            datas = datas.merge(labels[["user_id"]].drop_duplicates(), on="user_id", how="inner")
        else:
            self.labels = None

        self.user_id = datas['user_id'].drop_duplicates().sort_values().reset_index(drop=True)
        self.data_len = self.user_id.shape[0]
        datas = datas.sort_values(["user_id", "event_time"])
        self.att_mask = datas.set_index(["user_id"]).notnull().astype(np.int).reset_index(drop=False)

        # datas = datas.sort_values(["user_id", "event_time"])
        # self.att_mask = self.att_mask.sort_values(["user_id", "event_time"])
        if isStandScaler:
            datas[self.dense_cols] = ss.transform(datas[self.dense_cols])
        for cat_col in ["event_time"] + self.cat_cols:
            datas[cat_col] = datas[cat_col].map(
                lambda x: vocab_dic[(cat_col, x)] if (cat_col, x) in vocab_dic else 0)
        datas.fillna(0, inplace=True)
        if dropTime:
            datas.drop("event_time", axis=1, inplace=True)
            self.att_mask.drop("event_time", axis=1, inplace=True)
            self.time_num = 0
        self.datas = datas.groupby("user_id")

        self.att_mask = self.att_mask.groupby("user_id")
        self.pad_vector = [0 for i in range(self.cat_num + self.time_num)] + [0 for i in range(self.dense_num)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        user_id = self.user_id[item]
        user_id_df = pd.DataFrame({'user_id': user_id})
        if self.labels:
            label = self.labels.merge(user_id_df, on='user_id')['label'].values

        event_data = self.datas.get_group(user_id).iloc[-self.config["seq_len"]:, :].drop("user_id", axis=1).values
        event_attmask = self.att_mask.get_group(user_id).iloc[-self.config["seq_len"]:, :].drop("user_id",
                                                                                                axis=1).values
        inv_eventData = event_data[::-1]
        real_eventNum = len(event_data)
        padding_data = np.array([self.pad_vector for i in range(self.config["seq_len"] - real_eventNum)])
        if self.config["seq_len"] - real_eventNum > 0:
            event_data = np.concatenate([event_data, padding_data])
            inv_eventData = np.concatenate([inv_eventData, padding_data])
            event_attmask = np.concatenate([event_attmask, padding_data])
        masks = np.array([1 for i in range(real_eventNum)] + [0 for i in range(self.config["seq_len"] - real_eventNum)])
        if self.labels is not None:
            return (torch.as_tensor(inv_eventData[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(inv_eventData[:, (self.cat_num + self.time_num):].copy(), dtype=torch.float32), \
                    torch.as_tensor(event_data[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(event_data[:, (self.cat_num + self.time_num):].copy(), dtype=torch.float32), \
                    torch.as_tensor(event_attmask[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(event_attmask[:, (self.cat_num + self.time_num):].copy(), dtype=torch.long), \
                    torch.as_tensor(masks)), torch.as_tensor(label)
        else:
            return (torch.as_tensor(inv_eventData[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(inv_eventData[:, (self.cat_num + self.time_num):].copy(), dtype=torch.float32), \
                    torch.as_tensor(event_data[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(event_data[:, (self.cat_num + self.time_num):].copy(), dtype=torch.float32), \
                    torch.as_tensor(event_attmask[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(event_attmask[:, (self.cat_num + self.time_num):].copy(), dtype=torch.long), \
                    torch.as_tensor(masks))


class CatOneHotConcatData(TorchDataset):
    """
    get data for category column one-hot pre-mode：
    data: user_id event_time cat1~catN  dense1~denseN
    labels: user_id label
    """

    def __init__(self, datas, config, labels=None, isStandScaler=True):
        self.config = config
        self.seq_len = config["seq_len"]
        self.dense_cols = config["dense_cols"]
        self.dense_num = config["dense_num"]
        self.cat_cols = config["cat_cols"]
        self.cat_num = config["cat_num"]
        self.time_num = 0
        enc = config["oneHotEnc"]
        ss = config["standardScaler"]

        if labels is not None:
            self.labels = labels.sort_values('user_id').reset_index(drop=True)
            datas = datas.merge(labels[["user_id"]].drop_duplicates(), on="user_id", how="inner")
        else:
            self.labels = None

        self.user_id = datas['user_id'].drop_duplicates().sort_values().reset_index(drop=True)
        self.data_len = self.user_id.shape[0]
        final_datas = datas[["user_id", "event_time"]].copy()

        cat_datas = enc.transform(datas[self.cat_cols].fillna(-1)).todense()
        encCatCols = enc.get_feature_names()
        catEnc_df = pd.DataFrame(cat_datas, columns=encCatCols)

        if isStandScaler:
            dense_df = pd.DataFrame(ss.transform(datas[self.dense_cols]), columns=self.dense_cols)
        else:
            dense_df = datas[self.dense_cols]
        final_datas = pd.concat([final_datas, catEnc_df, dense_df], axis=1)
        final_datas.fillna(0, inplace=True)
        final_datas = final_datas.sort_values(["user_id", "event_time"])

        final_datas.drop("event_time", axis=1, inplace=True)
        self.datas = final_datas.groupby("user_id")

        self.pad_vector = [0 for i in range(len(encCatCols))] + [0 for i in range(self.dense_num)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):

        user_id = self.user_id[item]
        if self.labels is not None:
            label = self.labels['label'][item]
        event_data = self.datas.get_group(user_id).iloc[-self.config["seq_len"]:, :].drop("user_id", axis=1).values

        inv_eventData = event_data[::-1]
        real_eventNum = len(event_data)
        padding_data = np.array(
            [self.pad_vector for i in range(self.config["seq_len"] - real_eventNum)])
        if self.config["seq_len"] - real_eventNum > 0:
            event_data = np.concatenate([event_data, padding_data])
            inv_eventData = np.concatenate([inv_eventData, padding_data])

        masks = np.array([1 for i in range(real_eventNum)] + [0 for i in range(self.config["seq_len"] - real_eventNum)])
        if self.labels:
            return (torch.as_tensor(inv_eventData.copy(), dtype=torch.float32), \
                    torch.as_tensor(event_data.copy(), dtype=torch.float32), torch.as_tensor(masks)), torch.as_tensor(
                label)
        else:
            return (torch.as_tensor(inv_eventData.copy(), dtype=torch.float32), \
                    torch.as_tensor(event_data.copy(), dtype=torch.float32), torch.as_tensor(masks))


class AllEmbdSumData(TorchDataset):
    """
    get data for all column embedding pre-mode
    data: user_id event_time cat1~catN  dense1~denseN
    labels: user_id label
    """

    def __init__(self, datas, config, labels=None, isStandScaler=True, dropTime=True):
        self.config = config
        self.seq_len = config["seq_len"]
        self.dense_cols = get_name(json_read(self.config["schema_path"]))
        # self.dense_cols = config["dense_cols"]
        self.dense_num = config["dense_num"]
        self.cat_cols = config["cat_cols"]
        self.cat_num = config["cat_num"]
        self.time_num = config["time_num"]
        vocab_dic = config["vocab_dic"]
        ss = config["standScaler"]
        if labels is not None:
            self.labels = labels.sort_values('user_id').reset_index(drop=True)
            datas = datas.merge(labels[["user_id"]].drop_duplicates(), on="user_id", how="inner")
        else:
            self.labels = None

        self.user_id = datas['user_id'].drop_duplicates().sort_values().reset_index(drop=True)
        self.data_len = self.user_id.shape[0]
        datas = datas.sort_values(["user_id", "event_time"])
        self.att_mask = datas.set_index(["user_id", "event_time"]).notnull().astype(np.int).reset_index(drop=False)

        # datas = datas.sort_values(["user_id", "event_time"])
        # self.att_mask = self.att_mask.sort_values(["user_id", "event_time"])
        self.att_mask["event_time"] = 1
        if isStandScaler:
            datas[self.dense_cols] = ss.transform(datas[self.dense_cols])
        for cat_col in self.cat_cols:
            datas[cat_col] = datas[cat_col].map(
                lambda x: vocab_dic[(cat_col, x)] if (cat_col, x) in vocab_dic else vocab_dic[
                    cat_col])
        datas.fillna(0, inplace=True)
        if dropTime:
            datas.drop("event_time", axis=1, inplace=True)
            self.att_mask.drop("event_time", axis=1, inplace=True)
            self.time_num = 0
        self.datas = datas.groupby("user_id")

        self.att_mask = self.att_mask.groupby("user_id")
        self.pad_vector = [0 for i in range(self.cat_num + self.time_num)] + [0 for i in range(self.dense_num)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        """
        进行mask填充
        :param item:
        :return:
        """
        user_id = self.user_id[item]
        if self.labels is not None:
            label = self.labels['label'][item]

        event_data = self.datas.get_group(user_id).iloc[-self.seq_len:, :].drop("user_id", axis=1).values
        event_attmask = self.att_mask.get_group(user_id).iloc[-self.seq_len:, :].drop("user_id", axis=1).values
        inv_eventData = event_data[::-1]
        inv_eventAttmask = event_attmask[::-1]
        real_eventNum = len(event_data)
        padding_data = np.array(
            [self.pad_vector for i in range(self.config["seq_len"] - real_eventNum)])
        if self.config["seq_len"] - real_eventNum > 0:
            event_data = np.concatenate([event_data, padding_data])
            inv_eventData = np.concatenate([inv_eventData, padding_data])
            event_attmask = np.concatenate([event_attmask, padding_data])
            inv_eventAttmask = np.concatenate([inv_eventAttmask, padding_data])
        masks = np.array([1 for i in range(real_eventNum)] + [0 for i in range(self.config["seq_len"] - real_eventNum)])
        if self.labels is not None:
            return (torch.as_tensor(inv_eventData[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(inv_eventData[:, (self.cat_num + self.time_num):].copy(), dtype=torch.float32), \
                    torch.as_tensor(event_data[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(event_data[:, (self.cat_num + self.time_num):].copy(), dtype=torch.float32), \
                    torch.as_tensor(inv_eventAttmask.copy(), dtype=torch.float32), \
                    torch.as_tensor(event_attmask.copy(), dtype=torch.float32), torch.as_tensor(masks)), \
                   torch.as_tensor(label)
        else:
            return (torch.as_tensor(inv_eventData[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(inv_eventData[:, (self.cat_num + self.time_num):].copy(), dtype=torch.float32), \
                    torch.as_tensor(event_data[:, :(self.cat_num + self.time_num)].copy(), dtype=torch.long), \
                    torch.as_tensor(event_data[:, (self.cat_num + self.time_num):].copy(), dtype=torch.float32), \
                    torch.as_tensor(inv_eventAttmask.copy(), dtype=torch.float32), \
                    torch.as_tensor(event_attmask.copy(), dtype=torch.float32), torch.as_tensor(masks))
