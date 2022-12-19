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

import copy
from os.path import join
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from .utils import get_logger

TYPE_MAP = {
    'cat': str,
    'multi-cat': str,
    'str': str,
    'num': np.float64,
    'timestamp': 'str'
}

VERBOSITY_LEVEL = 'WARNING'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)
TIMESTAMP_TYPE_NAME = 'timestamp'
TRAIN_FILE = 'train_node_id.txt'
TRAIN_LABEL = 'train_label.tsv'
TEST_LABEL = 'test_label.tsv'
TEST_FILE = 'test_node_id.txt'
INFO_FILE = 'config.yml'
FEA_TABLE = 'feature.tsv'
EDGE_FILE = 'edge.tsv'

SEP = '\t'


def _date_parser(millisecs):
    if np.isnan(float(millisecs)):
        return millisecs

    return datetime.fromtimestamp(float(millisecs))


class Dataset:
    """"Dataset"""

    def __init__(self, dataset_dir):
        """
            train_dataset, test_dataset: list of strings
            train_label: np.array
        """
        self.dataset_dir_ = dataset_dir
        self.metadata_ = self._read_metadata(join(dataset_dir, INFO_FILE))
        self.edge_data = None
        self.train_indices = None
        self.train_label = None
        self.test_label = None
        self.test_indices = None
        self.fea_table = None
        self.get_data()

    def get_data(self):
        """get all training data"""
        data = {
            'fea_table': self.get_fea_table(),
            'edge_file': self.get_edge(),
            'train_indices': self.get_train_indices(),
            'test_indices': self.get_test_indices(),
            'train_label': self.get_train_label(),
            'test_label': self.get_test_label(),
        }
        return data

    def get_fea_table(self):
        """get train"""
        if self.fea_table is None:
            self.fea_table = self._read_dataset(
                join(self.dataset_dir_, FEA_TABLE))
        return self.fea_table

    def get_edge(self):
        """get edge file"""
        dtype = {
            'src_id': int,
            'dst_idx': int,
            'edge_weight': float
        }
        if self.edge_data is None:
            self.edge_data = pd.read_csv(
                join(self.dataset_dir_, EDGE_FILE), dtype=dtype, sep=SEP)
        return self.edge_data

    def get_train_label(self):
        """get train label"""
        dtype = {
            'node_index': int,
            'label': int,
        }
        if self.train_label is None:
            self.train_label = pd.read_csv(
                join(self.dataset_dir_, TRAIN_LABEL), dtype=dtype, sep=SEP)

        return self.train_label

    def get_test_label(self):
        """get train label"""
        dtype = {
            'node_index': int,
            'label': int,
        }
        if self.test_label is None:
            self.test_label = pd.read_csv(
                join(self.dataset_dir_, TEST_LABEL), dtype=dtype, sep=SEP)

        return self.test_label

    def get_test_indices(self):
        """get test index file"""
        if self.test_indices is None:
            with open(join(self.dataset_dir_, TEST_FILE), 'r') as ftmp:
                self.test_indices = [int(line.strip()) for line in ftmp]

        return self.test_indices

    def get_train_indices(self):
        """get train index file"""
        if self.train_indices is None:
            with open(join(self.dataset_dir_, TRAIN_FILE), 'r') as ftmp:
                self.train_indices = [int(line.strip()) for line in ftmp]

        return self.train_indices

    def get_metadata(self):
        """get metadata"""
        return copy.deepcopy(self.metadata_)

    @staticmethod
    def _read_metadata(metadata_path):
        with open(metadata_path, 'r') as ftmp:
            return yaml.safe_load(ftmp)

    def _read_dataset(self, dataset_path):
        schema = self.metadata_['schema']
        if isinstance(schema, dict):
            table_dtype = {key: TYPE_MAP[val] for key, val in schema.items()}
            date_list = [key for key, val in schema.items()
                         if val == TIMESTAMP_TYPE_NAME]
            dataset = pd.read_csv(
                dataset_path, sep=SEP, dtype=table_dtype,
                parse_dates=date_list, date_parser=_date_parser)
        else:
            dataset = pd.read_csv(dataset_path, sep=SEP)

        return dataset
