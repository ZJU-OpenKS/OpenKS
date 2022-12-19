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

import argparse

from .dataset import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import multiprocessing
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from .featuresGraph import *
from torch_geometric.data import Data
import random
import yaml
import sklearn
import joblib
import csv


class AbnormalNodeDetect(object):

    def __init__(self, config_file):
        """
        initialization
        :param config_file: configuration file
        """

        self.config_file = config_file
        self.config = yaml.safe_load(open(self.config_file))

        self.data_path = self.config["dataset_dir"]
        self.model_path = self.config["model_path"]
        self.result_path = self.config["result_path"]
        self.predict_data = self.config["predict_data"]

        self.seed = self.config['seed']
        self.targetAccuracy = self.config["targetAccuracy"]
        self.TrainTimes = self.config["TrainTimes"]
        self.is_train = self.config['is_train']

    def set_seed(self):
        """
        set seed
        :return:
        """

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True

    def generate_data(self, n_class):
        """
        generate data for model train or predict
        :param n_class: the number of class
        :return: train, test, train_label, test_label
        """

        dataset = Dataset(self.data_path)
        data = dataset.get_data()
        train_indices = data['train_indices']
        test_indices = data['test_indices']
        fea_table = data['fea_table']
        fea_table = fea_table.loc[:, (fea_table != fea_table.iloc[0]).any()].copy()
        edge_file = data['edge_file']
        train_label = data['train_label']
        test_label = data['test_label']
        p = Pool(multiprocessing.cpu_count() - 1)
        res = []
        res.append(
            p.apply_async(extract_1st_features,
                          args=(edge_file, train_indices, test_indices, train_label, n_class,)))
        res.append(p.apply_async(extract_2nd_features,
                                 args=(edge_file, train_indices, test_indices,)))
        res.append(p.apply_async(origin_feature_agg,
                                 args=(edge_file, train_indices, test_indices, fea_table,)))
        p.close()
        p.join()
        res1 = []
        for r in res:
            res1.append(r.get())
        X = np.hstack(res1)
        train_feature = []
        test_featuer = []
        for i in train_indices:
            train_feature.append(X[i])
        for j in test_indices:
            test_featuer.append(X[j])
        train = pd.DataFrame(train_feature)
        test = pd.DataFrame(test_featuer)
        tr_label = train_label.drop("node_index", 1)
        te_label = test_label.drop("node_index", 1)
        return train, test, tr_label, te_label

    def RFClassifierTrain(self, train, train_label, test, test_label,
                          model_file='RandomForestsClassifier.model'):
        """
        Random Forest Classifier
        :param train: train data features
        :param train_label: train data label
        :param test: test data features
        :param test_label: test data label
        :param model_file: model file name
        :return: model
        """

        times = 0
        BestAccuracy = 0
        address = self.model_path + model_file
        while times < self.TrainTimes:
            RandomForests = sklearn.ensemble.RandomForestClassifier(random_state=times)
            RandomForests.fit(train, train_label)
            PredictTestLabel = RandomForests.predict(test)
            PredictTest_Tensors = torch.tensor(PredictTestLabel)
            y = test_label['label'].values.tolist()
            test_label_tensor = torch.tensor(y)
            print('RandomForestsPredictAccuracy:')
            accuracy = PredictTest_Tensors.eq(test_label_tensor).sum().item() / len(PredictTest_Tensors)
            if accuracy > BestAccuracy:
                BestAccuracy = accuracy
                BestRF = RandomForests
            print(accuracy)
            if accuracy >= self.targetAccuracy:
                print('This Model Satisfy Target Accuracy In TestData Set.')
                print('Storing Model In File (%s)...' % (address))
                joblib.dump(RandomForests, address)
                return RandomForests
        print('Can not reach target Accuracy In %d times' % (self.TrainTimes))
        print('RandomForestsBestPredictAccuracy:')
        print(BestAccuracy)
        joblib.dump(BestRF, address)
        return RandomForests

    def RFClassifierPredict(self, test, RandomForestsModel=None,
                            model_file='RandomForestsClassifier.model', result_file='RFClassifierPredict.csv'):
        """
        Random Forest Classifier Prediction
        :param test: test data features
        :param RandomForestsModel: model
        :param model_file: model name
        :param result_file: result file name
        :return: Predict Results
        """
        model_address = self.model_path + model_file
        # model_address = self.result_path + model_file
        if RandomForestsModel == None:
            RandomForestsModel = joblib.load(model_address)
        PredictResults = RandomForestsModel.predict(test)
        PredictProba = RandomForestsModel.predict_proba(test)
        PredictClass_Proba = [PredictProba[SampleID][PredictResults[SampleID]] for SampleID in
                              range(len(PredictResults))]
        Predict = pd.DataFrame()
        Predict['ID'] = list(range(len(PredictResults)))
        Predict['PredictLabel'] = PredictResults
        Predict['Probably'] = PredictClass_Proba
        Predict_csv_Address = self.result_path + result_file
        Predict.to_csv(Predict_csv_Address, index=False)

        print('Predict Result from RandomForestsClassifier :')
        print(PredictResults)
        return Predict


def get_abnormal_node_from_kg(config_file):
    """
    abnormal node detect
    :param config_file: configuration file
    :return:
    """

    abnormalNodeDetect = AbnormalNodeDetect(config_file=config_file)

    abnormalNodeDetect.set_seed()
    train, test, train_label, test_label = abnormalNodeDetect.generate_data(7)

    if abnormalNodeDetect.is_train:
        abnormalNodeDetect.RFClassifierTrain(train, train_label, test, test_label)
        abnormalNodeDetect.RFClassifierPredict(test, RandomForestsModel=None)
    else:
        abnormalNodeDetect.RFClassifierPredict(test, RandomForestsModel=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='abnormal_action')
    parser.add_argument('-config_file', type=str, default='config.yml', help='the path of config file')

    args = parser.parse_args()
    config_file = args.config_file

    get_abnormal_node_from_kg(config_file)
