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


from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import random
from sklearn.metrics import roc_auc_score
import gc
import tqdm
import os
import time
import yaml
import argparse

from .utils import *
from .event_dataset import *
from .event_combine_model import *


class MultiEventTorch(object):

    def __init__(self, config_file):
        """
        Initialization
        :param config_file: configuration file
        """

        self.config_file = config_file
        self.config = yaml.safe_load(open(self.config_file))

        self.data_path_info = self.config["data_path_info"]
        self.model_path = self.data_path_info['model_path']
        self.result_path = self.data_path_info['result_path']

        self.train_feature_path = self.data_path_info['train_feature_path']
        self.valid_feature_path = self.data_path_info['valid_feature_path']
        self.test_feature_path = self.data_path_info['test_feature_path']
        self.train_label_path = self.data_path_info['train_label_path']
        self.valid_label_path = self.data_path_info['valid_label_path']
        # self.test_label_path = self.data_path_info['test_label_path']

        self.is_train = self.config['is_train']
        self.seed = self.config['seed']

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

        self.modelFunc = {
            "CatEmbdConcat": {"BiLSTM": CatEmbdConcatModel, "Transformer": CatEmbdConcatAttModel},
            "CatOneHotConcat": {"BiLSTM": CatOnehotConcatModel, "Transformer": CatOnehotConcatAttModel},
            "AllEmbdSum": {"BiLSTM": AllEmbdSumModel, "Transformer": AllEmbdSumAttModel}
        }

        self.datasetFunc = {
            "CatEmbdConcat": CatEmbdConcatData,
            "CatOneHotConcat": CatOneHotConcatData,
            "AllEmbdSum": AllEmbdSumData
        }

    def load_data(self):
        """
        load data
        :return: train data , valid data or test data
        """

        if self.is_train:
            # train data
            train_df = pd.read_csv(self.train_feature_path)
            self.train_label = pd.read_csv(self.train_label_path)
            self.train_df = train_df.sort_values(["user_id", "event_time"], ascending=True).reset_index(drop=True)
            # valid data
            valid_df = pd.read_csv(self.valid_feature_path)  # user_id event_time cat1~catN dense1~denseM
            self.valid_label = pd.read_csv(self.valid_label_path)
            self.valid_df = valid_df.sort_values(["user_id", "event_time"], ascending=True).reset_index(drop=True)
            self.train_valid_df = pd.concat([self.train_df, self.valid_df], axis=0, ignore_index=True)
        else:
            # test data
            test_df = pd.read_csv(self.test_feature_path)  # user_id event_time cat1~catN dense1~denseM
            self.test_df = test_df.sort_values(["user_id", "event_time"], ascending=True).reset_index(drop=True)

    def data_preprocess(self):
        """
        process dictionary map, standard scaler and one-hot
        :return: process result, model
        """

        schema = json_read(self.config["schema_path"])
        dense_cols = get_name(schema)

        # process dictionary map on category column or time column
        if self.is_train:
            dic_idx, total_dic = label_encoder(self.train_valid_df[self.config["cat_cols"]])
            pickle.dump(total_dic, open(os.path.join(self.model_path, self.config["model_Mode"] + '_vocab.pkl'), 'wb'))
            self.config["dic_len"] = dic_idx
            self.config["vocab_dic"] = total_dic
        else:
            total_dic = pickle.load(open(os.path.join(self.model_path, self.config["model_Mode"] + '_vocab.pkl'), "rb"))
            self.config["dic_len"] = len(total_dic)
            self.config["vocab_dic"] = total_dic

        # process standard scaler on numerical column
        if self.is_train:
            ss = StandardScaler()
            ss.fit(self.train_valid_df[dense_cols])
            # ss.fit(self.train_valid_df[self.config["dense_cols"]])
            pickle.dump(ss, open(os.path.join(self.model_path, self.config["model_Mode"] + '_standScaler.pkl'), "wb"))
            self.config["standScaler"] = ss
        else:
            ss = pickle.load(open(os.path.join(self.model_path, self.config["model_Mode"] + '_standScaler.pkl'), "rb"))
            self.config["standScaler"] = ss
        gc.collect()

        # process one-hot on time column
        if self.is_train:
            enc = OneHotEncoder()
            enc.fit(self.train_valid_df[self.config["cat_cols"]])
            pickle.dump(enc,
                        open(os.path.join(self.model_path, self.config["model_Mode"] + '_OneHotEncoder.pkl'), "wb"))
            self.config["oneHotEnc"] = enc
            self.config["oneHotConcat_dim"] = len(enc.get_feature_names()) + self.config["dense_num"]
        else:
            enc = pickle.load(
                open(os.path.join(self.model_path, self.config["model_Mode"] + '_OneHotEncoder.pkl'), "rb"))
            self.config["oneHotEnc"] = enc
            self.config["oneHotConcat_dim"] = len(enc.get_feature_names()) + self.config["dense_num"]

    def set_seed(self):
        """
        set random seed
        :return: 
        """

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(self.seed)

    def get_dataset(self):
        """
        get dataset for different pre-mode
        :return: train_dataset, valid_dataset or test_dataset
        """

        if self.is_train:
            pre_model_ = self.config["Pre_Mode"]
            total_len = len(self.train_label)
            idx = np.random.permutation(range(total_len))

            train_dataset = self.datasetFunc[pre_model_](self.train_df, self.config, labels=self.train_label,
                                                         isStandScaler=True, dropTime=True)
            valid_dataset = self.datasetFunc[pre_model_](self.valid_df, self.config, labels=self.valid_label,
                                                         isStandScaler=True, dropTime=True)

            if self.config["Pre_Mode"] not in ('CatEmbdConcat', 'CatOneHotConcat', 'AllEmbdSum'):
                raise Exception("Invalid Pre_Mode", self.config["Pre_Mode"])
            return train_dataset, valid_dataset

        else:
            pre_model_ = self.config["Pre_Mode"]
            test_dataset = self.datasetFunc[pre_model_](self.test_df, self.config, labels=None, isStandScaler=True,
                                                        dropTime=True)
            if self.config["Pre_Mode"] not in ('CatEmbdConcat', 'CatOneHotConcat', 'AllEmbdSum'):
                raise Exception("Invalid Pre_Mode", self.config["Pre_Mode"])
            return test_dataset

    def train(self):
        """
        train
        :return:
        """

        device = torch.device(self.config["device"] if torch.cuda.is_available() else 'cpu')
        train_dataset, valid_dataset = self.get_dataset()
        train_len = len(train_dataset)
        val_len = len(valid_dataset)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["batch_size"],
                                                       shuffle=True, num_workers=self.config["num_workers"],
                                                       pin_memory=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.config["batch_size"],
                                                       shuffle=True, num_workers=self.config["num_workers"],
                                                       pin_memory=True)

        best_val_loss = 100
        best_val_auc = 0
        # record training time
        t_ave = []
        best_epoch = 0
        pre_model = self.config["Pre_Mode"]
        post_model = self.config["model_Mode"]
        model = self.modelFunc[pre_model][post_model](self.config).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.08)  # 可选
        FLoss = FocalLossModel(2, torch.tensor(self.config["loss_weight"]).to(device))
        FLoss2 = FocalLossModel(2, torch.tensor(self.config["loss_weight"]).to(device))

        for epoch in range(self.config["Epochs"]):
            t1 = time.time()
            print("Processing epoch:", epoch)
            train_loss = 0
            model.train()
            for ii, (input_x, label) in tqdm.tqdm(enumerate(train_dataloader)):
                optimizer.zero_grad()
                out, _ = model(input_x)
                loss = FLoss(out, label.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * label.size()[0]
            train_loss = train_loss / train_len
            t2 = time.time()
            t_ave.append((t2 - t1))
            print("Epoch: ", epoch, "loss: ", train_loss, "time:", t2 - t1)
            if epoch % 3 == 0:
                model.eval()
                val_loss = 0
                pred = []
                val_label = []
                with torch.no_grad():
                    for ii, (input_x, label) in tqdm.tqdm(enumerate(valid_dataloader)):
                        input_x = [_input_x.to(device) for _input_x in input_x]
                        out, _ = model(input_x)
                        loss = FLoss2(out, label.to(device))
                        pred += list(out.cpu().numpy()[:, 1])
                        val_label += list(label.numpy())
                        val_loss += loss.item() * label.size()[0]
                    val_loss = val_loss / val_len
                    val_auc = roc_auc_score(val_label, np.exp(pred))
                    print("Epoch: ", epoch, "Val loss: ", val_loss, "val auc:", val_auc)

                    evaluate_val_res(pred, val_label)
                    if best_val_auc < val_auc:
                        best_val_auc = val_auc
                        best_epoch = epoch
                        torch.save({'state_dict': model.module.state_dict() if hasattr(model,
                                                                                       "module") else model.state_dict()},
                                   os.path.join(self.model_path + "eventCombineModel_" + self.config["model_Mode"]))
                    else:
                        if epoch - best_epoch > self.config["EarlyStop_rounds"]:
                            break

        torch.save({'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict()},
                   os.path.join(self.model_path, "eventCombineModel_last_" + self.config["model_Mode"]))
        print("best epoch: ", best_epoch, "best val AUC: ", best_val_auc, "avg_time:", sum(t_ave) / len(t_ave))

    def predict(self):
        """
        predict
        :return:
        """

        predict_user = self.test_df["user_id"].drop_duplicates().sort_values()

        device = torch.device(self.config["device"] if torch.cuda.is_available() else 'cpu')
        test_dataset = self.get_dataset()
        if self.config["Pre_Mode"] == "CatEmbdConcat" and self.config["model_Mode"] == 'CatEmbdConcatAttModel':
            model = CatEmbdConcatAttModel(self.config).to(device)
        elif self.config["Pre_Mode"] == "CatEmbdConcat" and self.config["model_Mode"] == 'CatEmbdConcatModel':
            model = CatEmbdConcatModel(self.config).to(device)
        elif self.config["Pre_Mode"] == "CatOneHotConcat" and self.config["model_Mode"] == 'CatOnehotConcatModel':
            model = CatOnehotConcatModel(self.config).to(device)
        elif self.config["Pre_Mode"] == "CatOneHotConcat" and self.config["model_Mode"] == 'CatOnehotConcatAttModel':
            model = CatOnehotConcatAttModel(self.config).to(device)
        elif self.config["Pre_Mode"] == "AllEmbdSum" and self.config["model_Mode"] == 'AllEmbdSumModel':
            model = AllEmbdSumModel(self.config).to(device)
        elif self.config["Pre_Mode"] == "AllEmbdSum" and self.config["model_Mode"] == 'AllEmbdSumAttModel':
            model = AllEmbdSumAttModel(self.config).to(device)
        else:
            raise Exception("Invalid Pre_Mode&Model_mode Pair")
        model.load_state_dict(torch.load(
            os.path.join(self.model_path, "eventCombineModel_last_" + self.config["model_Mode"])['state_dict']))
        predictdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config["batch_size"],
                                                        shuffle=False, num_workers=self.config["num_workers"],
                                                        pin_memory=True)

        model.eval()
        with torch.no_grad():
            predictEmbd = []
            pred = []
            for ii, input_x in tqdm.tqdm(enumerate(predictdataloader)):
                out, embd_out = model(input_x)
                pred += list(out.cpu().numpy()[:, 1])
                predictEmbd.append(embd_out.cpu().numpy())

            # save the embedding results
            predictEmbd = np.concatenate(predictEmbd)
            predictEmbd = pd.DataFrame(predictEmbd)
            Embd_cols = ["Embd_" + str(c) for c in predictEmbd.columns]
            predictEmbd.columns = Embd_cols
            predictEmbd["user_id"] = predict_user
            predictEmbd[["user_id"] + Embd_cols].to_csv(os.path.join(self.result_path, self.config["model_Mode"]),
                                                        index=None)

            res = pd.DataFrame()
            res["user_id"] = predict_user
            res["pred"] = np.exp(pred)
            res.to_csv(os.path.join(self.result_path, self.config["model_Mode"]), index=None)


def get_abnormal_node_event(config_file):
    """
    Select training or testing according to parameters
    :param config_file: configuration file
    :return:
    """

    multiEventTorch = MultiEventTorch(config_file=config_file)

    multiEventTorch.set_seed()
    multiEventTorch.load_data()
    multiEventTorch.data_preprocess()

    if multiEventTorch.is_train:
        multiEventTorch.train()
    else:
        multiEventTorch.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='abnormal_event')
    parser.add_argument('-config_file', type=str, default='config.yml', help='the path of config file')

    args = parser.parse_args()
    config_file = args.config_file

    get_abnormal_node_event(config_file)
