# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import argparse
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.optim import optimizer
import numpy as np
from sklearn.model_selection import train_test_split
from ..model import KGC1LearnModel, TorchDataset

import time,copy
from collections import defaultdict

from torch.optim.lr_scheduler import ExponentialLR

logger = logging.getLogger(__name__)


# class DataSet(TorchDataset):
# 	def __init__(self, triples):
# 		self.triples = triples
#
# 	def __len__(self):
# 		return len(self.triples)
#
# 	def __getitem__(self, index):
# 		head, relation, tail = self.triples[index]
# 		return head, relation, tail


class Data:

    def __init__(self, data_dir="data/FB15k", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        if data_type == "train":
            data = []
            big_dict = {}
            with open("%s/%s" % (data_dir, "triples"), "r") as f:
                triples = f.read().strip().split("\n")
            with open("%s/%s" % (data_dir, "entities"), "r") as f:
                entities = f.read().strip().split("\n")
                for item in entities:
                    tmp_list = item.split()
                    big_dict[tmp_list[0]] = tmp_list[2]
            for item in triples:
                tmp = item.split()
                data.append([big_dict[tmp[0]],tmp[1],big_dict[tmp[2]] ])
        else:
            with open("%s/%s.txt" % (data_dir, data_type), "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split() for i in data]
        if reverse:
            data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities


class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.,test_mode=False,handle_data=None,model=None,model_dir=None):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.test_mode=test_mode
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        self.model = model
        self.handle_data = handle_data
        self.model_dir = model_dir

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(self.handle_data.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets


    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.handle_data.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

    def jest_test(self):
        print("Testing the model...")
        self.entity_idxs = {self.handle_data.entities[i]: i for i in range(len(self.handle_data.entities))}
        self.relation_idxs = {self.handle_data.relations[i]: i for i in range(len(self.handle_data.relations))}

        train_data_idxs = self.get_data_idxs(self.handle_data.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = self.model(self.handle_data, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        saved_params = torch.load(self.model_dir+'/checkpoint.mdl', map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        print("Starting testing...")
        start_test = time.time()
        self.evaluate(model, self.handle_data.test_data)
        print(time.time() - start_test)

    def train_and_eval(self):
        print("Training the model...")
        self.entity_idxs = {self.handle_data.entities[i]:i for i in range(len(self.handle_data.entities))}
        self.relation_idxs = {self.handle_data.relations[i]:i for i in range(len(self.handle_data.relations))}

        train_data_idxs = self.get_data_idxs(self.handle_data.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = self.model(self.handle_data, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(it)
            print(time.time()-start_train)
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, self.handle_data.valid_data)
                if not it%2:
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, self.handle_data.test_data)
                    sd=copy.copy(model.state_dict())
                    params = {'state_dict': sd}
                    torch.save(params, self.model_dir+'/checkpoint.mdl')
                    print(sd.keys())
                    print('now save checkpoint to project path')
                    print(time.time()-start_test)


@KGC1LearnModel.register("KGC1Learn", "PyTorch")
class KGC1LearnTorch(KGC1LearnModel):
    def __init__(self, name='pytorch-default', graph=None, model=None, args=None):
        self.name = name
        self.graph = graph
        self.args = args
        self.model = model

        torch.backends.cudnn.deterministic = True
        seed = 20
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available:
            torch.cuda.manual_seed_all(seed)
        handle_data = Data(data_dir=args["data_dir"], reverse=True)
        if not handle_data:
            raise Exception("no_data")
        self.experiment = Experiment(num_iterations=args["num_iterations"], batch_size=args["batch_size"], learning_rate=args["learning_rate"],decay_rate=args["dr"], ent_vec_dim=args["edim"], rel_vec_dim=args["rdim"], cuda=args["cuda"],input_dropout=args["input_dropout"], hidden_dropout1=args["hidden_dropout1"],
                        hidden_dropout2=args["hidden_dropout2"], label_smoothing=args["label_smoothing"],
                        test_mode=args["test_mode"],handle_data=handle_data,model=model,model_dir=args["model_dir"])

    def triples_reader(self, ratio=0.01):
        """read from triple data files to id triples"""
        pass

    def triples_generator(self, train_triples, device):
        pass

    def hit_at_k(self, predictions, ground_truth_idx, device, k=10):
        """how many true entities in top k similar ones"""
        pass

    def mrr(self, predictions, ground_truth_idx):
        """Mean Reciprocal Rank"""
        pass

    def evaluate(self,):
        """predicting validation and test set and show performance metrics"""
        self.experiment.train_and_eval()

    def load_model(self, model_path, model, opt):
        """load model from local model file"""
        pass
    def save_model(self, model, optim, epoch, best_score, model_path):
        """save model to local file"""
        pass

    def run(self, dist=False):
        if not self.args["test_mode"]:
            self.evaluate()
        else:
            self.experiment.jest_test()

