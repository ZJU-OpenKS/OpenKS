import warnings
warnings.filterwarnings('ignore')

import os
import math
import random
from shutil import copyfile
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader, DataListLoader

from utils import torch_utils, scorer
from utils.torch_utils import arg_max

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataset, evaluate_type="gnn"):
    if evaluate_type == "gnn":
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        all_preds = []
        all_golds = []
        all_loss = 0
        for data in loader:
            data.to(device)
            target = data.y.long() # in case data.y is float
            _, preds, loss = model.predict(data)
            all_preds += preds
            all_golds += target.tolist()
            all_loss += loss
        acc = scorer.score(all_golds, all_preds)[0]
        return acc, all_loss

    elif evaluate_type == "gk":
        labels = [d[2] for d in dataset]
        target = torch.LongTensor(labels).to(device)
        _, preds, loss = model.predict(dataset)
        acc = scorer.score(labels, preds)[0]
        return acc, loss


class Trainer(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, model, model_type="gnn"):
        self.opt = opt
        self.model_type = model_type
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        self.model.to(device)
        # self.model = torch_geometric.nn.DataParallel(self.model, device_ids=[0, 1])
        self.criterion.to(device)

        self.optimizer = torch_utils.get_optimizer(opt["optim"], self.parameters, opt["lr"], opt["weight_decay"])


    def train(self, dataset_train, dataset_val):
        opt = self.opt.copy()
        if opt["model_save_dir"] == "gnn":
            train_label_loader = DataLoader(dataset_train, batch_size=opt["batch_size"], shuffle=True)
            # train_label_loader = DataListLoader(dataset_train, batch_size=opt["batch_size"], shuffle=True)
        val_acc_history = []

        # start training
        epoch = 0
        patience = 0
        while True:
            epoch = epoch + 1
            train_loss = 0
            train_correct = 0

            if opt["model_save_dir"] == "gnn":
                for data in train_label_loader:
                    loss, correct = self.update(data)
                    train_loss += loss
                    train_correct += correct
            else:
                random.shuffle(dataset_train)
                for i in range(0, len(dataset_train), opt["batch_size"]):
                    data = dataset_train[i:i+opt["batch_size"]]
                    loss, correct = self.update(data)
                    train_loss += loss
                    train_correct += correct

            # print("Evaluating on val set...")
            if self.model_type == "gnn":
                val_acc, val_loss = evaluate(self, dataset_val, evaluate_type="gnn")
            else:
                val_acc, val_loss = evaluate(self, dataset_val, evaluate_type="gk")

            # print training information
            train_acc = train_correct / len(dataset_train)
            train_loss = train_loss / len(dataset_train)
            val_loss = val_loss / len(dataset_val)
            print("epoch {}: train_loss = {:.6f}, val_loss = {:.6f}, train_acc = {:.6f}, val_acc = {:.6f}".format(epoch, \
                    train_loss, val_loss, train_acc, val_acc))

            # save the current model
            model_file = opt["model_save_dir"] + "/checkpoint_epoch_{}.pt".format(epoch)
            self.save(model_file, epoch)
            if epoch == 1 or val_acc > max(val_acc_history):  # new best
                path = opt["model_save_dir"] + "/best_model.pt"
                if os.path.exists(path):
                    os.remove(path)
                copyfile(model_file, path)
                # print("new best model saved.")
                patience = 0
            else:
                patience = patience + 1
            
            if epoch % opt["save_epoch"] != 0:
                os.remove(model_file)

            val_acc_history += [val_acc]

            if self.model_type == "gnn":
                if opt["patience"] != 0:
                    if patience == opt["patience"] and epoch > opt["gnn_num_epoch"]: break
                else:
                    if epoch == opt["gnn_num_epoch"]: break
            else:
                if opt["patience"] != 0:
                    if patience == opt["patience"] and epoch > opt["memnn_num_epoch"]: break
                else:
                    if epoch == opt["memnn_num_epoch"]: break
        print("Training ended with {} epochs.".format(epoch))


    # train the model with a batch
    def update(self, data):
        """ Run a step of forward and backward model update. """
        self.model.train()
        self.optimizer.zero_grad()
      
        if self.opt["model_save_dir"] == "gnn":
            data.to(device)
            # x, edge_index, batch = data.x, data.edge_index, data.batch
            # if x is None or x.shape[1] == 0: # torch.Size([num, 0])
            #     x = torch.ones((batch.shape[0], 1)).to(device)
            target = data.y.long() # in case data.y is float

            logits = self.model(data)
        else:
            x = torch.FloatTensor([d[0] for d in data]).to(device)
            q = torch.FloatTensor([d[1] for d in data]).to(device)
            target = torch.LongTensor([d[2] for d in data]).to(device)
            logits = self.model(x, q)

        loss = self.criterion(logits, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt["max_grad_norm"])
        self.optimizer.step()

        probs = F.softmax(logits, dim=-1)
        _, preds = probs.max(dim=1)
        correct = preds.eq(target).sum().item()
        loss_train = loss.item()
        return loss_train, correct


    def predict(self, data):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """

        self.model.eval()
        if self.opt["model_save_dir"] == "gnn":
            # x, edge_index, batch = data.x, data.edge_index, data.batch
            # if x is None or x.shape[1] == 0: # torch.Size([num, 0]):
            #     x = torch.ones((batch.shape[0], 1)).to(device)
            logits = self.model(data)
            target = data.y.long() # in case data.y is float
        else: # "gk"
            x = torch.FloatTensor([d[0] for d in data]).to(device)
            q = torch.FloatTensor([d[1] for d in data]).to(device)
            target = torch.LongTensor([d[2] for d in data]).to(device)
            logits = self.model(x, q)

        loss = None if target is None else self.criterion(logits, target).item()

        probs = F.softmax(logits, dim=-1).tolist()
        preds = np.argmax(probs, axis=1).tolist()
        return probs, preds, loss


    def retrieve(self, dataset, k_samples, label_distribution=None):
        probs, target = [], []
        if self.opt["model_save_dir"] == "gnn":
            train_unlabel_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            for data in train_unlabel_loader:
                data.to(device)
                probs += self.predict(data)[0]
                target += data.y.long().tolist()
        else: # "gk"
            probs = self.predict(dataset)[0]
            target = [d[2] for d in dataset]

        meta_idxs = []
        num_instance = len(dataset)

        if label_distribution:
            label_distribution = {k: math.ceil(v * k_samples) for k, v in label_distribution.items()}

        ranking = list(zip(range(num_instance), probs))
        ranking = sorted(ranking, key=lambda x: max(x[1]), reverse=True)
        # selection
        for eid, prob in ranking:
            if len(meta_idxs) == k_samples:
                break
            class_id, _ = arg_max(prob)
            if label_distribution:
                if not label_distribution[class_id]:
                    continue
                label_distribution[class_id] -= 1
            meta_idxs.append((eid, class_id, target[eid]))
        return meta_idxs


    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    # save the model
    def save(self, filename, epoch):
        params = {
            "model": self.model.state_dict(),  # model parameters
            "encoder": self.model.encoder.state_dict(),
            "classifier": self.model.classifier.state_dict(),
            "config": self.opt,  # options
            "epoch": epoch,  # current epoch
            "model_type": self.model_type  # current epoch
        }
        try:
            torch.save(params, filename)
            # print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    # load the model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.encoder.load_state_dict(checkpoint["encoder"])
        self.model.classifier.load_state_dict(checkpoint["classifier"])
        self.opt = checkpoint["config"]
        self.model_type = checkpoint["model_type"]
        self.criterion = nn.CrossEntropyLoss()
