"""
Utility functions for torch.
"""
import torch
from torch.optim import Optimizer
import numpy as np
import json


def get_optimizer(name, parameters, lr, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "adagrad":
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "adamax":
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# model IO


def save(model, optimizer, opt, filename):
    params = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": opt}
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")


def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump["model"])
    if optimizer is not None:
        optimizer.load_state_dict(dump["optimizer"])
    opt = dump["config"]
    return model, optimizer, opt


def load_config(filename):
    dump = torch.load(filename)
    return dump["config"]


def arg_max(l):
    bvl, bid = -1, -1
    for k in range(len(l)):
        if l[k] > bvl:
            bvl = l[k]
            bid = k
    return bid, bvl


def save_config(config, path, verbose=True):
    with open(path, "w") as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config



"""Preprocess data"""
import os
import random
import collections
from collections import Counter
import numpy as np
from sklearn.utils import Bunch


def load_data(ds_name, root, is_symmetric=True, produce_labels_nodes=True):
    ngc = dict()
    # edge line correspondence
    elc = dict()
    # dictionary that keeps sets of edges
    Graphs = dict()
    # dictionary of labels for nodes
    node_labels = dict()
    # dictionary of labels for edges
    edge_labels = dict()

    # Associate graphs nodes with indexes
    with open(os.path.join(root, "%s/%s/raw/%s_graph_indicator.txt"%(ds_name, ds_name, ds_name)), "r") as f:
        for (i, line) in enumerate(f, 1):
            ngc[i] = int(line[:-1])
            if int(line[:-1]) not in Graphs:
                Graphs[int(line[:-1])] = set()
            if int(line[:-1]) not in node_labels:
                node_labels[int(line[:-1])] = dict()
            if int(line[:-1]) not in edge_labels:
                edge_labels[int(line[:-1])] = dict()
  
    # Extract graph edges
    with open(os.path.join(root, "%s/%s/raw/%s_A.txt"%(ds_name, ds_name, ds_name)), "r") as f:
        for (i, line) in enumerate(f, 1):
            edge = line[:-1].replace(' ', '').split(",")
            elc[i] = (int(edge[0]), int(edge[1]))
            Graphs[ngc[int(edge[0])]].add((int(edge[0]), int(edge[1])))
            if is_symmetric:
                Graphs[ngc[int(edge[1])]].add((int(edge[1]), int(edge[0])))

    # Extract node labels
    node_labels_path = os.path.join(root, "%s/%s/raw/%s_node_labels.txt"%(ds_name, ds_name, ds_name))
    if os.path.exists(node_labels_path):
        with open(node_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = int(line[:-1])
    elif produce_labels_nodes:
        for i in range(1, len(Graphs)+1):
            node_labels[i] = dict(Counter(s for (s, d) in Graphs[i] if s != d))
    
    Gs = list()
    for i in range(1, len(Graphs)+1):
        Gs.append([Graphs[i], node_labels[i], edge_labels[i]])
    
    classes = []
    with open(os.path.join(root, "%s/%s/raw/%s_graph_labels.txt"%(ds_name, ds_name, ds_name)), "r") as f:
        for line in f:
            classes.append(int(line[:-1])) 

    classes = np.array(classes, dtype=np.int)
    return Gs
    # return Bunch(data=Gs, y=classes)