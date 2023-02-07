"""Select new instances given prediction and retrieval modules"""
import math
import collections
import torch
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.datasets import TUDataset

from utils import scorer


def get_relation_distribution(dataset):
    """Get relation distribution of a dataset

    Args:
      dataset (data.Dataset or list): The dataset to consider
    """
    if isinstance(dataset, TUDataset):
        counter = collections.Counter(dataset.data.y.tolist())
    else:
        counter = collections.Counter([pred for eid, pred, actual in dataset])
    label_distribution = {k: v / len(dataset) for k, v in counter.items()}
    return label_distribution


def split_samples(meta_idxs, dataset=None):
    """Split dataset using idxs

    Args:
        dataset (data.Dataset): Dataset instance
        meta_idxs (list): List of indexes with the form (idx, predict_label, gold_label)
    """
    new_examples, rest_examples = [], []
    new_ids = [idx for idx, pred, actual in meta_idxs]
    total_ids = [i for i in range(len(dataset))]
    rest_ids = list(set(total_ids).difference(set(new_ids)))
    pred_list = [pred for idx, pred, actual in meta_idxs]
    new_examples_truth = [dataset[i] for i in new_ids]
    new_examples = []
    for i in range(len(new_ids)):
        new_examples_truth[i]=new_examples_truth[i][0]
        x = new_examples_truth[i].x 
        edge_index = new_examples_truth[i].edge_index
        edge_attr = new_examples_truth[i].edge_attr
        y = torch.LongTensor([pred_list[i]])
        if edge_attr == None:
            new_examples.append((Data(x=x, y=y, edge_index=edge_index),Data(x=x, y=y, edge_index=edge_index)))
        else:
            new_examples.append((Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr),Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)))

    rest_examples = [dataset[i] for i in rest_ids]
    return new_examples, rest_examples


def intersect_samples(meta_idxs_p, s_retrieve_fn, k_samples, prior_distribution):
    upperbound, meta_idxs, confidence_idxs_s = k_samples, [], []
    while len(meta_idxs) < min(k_samples, len(meta_idxs_p)):
        upperbound = math.ceil(1.25 * upperbound)
        ori_meta_idxs_s = s_retrieve_fn(upperbound, prior_distribution)
        meta_idxs = sorted(set(meta_idxs_p[:upperbound]).intersection(set(ori_meta_idxs_s)))[:k_samples]
        if upperbound > k_samples * 30:  # set a limit for growing upperbound
            break
    print("Unlabel on combination...")
    gold, guess = [actual for _, _, actual in meta_idxs], [pred for _, pred, _ in meta_idxs]
    acc = scorer.score(gold, guess)[0]
    print('acc_inter:'+str(acc))
    return meta_idxs


def select_samples(opt, k_samples, default_distribution, model_p, model_s, train_unlabelset):
    max_upperbound = int(math.ceil(k_samples * opt["selector_upperbound"]))
    # predictor selection
    meta_idxs_p = model_p.retrieve(train_unlabelset, len(train_unlabelset))  # retrieve all the samples
    print("Unlabel on predictor: ")  # Track performance of predictor alone
    gold, guess = [t[2] for t in meta_idxs_p[:k_samples]], [t[1] for t in meta_idxs_p[:k_samples]]
    acc = scorer.score(gold, guess)[0]
    print('acc_p:' + str(acc))

    # for self-training
    if opt["integrate_method"] == "p_only":
        return split_samples(train_unlabelset, meta_idxs_p[:k_samples], opt["batch_size"])

    # selector selection
    label_distribution = None
    if opt["integrate_method"] == "s_only" or max_upperbound == 0:
        label_distribution = default_distribution
    else:
        label_distribution = get_relation_distribution(meta_idxs_p[:max_upperbound])

    def s_retrieve_fn(k_samples, label_distribution):
        return model_s.retrieve(train_unlabelset, k_samples, label_distribution=label_distribution)

    meta_idxs_s = s_retrieve_fn(k_samples, label_distribution)
    print("Unlabel on selector: ")
    gold, guess = [t[2] for t in meta_idxs_s], [t[1] for t in meta_idxs_s]   
    acc = scorer.score(gold, guess)[0]
    print('acc_s:' + str(acc))

    # If we only care about performance of selector
    if opt["integrate_method"] == "s_only":
        return split_samples(train_unlabelset, meta_idxs_s)

    # integrate method
    if opt["integrate_method"] == "intersection":
        meta_idxs = intersect_samples(meta_idxs_p, s_retrieve_fn, k_samples, label_distribution)
    else:
        raise NotImplementedError("integrate_method {} not implemented".format(opt["integrate_method"]))
    return split_samples(meta_idxs, dataset=train_unlabelset)
