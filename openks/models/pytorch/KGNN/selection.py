import math
import collections
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch

from utils import scorer


def get_relation_distribution(dataset):
    """Get distribution of a dataset

    Args:
      dataset (data.Dataset or list): The dataset to consider
    """
    if isinstance(dataset, TUDataset):
        counter = collections.Counter(dataset.data.y.tolist())
    else:
        counter = collections.Counter([pred for eid, pred, actual in dataset])
    label_distribution = {k: v / len(dataset) for k, v in counter.items()}
    return label_distribution


def split_samples(meta_idxs, dataset=None, GK_train_unlabel=None, GK_train_unlabel_y=None):
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
        x = new_examples_truth[i].x 
        edge_index = new_examples_truth[i].edge_index
        edge_attr = new_examples_truth[i].edge_attr
        y = torch.LongTensor([pred_list[i]])
        if edge_attr == None:
            new_examples.append(Data(x=x, y=y, edge_index=edge_index))
        else:
            new_examples.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr))

    rest_examples = [dataset[i] for i in rest_ids]

    GK_new_examples = [GK_train_unlabel[i] for i in new_ids]
    GK_new_examples_y = pred_list
    GK_rest_examples = [GK_train_unlabel[i] for i in rest_ids]
    GK_rest_examples_y = [GK_train_unlabel_y[i] for i in rest_ids]
    return new_examples, rest_examples, \
           GK_new_examples, GK_new_examples_y, \
           GK_rest_examples, GK_rest_examples_y


def split_samples_p(meta_idxs, dataset=None, GK_train_unlabel=None, GK_train_unlabel_y=None):
    """all the samples predicted by gnn"""
    new_ids = [idx for idx, pred, actual in meta_idxs]
    pred_list = [pred for idx, pred, _ in meta_idxs]

    GK_new_examples = [GK_train_unlabel[i] for i in new_ids]
    GK_new_examples_y = pred_list
    return GK_new_examples, GK_new_examples_y


def split_samples_q(meta_idxs, dataset=None, GK_train_unlabel=None, GK_train_unlabel_y=None):
    """all the samples predicted by gk"""
    new_examples, new_ids = [], [idx for idx, pred, actual in meta_idxs]
    pred_list = [pred for idx, pred, _ in meta_idxs][:len(dataset)]
    pred_tensor = torch.tensor(pred_list).type_as(dataset.data.y)
    new_ids = new_ids[:len(dataset)]
    new_examples = dataset[new_ids]
    new_examples.data.y[new_examples.indices()] = pred_tensor
    return new_examples


def intersect_samples(meta_idxs_p, s_retrieve_fn, k_samples, prior_distribution):
    upperbound, meta_idxs = k_samples, []
    while len(meta_idxs) < min(k_samples, len(meta_idxs_p)):
        upperbound = math.ceil(1.25 * upperbound)
        ori_meta_idxs_s = s_retrieve_fn(upperbound, prior_distribution)
        meta_idxs = sorted(set(meta_idxs_p[:upperbound]).intersection(set(ori_meta_idxs_s)))[:k_samples]
        if upperbound > k_samples * 30: # set a limit for growing upperbound
            break
    print("Unlabel on combination...")
    gold, guess = [actual for _, _, actual in meta_idxs], [pred for _, pred, _ in meta_idxs]
    acc = scorer.score(gold, guess)[0]
    print("retrieve_inter_acc:" + str(acc))
    return meta_idxs


def select_samples(opt, k_samples, label_distribution,
                   model_p=None, model_q=None, 
                   train_unlabelset=None, train_unlabel_data=None,
                   GK_train_unlabel=None, GK_train_unlabel_y=None):
    max_upperbound = math.ceil(k_samples * opt["gk_upperbound"])
    # gnn selection
    meta_idxs_p = model_p.retrieve(train_unlabelset, len(train_unlabelset)) # retrieve all the samples
    print("Unlabel on gnn: ")
    gold, guess = [t[2] for t in meta_idxs_p], [t[1] for t in meta_idxs_p]
    acc = scorer.score(gold, guess)[0]
    print("retrieve_p_acc:" + str(acc))

    def q_retrieve_fn(k_samples, label_distribution=None):
        return model_q.retrieve(train_unlabel_data, len(train_unlabel_data)) # k_samples, label_distribution=label_distribution

    meta_idxs_q = q_retrieve_fn(k_samples, label_distribution)
    print("Unlabel on gk: ")
    gold, guess = [t[2] for t in meta_idxs_q], [t[1] for t in meta_idxs_q]   
    acc = scorer.score(gold, guess)[0]
    print("retrieve_q_acc:" + str(acc))

    if opt["integrate_method"] == "intersection":
        meta_idxs = sorted(set(meta_idxs_p[:max_upperbound]).intersection(set(meta_idxs_q[:max_upperbound])))[:k_samples]
    # for self-training
    elif opt["integrate_method"] == "p_only":
        return split_samples(train_unlabelset, meta_idxs_p[:k_samples], opt["batch_size"])

    # gk selection
    # label_distribution = None
    # if opt["integrate_method"] == "q_only" or max_upperbound == 0:
    #     label_distribution = default_distribution
    # else:
    #     label_distribution = get_relation_distribution(meta_idxs_p[:max_upperbound])
    
    return split_samples(meta_idxs,
                         dataset=train_unlabelset,
                         GK_train_unlabel=GK_train_unlabel, 
                         GK_train_unlabel_y=GK_train_unlabel_y)
