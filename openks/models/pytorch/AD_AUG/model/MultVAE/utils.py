import os
import pandas as pd
import numpy as np
from scipy import sparse
import numpy as np, scipy.sparse as sparse
import torch, torch.nn as nn, torch.optim as optim


def ndcg_binary_at_k_batch(x_pred, heldout_batch, k=100):
    batch_users = x_pred.shape[0]
    idx_topk_part = np.argpartition(-x_pred, k, axis=1)
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0
    return ndcg


def recall_at_k_batch(x_pred, heldout_batch, k=10):
    batch_users = x_pred.shape[0]
    idx = np.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, x_true_binary.sum(axis=1))
    recall[np.isnan(recall)] = 0
    return recall


def load_data(data_dir='../../data/ml-latest-small/'):
    with open(os.path.join(data_dir, 'unique_sid.txt'), 'r') as f:
        unique_sid = [line.strip() for line in f]
    with open(os.path.join(data_dir, 'unique_uid.txt'), 'r') as f:
        unique_uid = [line.strip() for line in f]

    n_items = len(unique_sid)
    n_users = len(unique_uid)

    train_data = load_csv_data(os.path.join(data_dir, 'train.csv'), n_users, n_items)

    valid_data = load_csv_data(os.path.join(data_dir, 'valid.csv'), n_users, n_items)

    test_data = load_csv_data(os.path.join(data_dir, 'test.csv'), n_users, n_items)

    assert n_items == train_data.shape[1]
    assert n_items == valid_data.shape[1]
    assert n_items == test_data.shape[1]

    return (n_users, n_items, train_data, valid_data, test_data)


def load_csv_data(csv_file, n_users, n_items):
    tp = pd.read_csv(csv_file)
    # n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data