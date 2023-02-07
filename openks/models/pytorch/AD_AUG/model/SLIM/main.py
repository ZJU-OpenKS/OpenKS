import argparse
import random
import os

import numpy as np
import torch
from slim import SLIM
from utils import load_data, ndcg_binary_at_k_batch, recall_at_k_batch

ARG = argparse.ArgumentParser()
ARG.add_argument('--data', type=str, required=True)
ARG.add_argument('--mode', type=str, default='trn',
                 help='trn/tst/vis, for training/testing/visualizing.')
ARG.add_argument('--logdir', type=str, default='./runs/')
ARG.add_argument('--seed', type=int, default=98765,
                 help='Random seed. Ignored if < 0.')
ARG.add_argument('--epoch', type=int, default=2000,
                 help='Number of training epochs.')
ARG.add_argument('--batch', type=int, default=1024,
                 help='Training batch size.')
ARG.add_argument('--lr', type=float, default=1e-3,
                 help='Initial learning rate.')
ARG.add_argument('--l1_reg', type=float, default=1e-3,
                 help='L1_reg.')
ARG.add_argument('--l2_reg', type=float, default=1e-3,
                 help='L2_reg.')
ARG.add_argument('--rg', type=float, default=0.0,
                 help='L2 regularization.')
ARG.add_argument('--keep', type=float, default=0.5,
                 help='Keep probability for dropout, in (0,1].')
ARG.add_argument('--beta', type=float, default=0.2,
                 help='Strength of disentanglement, in (0,oo).')
ARG.add_argument('--dfac', type=int, default=100,
                 help='Dimension of each facet.')
ARG.add_argument('--topk', type=int, default=100,
                 help='Topk.')
ARG.add_argument('--intern', type=int, default=50,
                 help='Report interval.')
ARG.add_argument('--log', type=str, default=None,
                 help='The log file path.')
ARG.add_argument('--save_name', type=str, default=None,
                 help='Save model to ./saved_models/')
ARG.add_argument('--patience', type=int, default=50,
                 help='extra iterations before early-stopping')
ARG = ARG.parse_args()

def seed_torch(seed=1453):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def valid_vae(vad_data_tr, vad_data_te, model, arg, device):
    model.eval()
    ndcg100_list, recall20_list, recall50_list = [], [], []
    logits = model.predict(vad_data_tr, 1024)
    logits[vad_data_te.nonzero(as_tuple=True)] = -np.inf
    ndcg100_list.append(
        ndcg_binary_at_k_batch(logits, vad_data_te, 100)
    )
    recall20_list.append(
        recall_at_k_batch(logits, vad_data_te, 20)
    )
    recall50_list.append(
        recall_at_k_batch(logits, vad_data_te, 50)
    )
    return np.mean(np.concatenate(ndcg100_list)), np.mean(np.concatenate(recall20_list)), np.mean(np.concatenate(recall50_list))


def train(train_data, valid_data, test_data, arg, device):
    f_str = '''
---------------------------------------------
    Epoch {} / {}
---------------------------------------------
NDCG@100:               {}
Recall@20:             {}
Recall@50:             {}
Best NDCG@100:          {}
Best Recall@20:        {}
Best Recall@50:        {}

    '''
    n_train = train_data.shape[0]
    idxlist = list(range(n_train))


    model = SLIM(arg, n_users, n_items, device).to(device)

    best_ndcg100, best_recall20, best_recall50 = 0, 0, 0

    np.random.shuffle(idxlist)
    rec_losses, kls, regs = [], [], []
    model.train()
    recon_loss = model(train_data)
    ndcg100, recall20, recall50 = valid_vae(train_data, valid_data, model, arg, device)
    ndcg_te100, recall_te20, recall_te50 = valid_vae(train_data, test_data, model, arg, device)
    print(f'\nEpoch {1}\n\tRecon_loss: {np.mean(rec_losses)}, KL: {np.mean(kls)}, L2: {np.mean(regs)}')
    print(f_str.format(1, arg.epoch, \
            ndcg100, recall20, recall50, best_ndcg100, best_recall20, best_recall50))
    print(f'NDCG100_test:\t{ndcg_te100}\nRecall20_test:\t{recall_te20}\nRecall50_test:\t{recall_te50}', flush=True)

    return ndcg_te100, recall_te20, recall_te50


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (n_users, n_items, train_data, valid_data, test_data) = load_data(ARG.data)
    print(f'\nData loaded from `{ARG.data}` complete:\n')
    print(f'\tn_items: {n_items}\n\tTrain size: {train_data.shape[0]}\n\tValid size: {valid_data.shape[0]}'
          f'\n\tValid size: {test_data.shape[0]}\n')
    ndcg_te100, recall_te20, recall_te50 = train(train_data, valid_data, test_data, ARG, device)
    print(f'Best test NDCG:{ndcg_te100}\nBest test recall@20:{recall_te20}\nBest test recall@50: {recall_te50}',
          flush=True)