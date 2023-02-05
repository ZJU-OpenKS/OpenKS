import argparse
import random
import os

import numpy as np, scipy.sparse as sparse
import torch, torch.nn as nn, torch.optim as optim
from vae import MultVAE
from dae import MultDAE
from utils import load_data, ndcg_binary_at_k_batch, recall_at_k_batch

ARG = argparse.ArgumentParser()
ARG.add_argument('--data', type=str, required=True)
ARG.add_argument('--model', type=str, default='multvae',
                 help='multvae, multdae')
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
ARG.add_argument('--rg', type=float, default=0.0,
                 help='L2 regularization.')
ARG.add_argument('--keep', type=float, default=0.5,
                 help='Keep probability for dropout, in (0,1].')
ARG.add_argument('--beta', type=float, default=0.2,
                 help='Strength of disentanglement, in (0,oo).')
ARG.add_argument('--std', type=float, default=0.075,
                 help='Standard deviation of the Gaussian prior.')
ARG.add_argument('--dfac', type=int, default=100,
                 help='Dimension of each facet.')
ARG.add_argument('--intern', type=int, default=50,
                 help='Report interval.')
ARG.add_argument('--log', type=str, default=None,
                 help='The log file path.')
ARG.add_argument('--save_name', type=str, default=None,
                 help='Save model to ./saved_models/')
ARG.add_argument('--patience', type=int, default=50,
                 help='extra iterations before early-stopping')
ARG = ARG.parse_args()

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def valid_vae(vad_data_tr, vad_data_te, VAE, arg, device):
    VAE.eval()
    n_vad = vad_data_tr.shape[0]
    idxlist_vad = list(range(n_vad))
    ndcg100_list, recall20_list, recall50_list = [], [], []
    for bnum, st_idx in enumerate(range(0, n_vad, arg.batch)):
        end_idx = min(st_idx + arg.batch, n_vad)
        x = vad_data_tr[idxlist_vad[st_idx:end_idx]]
        # if sparse.isspmatrix(x):
        #     x = x.toarray()
        # x = torch.Tensor(x.astype(np.float32)).to(device)
        if arg.model == 'multvae':
            logits, _, _ = VAE(x, is_train=False)
        elif arg.model == 'multdae':
            logits, _ = VAE(x, is_train=False)
        logits[x.nonzero(as_tuple=True)] = -np.inf
        ndcg100_list.append(
            ndcg_binary_at_k_batch(logits.cpu().detach().numpy(), vad_data_te[idxlist_vad[st_idx:end_idx]], 100)
        )
        recall20_list.append(
            recall_at_k_batch(logits.cpu().detach().numpy(), vad_data_te[idxlist_vad[st_idx:end_idx]], 20)
        )
        recall50_list.append(
            recall_at_k_batch(logits.cpu().detach().numpy(), vad_data_te[idxlist_vad[st_idx:end_idx]], 50)
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
    # n_items = train_data.shape[1]

    idxlist = list(range(n_train))

    num_batches = int(np.ceil(float(n_train) / arg.batch))
    total_anneal_steps = 5 * num_batches

    if arg.model == 'multvae':
        VAE = MultVAE(arg, n_items, device).to(device)
    elif arg.model == 'multdae':
        VAE = MultDAE(arg, n_items, device).to(device)

    opt = optim.Adam(VAE.parameters(), lr=arg.lr, weight_decay=arg.rg)

    best_ndcg100, best_recall20, best_recall50 = 0, 0, 0
    best_epoch = -1
    update_count = 0

    if sparse.isspmatrix(train_data):
        train_data = train_data.toarray()
    train_data = torch.Tensor(train_data.astype(np.float32)).to(device)

    for epoch in range(arg.epoch):
        np.random.shuffle(idxlist)
        rec_losses, kls, regs = [], [], []
        VAE.train()
        for bnum, st_idx in enumerate(range(0, n_train, ARG.batch)):
            end_idx = min(st_idx + ARG.batch, n_train)
            x = train_data[idxlist[st_idx:end_idx]]
            anneal = (min(arg.beta, 1. * update_count / total_anneal_steps))\
                if total_anneal_steps > 0 else arg.beta

            if arg.model == 'multvae':
                _, recon_loss, kl = VAE(x, is_train=True)
                neg_elbo = recon_loss + anneal * kl
                # neg_elbo = recon_loss + anneal * kl + reg_var

                rec_losses.append(recon_loss.detach().cpu().numpy())
                kls.append(kl.detach().cpu().numpy())
                regs.append(0)
            elif arg.model == 'multdae':
                _, recon_loss = VAE(x, is_train=True)
                neg_elbo = recon_loss
                rec_losses.append(recon_loss.detach().cpu().numpy())
                kls.append(0)
                regs.append(0)

            opt.zero_grad()
            neg_elbo.backward()
            opt.step()
            update_count += 1

        ndcg100, recall20, recall50 = valid_vae(train_data, valid_data, VAE, arg, device)

        if ndcg100 > best_ndcg100:
            best_ndcg100 = ndcg100
            best_recall20 = recall20
            best_recall50 = recall50
            best_epoch = epoch
            if arg.save_name is not None:
                torch.save(VAE.state_dict(), './saved_models/'+arg.save_name+'.pth')
            ndcg_te100, recall_te20, recall_te50 = valid_vae(train_data, test_data, VAE, arg, device)

        if (epoch + 1) % arg.intern == 0:
            if arg.log is not None:
                with open(arg.log, 'a') as f:
                    f.write(f'Epoch {epoch}\n\tRecon_loss: {np.mean(rec_losses)}, KL: {np.mean(kls)}, L2: {np.mean(regs)}')
                    f.write(f_str.format(epoch+1, arg.epoch, \
                        ndcg100, recall20, recall50, best_ndcg100, best_recall20, best_recall50))
                    f.write(f'NDCG100_test:\t{ndcg_te100}\nRecall20_test:\t{recall_te20}\nRecall50_test:\t{recall_te50}')
            else:
                print(f'\nEpoch {epoch+1}\n\tRecon_loss: {np.mean(rec_losses)}, KL: {np.mean(kls)}, L2: {np.mean(regs)}')
                print(f_str.format(epoch+1, arg.epoch, \
                        ndcg100, recall20, recall50, best_ndcg100, best_recall20, best_recall50))
                print(f'NDCG100_test:\t{ndcg_te100}\nRecall20_test:\t{recall_te20}\nRecall50_test:\t{recall_te50}', flush=True)

        if epoch - best_epoch >= arg.patience:
            print('Stop training after %i epochs without improvement on validation.' % arg.patience)
            break

    return ndcg_te100, recall_te20, recall_te50


if __name__ == '__main__':
    seed_torch(ARG.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (n_users, n_items, train_data, valid_data, test_data) = load_data(ARG.data)
    print(f'\nData loaded from `{ARG.data}` complete:\n')
    print(f'\tn_items: {n_items}\n\tTrain size: {train_data.shape[0]}\n\tValid size: {valid_data.shape[0]}'
          f'\n\tTest size: {test_data.shape[0]}\n')
    ndcg_te100, recall_te20, recall_te50 = train(train_data, valid_data, test_data, ARG, device)
    print(f'Best test NDCG:{ndcg_te100}\nBest test recall@20:{recall_te20}\nBest test recall@50: {recall_te50}',
          flush=True)

