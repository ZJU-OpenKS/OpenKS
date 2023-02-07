import argparse
import random
import os

import numpy as np, scipy.sparse as sparse
import torch, torch.optim as optim
from D_CDAE import ADV as ADV_D
from M_CDAE import ADV as ADV_M
from utils import load_data, ndcg_binary_at_k_batch, recall_at_k_batch

ARG = argparse.ArgumentParser()
ARG.add_argument('--data', type=str, required=True)
ARG.add_argument('--mode', type=str, default='trn',
                 help='trn/tst/vis, for training/testing/visualizing.')
ARG.add_argument('--logdir', type=str, default='./runs/')
ARG.add_argument('--seed', type=int, default=98765,
                 help='Random seed. Ignored if < 0.')
ARG.add_argument('--epoch', type=int, default=5000,
                 help='Number of training epochs.')
ARG.add_argument('--type', type=str, default='data',
                 help='data/model')
ARG.add_argument('--batch', type=int, default=1024,
                 help='Training batch size.')
ARG.add_argument('--lr', type=float, default=1e-3,
                 help='Initial learning rate.')
ARG.add_argument('--lr_aug', type=float, default=1e-3,
                 help='Learning rate of augmenter.')
ARG.add_argument('--rg', type=float, default=0.0,
                 help='L2 regularization.')
ARG.add_argument('--rg_aug', type=float, default=1e4,
                 help='L2 regularization for augmenter.')
ARG.add_argument('--alpha', type=float, default=1,
                 help='Weight of mutual information, in (0,oo).')
ARG.add_argument('--alpha_aug', type=float, default=1,
                 help='Weight of mutual information of augmenter, in (0,oo).')
ARG.add_argument('--keep', type=float, default=0.5,
                 help='Keep probability for dropout, in (0,1].')
ARG.add_argument('--beta', type=float, default=0.2,
                 help='Strength of disentanglement, in (0,oo).')
ARG.add_argument('--tau_aug', type=float, default=1.0,
                 help='Temperature of Gumbel-Max reparametrization, in (0,oo).')
# ARG.add_argument('--tau', type=float, default=0.1,
#                  help='Temperature of sigmoid/softmax, in (0,oo).')
ARG.add_argument('--std', type=float, default=0.075,
                 help='Standard deviation of the Gaussian prior.')
ARG.add_argument('--dfac', type=int, default=100,
                 help='Dimension of each facet.')
ARG.add_argument('--proj_hid', type=int, default=100,
                 help='Dimension of projection head.')
# ARG.add_argument('--nogb', action='store_true', default=False,
#                  help='Disable Gumbel-Softmax sampling.')
ARG.add_argument('--intern', type=int, default=50,
                 help='Report interval.')
ARG.add_argument('--log', type=str, default=None,
                 help='The log file path.')
ARG.add_argument('--save_name', type=str, default=None,
                 help='Save model to ./saved_models/')
ARG.add_argument('--patience', type=int, default=50,
                 help='extra iterations before early-stopping')
ARG = ARG.parse_args()


def print_arg(arg):
    arg_format = f'''
=========================================================================================
|   Epoch: {arg.epoch}  Batch: {arg.batch}             
|   Rec_lr: {arg.lr}    Aug_lr: {arg.lr_aug}
|   Aug_reg_weight: {arg.rg_aug}    
|   Alpha: {arg.alpha}  Alpha_aug: {arg.alpha_aug}
|   Log file path: {arg.log}
=========================================================================================
'''
    return arg_format


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def valid_vae(vad_data_tr, vad_data_te, model, arg, device):
    VAE = model.D
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
        u = torch.LongTensor([idx for idx in idxlist_vad[st_idx:end_idx]]).to(device)
        # logits, _, _ = VAE(u, x, x, is_train=False)
        if arg.type == 'model':
            aug = model.G(x, x, is_train=False)
            gumbel = model.GumbelMax(aug)
            aug_graph = torch.zeros_like(x).to(device)
            aug_graph[x == 0] = (1 - (1 - x) * gumbel)[x == 0]
            aug_graph[x == 1] = (x * gumbel)[x == 1]

        if arg.type == 'data':
            logits, _, _ = VAE(x, x, is_train=False)
        elif arg.type == 'model':
            logits, _, _ = VAE(aug_graph, x, is_train=False)

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
    return np.mean(np.concatenate(ndcg100_list)), np.mean(np.concatenate(recall20_list)), np.mean(
        np.concatenate(recall50_list))


def train(train_data, valid_data, test_data, arg, device):
    f_str = '''
---------------------------------------------
    Epoch {} / {}
---------------------------------------------
VAE loss: {} = {} + beta * {} - alpha * {}
Aug loss: {} = alpha_aug * {} + reg_aug * {}
NDCG@100:               {}
Recall@20:             {}
Recall@50:             {}
Best NDCG@100:          {}
Best Recall@20:        {}
Best Recall@50:        {}
    '''

    n_train = train_data.shape[0]
    # n_valid = valid_data.shape[0]
    # n_item = train_data.shape[1]
    idxlist = list(range(n_train))

    if arg.type == 'data':
        model = ADV_D(n_users, n_items, arg, device).to(device)
    elif arg.type == 'model':
        model = ADV_M(n_users, n_items, arg, device).to(device)

    opt = optim.Adam(model.D.parameters(), lr=arg.lr, weight_decay=arg.rg)
    opt_aug = optim.Adam(model.G.parameters(), lr=arg.lr_aug, weight_decay=arg.rg)

    update_count = 0
    num_batches = int(np.ceil(float(n_train) / arg.batch))
    total_anneal_steps = 5 * num_batches
    best_ndcg100, best_recall20, best_recall50 = 0, 0, 0
    best_epoch = -1
    if arg.log is not None:
        with open(arg.log, 'a') as f:
            f.write(print_arg(arg))

    if sparse.isspmatrix(train_data):
        train_data = train_data.toarray()
    train_data = torch.Tensor(train_data.astype(np.float32)).to(device)
    arg.rg_aug_or = arg.rg_aug

    for epoch in range(arg.epoch):
        np.random.shuffle(idxlist)
        rec_losses, aug_losses, mi_recs, mi_augs = [], [], [], []
        recons, kls, reg_augs, drops, adds = [], [], [], [], []
        for bnum, st_idx in enumerate(range(0, n_train, arg.batch)):
            end_idx = min(st_idx + arg.batch, n_train)
            user_list = idxlist[st_idx:end_idx]
            x = train_data[user_list]
            u = torch.LongTensor(user_list).to(device)

            anneal = (min(arg.beta, 1. * update_count / total_anneal_steps)) \
                if total_anneal_steps > 0 else arg.beta
            update_count += 1

            rec_loss, aug_loss, mi_rec, mi_aug, recon, kl, reg_aug, drop, add = \
                model(u, x, anneal, opt, opt_aug, is_train=True)

            rec_losses.append(rec_loss.detach().cpu().numpy())
            aug_losses.append(aug_loss.detach().cpu().numpy())
            mi_recs.append(mi_rec.detach().cpu().numpy())
            mi_augs.append(mi_aug.detach().cpu().numpy())
            recons.append(recon.detach().cpu().numpy())
            kls.append(kl.detach().cpu().numpy())
            reg_augs.append(reg_aug.detach().cpu().numpy())
            drops.append(drop.detach().cpu().numpy())
            adds.append(add.detach().cpu().numpy())

        ndcg100, recall20, recall50 = valid_vae(train_data, valid_data, model, arg, device)
        arg.rg_aug = arg.rg_aug * 0.999

        if ndcg100 > best_ndcg100:
            best_ndcg100 = ndcg100
            best_recall20 = recall20
            best_recall50 = recall50
            best_epoch = epoch
            if arg.save_name is not None:
                torch.save(model.state_dict(), './saved_models/' + arg.save_name + '_args.pth')
            ndcg_te100, recall_te20, recall_te50 = valid_vae(train_data, test_data, model, arg, device)

        if (epoch + 1) % arg.intern == 0:
            if arg.log is not None:
                with open(arg.log, 'a') as f:
                    f.write(f_str.format(epoch + 1, arg.epoch, np.mean(rec_losses),
                                         np.mean(recons), np.mean(kls), np.mean(mi_recs),
                                         np.mean(aug_losses), np.mean(mi_augs), np.mean(reg_augs),
                                         ndcg100, recall20, recall50, best_ndcg100, best_recall20, best_recall50))
                    f.write(
                        f'NDCG100_test:\t{ndcg_te100}\nRecall20_test:\t{recall_te20}\nRecall50_test:\t{recall_te50}\n')
            else:
                print(f_str.format(epoch + 1, arg.epoch, np.mean(rec_losses),
                                   np.mean(recons), np.mean(kls), np.mean(mi_recs),
                                   np.mean(aug_losses), np.mean(mi_augs), np.mean(reg_augs),
                                   ndcg100, recall20, recall50, best_ndcg100, best_recall20, best_recall50))
                print(f'NDCG100_test:\t{ndcg_te100}\nRecall20_test:\t{recall_te20}\nRecall50_test:\t{recall_te50}')
                print(f'Drop: {np.mean(drops)}\tAdds:{np.mean(adds)}', flush=True)
                print(f'Reg_Aug: {arg.rg_aug}', flush=True)

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

