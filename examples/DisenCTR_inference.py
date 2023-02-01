import argparse, random, os
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from openks.models.pytorch.DisenCTR.ctr_model import CTR_Model_HAW
from openks.models.pytorch.DisenCTR.dataloader import Haw_data, collate_HAW
from sklearn.metrics import roc_auc_score, ndcg_score, log_loss
import logging
import sys
from torch.utils.data import dataloader
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

ARG = argparse.ArgumentParser()
ARG.add_argument('--epoch', type = int, default = 140,
                 help = 'Epoch num.')
ARG.add_argument('--seed', type = int, default = 98765,
                 help = 'Random seed.')
ARG.add_argument('--batch', type = int, default = 1024,
                 help = 'Training batch size.')
ARG.add_argument('--data', type = str, default = 'ML',
                 help = 'Training dataset.')
ARG.add_argument('--K', type = int, default = 4,
                 help = 'Numer of disentangled intentions.')
ARG.add_argument('--beta', type = float, default = 0.5,
                 help = 'Hyper beta for softplus.')
ARG.add_argument('--tau', type = float, default = 1,
                 help = 'Hyper tau for Gumbel softmax.')
ARG.add_argument('--alpha', type = float, default = 1,
                 help = 'Hyper alpha for target embedding.')
ARG.add_argument('--nConvs', type = int, default = 2,
                 help = 'Numer of conv layers.')
ARG.add_argument('--embed', type = int, default = 64,
                 help = 'U/I embedding size.')
ARG.add_argument('--hid', type = int, default = 256,
                 help = 'Recommending hidden size.')
ARG.add_argument('--patience', type = int, default = 20,
                 help = 'Early stopping patience.')
ARG.add_argument('--lr', type = float, default = 1e-3,
                 help = 'Learning rate.')
ARG.add_argument('--cudaID', type = str, default = None,
                 help = 'Denote cudaID.')
ARG.add_argument('--log', type = str, default = None,
                 help = 'Log file path.')
ARG.add_argument('--save', type = str, default = 'model/',
                 help = 'path to save the final model')

ARG = ARG.parse_args()


def cal_ndcg(predicts, labels, user_ids, k):
    d = {'user': np.squeeze(user_ids), 'predict': np.squeeze(predicts), 'label': np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()

    ndcg = []
    for user_id in user_unique:
        user_srow = df.loc[df['user'] == user_id]
        upred = user_srow['predict'].tolist()
        if len(upred) < 2:
            continue
        ulabel = user_srow['label'].tolist()
        ndcg.append(ndcg_score([ulabel], [upred], k = k))

    return np.mean(np.array(ndcg))


def eval_model(model, dataset, arg):
    loader = DataLoader(dataset, arg.batch, collate_fn = collate_HAW, shuffle = True, num_workers = 8)
    model.eval()
    preds, labels, uids = [], [], []
    with torch.no_grad():
        for bn, batch in enumerate(loader):
            uid, sid, sub_nodes, sub_edges, y, seqs, clk_time, tar_edges, tar_nodes = batch
            clk_time = clk_time.to(device)
            y = np.array(y)
            logits = torch.sigmoid(model(uid, sid, sub_nodes, sub_edges, seqs, clk_time, tar_edges, tar_nodes)) \
                .squeeze().clone().detach().cpu().numpy()
            preds.append(logits)
            labels.append(y)
            uids.append(uid.numpy())

    preds = np.concatenate(preds, 0, dtype = np.float64)
    labels = np.concatenate(labels, 0, dtype = np.float64)
    uids = np.concatenate(uids, 0)
    ndcg10 = cal_ndcg(preds, labels, uids, 10)
    auc = roc_auc_score(labels, preds)
    logloss = log_loss(labels, preds)
    return logloss, auc, ndcg10, preds


def train_test(model, tr_set, va_set, te_set, arg, device):
    opt = torch.optim.Adam(model.parameters(), lr = arg.lr)
    batch_num = tr_set.len // arg.batch
    loader = DataLoader(tr_set, arg.batch, collate_fn = collate_HAW, shuffle = True, num_workers = 16)
    best_auc, best_epoch, best_loss = 0., 0., 0.
    logloss_test, auc_test, ndcg10_test = 0., 0., 0.

    for epoch in range(arg.epoch):
        losses = []
        model.train()
        for bn, batch in enumerate(loader):
            uid, sid, sub_nodes, sub_edges, y, seqs, clk_time, tar_edges, tar_nodes = batch
            clk_time = clk_time.to(device)
            y = torch.Tensor(y).to(device)
            logits = model(uid, sid, sub_nodes, sub_edges, seqs, clk_time, tar_edges, tar_nodes).squeeze()
            loss = torch.nn.BCEWithLogitsLoss()(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if (bn + 1) % 100 == 0:
                logging.info(f'Batch: {bn + 1} / {batch_num}, loss: {loss.item()}')
            losses.append(loss.item())

        logloss, auc, ndcg10 = eval_model(model, va_set, arg)
        logging.info('')
        logging.info(f'Epoch: {epoch + 1} / {arg.epoch}, AUC: {auc}, loss: {logloss}, ndcg@10: {ndcg10}')
        if epoch - best_epoch == arg.patience:
            logging.info(f'Stop training after {arg.patience} epochs without valid improvement.')
            break

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            logloss_test, auc_test, ndcg10_test = eval_model(model, te_set, arg)
        logging.info(
            f'Best valid AUC: {best_auc} at epch {best_epoch}, test AUC: {auc_test}, loss: {logloss_test}, ndcg@10: {ndcg10_test}\n')

    logging.info(f'Training finished, best epoch {best_epoch}')
    logging.info(f'Valid AUC: {best_auc}, Test AUC: {auc_test}, loss: {logloss_test}, ndcg@10: {ndcg10_test}')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_torch(ARG.seed)
    ARG.save = ARG.save + ARG.data

    LOG_FORMAT = "%(asctime)s  %(message)s"
    DATE_FORMAT = "%m/%d %H:%M"
    if ARG.log is not None:
        logging.basicConfig(filename = ARG.log, level = logging.DEBUG, format = LOG_FORMAT, datefmt = DATE_FORMAT)
    else:
        logging.basicConfig(level = logging.DEBUG, format = LOG_FORMAT, datefmt = DATE_FORMAT)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_ds = Haw_data(mode = 'train', datatype = ARG.data)
    va_ds = Haw_data(mode = 'valid', datatype = ARG.data)
    te_ds = Haw_data(mode = 'test', datatype = ARG.data)
    user_count, item_count = tr_ds.n_user, tr_ds.n_item
    logging.info('Training data loaded')

    model = CTR_Model_HAW(user_count, item_count, ARG.embed // ARG.K, ARG.hid, ARG.beta, tr_ds.entire_graph,
                          ARG.K, ARG.nConvs, tr_ds.max_len, device, tr_ds.max_time, tr_ds.min_time,
                          ARG.tau, ARG.alpha).to(device)
    ckpt = torch.load(ARG.save)
    model.load_state_dict(ckpt['model_state_dict'])

    logging.info(f'Users: {model.n_user}\tItems: {model.n_item}\n')

    logloss_test, auc_test, ndcg10_test, preds = eval_model(model, te_ds, ARG)
    logging.info(f'Test AUC: {auc_test}, loss: {logloss_test}, ndcg@10: {ndcg10_test}')

