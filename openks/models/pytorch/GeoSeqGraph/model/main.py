import os, torch, random, argparse, logging
import pickle
from dataloader import MultiSessionsGraph
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, ndcg_score, log_loss
import numpy as np
import pandas as pd
from HGS_POI import EmbeddingLayer, SeqGraph, GeoGraph
from compress import CompReSSMomentum
import torch.nn as nn

ARG = argparse.ArgumentParser()
ARG.add_argument('--epoch', type=int, default=140,
                 help='Epoch num.')
ARG.add_argument('--seed', type=int, default=98765,
                 help='Random seed.')
ARG.add_argument('--batch', type=int, default=1024,
                 help='Training batch size.')
ARG.add_argument('--data', type=str, default='nyc',
                 help='Training dataset.')
ARG.add_argument('--gpu', type=int, default=None,
                 help='Denote training device.')
ARG.add_argument('--patience', type=int, default=10,
                 help='Early stopping patience.')
ARG.add_argument('--embed', type=int, default=64,
                 help='Embedding dimension.')
ARG.add_argument('--gcn_num', type=int, default=2,
                 help='Num of GCN.')
ARG.add_argument('--max_step', type=int, default=2,
                 help='Steps of random walk.')
ARG.add_argument('--hid_graph_num', type=int, default=16,
                 help='Num of hidden graphs.')
ARG.add_argument('--hid_graph_size', type=int, default=10,
                 help='Size of hidden graphs')
ARG.add_argument('--weight_decay', type=float, default=5e-4,
                 help='Weight decay rate')
ARG.add_argument('--lr', type=float, default=1e-3,
                 help='Learning rate.')
ARG.add_argument('--log', type=str, default=None,
                 help='Log file path.')
ARG.add_argument('--con_weight', type=float, default=0.01,
                 help='Weight of consistency loss')
ARG.add_argument('--compress_memory_size', type=int, default=12800,
                 help='Memory bank size')
ARG.add_argument('--compress_t', type=float, default=0.01,
                 help='Softmax temperature')

ARG = ARG.parse_args()


def cal_ndcg(predicts, labels, user_ids, k):
    d = {'user': np.squeeze(user_ids), 'predict':np.squeeze(predicts), 'label':np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()

    ndcg = []
    for user_id in user_unique:
        user_srow = df.loc[df['user'] == user_id]
        upred = user_srow['predict'].tolist()
        if len(upred) < 2:
            continue
        ulabel = user_srow['label'].tolist()
        ndcg.append(ndcg_score([ulabel], [upred], k=k)) 

    return np.mean(np.array(ndcg))


def eval_model(Seq_encoder, Geo_encoder, Poi_embeds, dataset, arg, device):
    loader = DataLoader(dataset, arg.batch, shuffle=True)
    seq_preds, geo_preds, labels, pois = [], [], [], []
    Seq_encoder.eval()
    Geo_encoder.eval()

    with torch.no_grad():
        for bn, batch in enumerate(loader):
            _, seq_logit = Seq_encoder(batch.to(device), Poi_embeds)
            _, geo_logit = Geo_encoder(batch.to(device), Poi_embeds)

            seq_logits = torch.sigmoid(seq_logit)\
                .squeeze().clone().detach().cpu().numpy()
            geo_logits = torch.sigmoid(geo_logit)\
                .squeeze().clone().detach().cpu().numpy()

            seq_preds.append(seq_logits)
            geo_preds.append(geo_logits)
            labels.append(batch.y.squeeze().cpu().numpy())
    
    seq_preds = np.concatenate(seq_preds, 0, dtype=np.float64)
    geo_preds = np.concatenate(geo_preds, 0, dtype=np.float64)
    labels = np.concatenate(labels, 0, dtype=np.float64)
    seq_auc = roc_auc_score(labels, seq_preds)
    seq_logloss = log_loss(labels, seq_preds)
    geo_auc = roc_auc_score(labels, geo_preds)
    geo_logloss = log_loss(labels, geo_preds)
    return seq_auc, seq_logloss, geo_auc, geo_logloss


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_test(tr_set, va_set, te_set, arg, dist_edges, dist_vec, device):
    Seq_encoder = SeqGraph(n_user, n_poi, arg.max_step, arg.embed, arg.hid_graph_num, arg.hid_graph_size, device).to(device)
    Geo_encoder = GeoGraph(n_user, n_poi, arg.gcn_num, arg.embed, dist_edges, dist_vec, device).to(device)
    Poi_embeds = EmbeddingLayer(n_poi, arg.embed).to(device)
    Sim_criterion = CompReSSMomentum(arg.embed, arg.compress_memory_size, arg.compress_t, device).to(device)

    opt = torch.optim.Adam([
                {'params': Seq_encoder.parameters()},
                {'params': Geo_encoder.parameters()},
                {'params': Poi_embeds.parameters()}], lr=arg.lr)#, weight_decay=arg.weight_decay)

    batch_num = len(tr_set) // arg.batch
    train_loader = DataLoader(tr_set, arg.batch, shuffle=True)
    bank_loader = DataLoader(tr_set, arg.batch, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_epoch = 0., 0.
    test_auc, test_loss = 0., 0.

    for epoch in range(arg.epoch):
        Seq_encoder.train()
        Geo_encoder.train()
        for bn, (trn_batch, bnk_batch) in enumerate(zip(train_loader, bank_loader)):
            trn_batch, bnk_batch = trn_batch.to(device), bnk_batch.to(device)
            label = trn_batch.y.float()

            _, seq_pred = Seq_encoder(trn_batch, Poi_embeds)
            seq_bnk_enc, _ = Seq_encoder(bnk_batch, Poi_embeds)
            seq_sup_loss = criterion(seq_pred.squeeze(), label)

            _, geo_pred = Geo_encoder(trn_batch, Poi_embeds)
            geo_bnk_enc, _ = Geo_encoder(bnk_batch, Poi_embeds)
            geo_sup_loss = criterion(geo_pred.squeeze(), label)
            
            unsup_loss = Sim_criterion(seq_bnk_enc, geo_bnk_enc)
            loss = seq_sup_loss + geo_sup_loss + arg.con_weight * unsup_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (bn + 1) % 200 == 0:
                logging.info(f'''Batch: {bn + 1} / {batch_num}, loss: {loss.item()} = 
                                Seq:{ seq_sup_loss.item()} + Geo:{ geo_sup_loss.item()} + Con: {unsup_loss.mean(-1).item()}''')

        seq_auc, seq_logloss, geo_auc, geo_logloss= eval_model(Seq_encoder, Geo_encoder, Poi_embeds, va_set, arg, device)
        logging.info('')
        logging.info(f'''Epoch: {epoch + 1} / {arg.epoch}, 
            Seq_AUC: {seq_auc}, Seq_loss: {seq_logloss}, Geo_AUC: {geo_auc}, Geo_loss: {geo_logloss}''')

        if epoch - best_epoch == arg.patience:
            logging.info(f'Stop training after {arg.patience} epochs without valid improvement.')
            break
        if(seq_auc > best_auc or geo_auc > best_auc):
            best_auc = max(seq_auc, geo_auc)
            best_epoch = epoch
            bst_seq_auc, bst_seq_logloss, bst_geo_auc, bst_geo_logloss = eval_model(Seq_encoder, Geo_encoder, Poi_embeds, te_set, arg, device)
            test_auc = max(seq_auc, geo_auc)
            test_loss = min(seq_logloss, geo_logloss)

        logging.info(f'''Best valid AUC: {best_auc} at epch {best_epoch}, 
            Seq_AUC: {bst_seq_auc}, Seq_loss: {bst_seq_logloss}, Geo_AUC: {bst_geo_auc}, Geo_loss: {bst_geo_logloss}\n''')

    logging.info(f'Training finished, best epoch {best_epoch}')
    logging.info(f'Valid AUC: {best_auc}, Test AUC: {test_auc}, Test logloss: {test_loss}')


if __name__ == '__main__':
    seed_torch(ARG.seed)

    LOG_FORMAT = "%(asctime)s  %(message)s"
    DATE_FORMAT = "%m/%d %H:%M"
    if ARG.log is not None:
        logging.basicConfig(filename=ARG.log, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    with open(f'../processed/{ARG.data}/raw/val.pkl', 'rb') as f:
        tmp = pickle.load(f)
        n_user, n_poi = pickle.load(f)
        del tmp

    train_set = MultiSessionsGraph(f'../processed/{ARG.data}', phrase='train')
    val_set = MultiSessionsGraph(f'../processed/{ARG.data}', phrase='test')
    test_set = MultiSessionsGraph(f'../processed/{ARG.data}', phrase='val')

    with open(f'../processed/{ARG.data}/dist_graph.pkl', 'rb') as f:
        dist_edges = torch.LongTensor(pickle.load(f))
        dist_nei = pickle.load(f)
    dist_vec = np.load(f'../processed/{ARG.data}/dist_on_graph.npy')

    logging.info(f'Data loaded.')
    logging.info(f'user: {n_user}\tpoi: {n_poi}')
    device = torch.device('cpu') if ARG.gpu is None else torch.device(f'cuda:{ARG.gpu}')
    train_test(train_set, test_set, val_set, ARG, dist_edges, dist_vec, device)


