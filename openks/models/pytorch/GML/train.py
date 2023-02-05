import numpy as np
from torch.optim import lr_scheduler
from tqdm import tqdm
from model import *
# from GTN import *
from metrics import *
import pickle
import time

class Trainer(object):
    def __init__(self, args, data_loader):
        super(Trainer, self).__init__()
        self.topK = args.topk
        self.device = args.device
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.n_layer = args.n_layer
        self.cur_dim = args.cur_dim
        self.n_domain = args.n_domain
        self.log_step = args.log_step
        self.save = args.save
        self.patience = args.patience
        self.epochs = args.epochs
        # self.item_sample = args.item_sample
        if args.mode == 'train':
            self.train_data_loader = data_loader[0]
            self.valid_data_loader = data_loader[1]
            self.test_data_loader = data_loader[2]
        else:
            self.test_data_loader = data_loader
        self.rs_mutual = rs_mutual(args).to(self.device)
        self.mutual_optimizer = torch.optim.Adam(list(self.rs_mutual.parameters()), lr=args.lr)
        self.mutual_scheduler = lr_scheduler.StepLR(self.mutual_optimizer, step_size=args.decay_step, gamma=args.decay)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self):
        best_ndcg = 0.0
        best_epoch = -1
        # self.valid()
        # self.test()
        print('training')
        for epoch in range(self.epochs):
            print('Start epoch: ', epoch)
            self.train_one_epoch(epoch)
            valid_precision, valid_recall, valid_hr, valid_ndcg = self.valid()
            self.test()
            self.mutual_scheduler.step()
            if valid_ndcg[-1] > best_ndcg:
                best_epoch = epoch
                best_ndcg = valid_ndcg[-1]
                self.save_model()
                print('Model save for higher ndcg %f in %s' % (best_ndcg, self.save))
            if epoch - best_epoch >= self.patience:
                print('Stop training after %i epochs without improvement on validation.' % self.patience)
                break
        self.load_model()
        self.test()

    def train_one_epoch(self, epoch):
        self.rs_mutual.train()
        epoch_step_loss = []
        print("epoch ", epoch, ' training')
        for step, batch_data in enumerate(self.train_data_loader):
            users = [u.to(self.device).squeeze(dim=-1) for u in batch_data[0]]
            domains = batch_data[1].to(self.device)
            items = [i.to(self.device) for i in batch_data[2]]
            # items = batch_data[2]
            # users = users.to(self.device).squeeze(dim=-1)
            # items = items.to(self.device)
            label = batch_data[-1].to(self.device)
            user_common_neigh = [neigh.to(self.device) for neigh in batch_data[3][-1]]
            user_domain_neigh = [[neigh.to(self.device) for neigh in neighs] for neighs in batch_data[3][:-1]]
            item_common_neigh = [neigh.to(self.device) for neigh in batch_data[4][-1]]
            item_domain_neigh = [[neigh.to(self.device) for neigh in neighs] for neighs in batch_data[4][:-1]]

            prediction, prediction_common, prediction_domain, user_common_feature, _, _, _, user_domain_features = \
                self.rs_mutual(users, items, domains, user_common_neigh, item_common_neigh, user_domain_neigh, item_domain_neigh)

            mutual_loss = self.rs_mutual.get_loss(prediction, prediction_common, prediction_domain, label,
                                                  user_common_feature, user_domain_features)

            self.mutual_optimizer.zero_grad()
            mutual_loss.backward()
            # print(self.rs_mutual.user_domain_embedding_lists[0].weight)
            # time.sleep(10)
            self.mutual_optimizer.step()

            epoch_step_loss.append(mutual_loss.item())

            if (step % self.log_step == 0) and step > 0:
                print(f'Train epoch: {epoch}[{step}/{len(self.train_data_loader)} '
                      f'({100. * step / len(self.train_data_loader):.0f}%)]\t '
                      f'Lr:{self.get_lr(self.mutual_optimizer):.6f}, '
                      f'Loss: {mutual_loss.item():.6f}, '
                      f'AvgL: {np.mean(epoch_step_loss):.6f}')

    def eval(self, mode, visual_emb=False):
        datapath = '/home/wyf/project/DGML/meituan1/'
        with open(datapath + 'feeds_non.pkl', 'rb') as f:
            feeds_non = pickle.load(f)
        with open(datapath + 'poi_non.pkl', 'rb') as f:
            poi_non = pickle.load(f)
        print('overlap load!')

        self.rs_mutual.eval()
        eval_p_common = []
        eval_r_common = []
        eval_h_common = []
        eval_ndcg_common = []
        eval_len_common = []

        eval_p_domain = []
        eval_r_domain = []
        eval_h_domain = []
        eval_ndcg_domain = []
        eval_len_domain = []
        if mode == 'valid':
            eval_data_loader = self.valid_data_loader
            eval_p = []
            eval_r = []
            eval_h = []
            eval_ndcg = []
            eval_len = []
        else:
            eval_data_loader = self.test_data_loader
            eval_p = [[] for _ in range(self.n_domain)]
            eval_r = [[] for _ in range(self.n_domain)]
            eval_h = [[] for _ in range(self.n_domain)]
            eval_ndcg = [[] for _ in range(self.n_domain)]
            eval_len = [[] for _ in range(self.n_domain)]

            if visual_emb:
                users_common_emb = {}
                items_common_emb = {}
                users_domain_emb_lists = [{} for _ in range(self.n_domain)]
                items_domain_emb_lists = [{} for _ in range(self.n_domain)]
                users_items = [{} for _ in range(self.n_domain)]

        n_eval = len(eval_data_loader)
        ii, jj = 1, 1
        with torch.no_grad():
            for step, batch_data in tqdm(enumerate(eval_data_loader), total=n_eval):
                users = [u.to(self.device).squeeze(dim=-1) for u in batch_data[0]]
                domains = batch_data[1].to(self.device)
                items = [i.to(self.device) for i in batch_data[2]]
                batch_size = users[0].shape[0]
                # users = users.to(self.device).squeeze(dim=-1)
                # items = items.to(self.device)

                user_common_neigh = [neigh.to(self.device) for neigh in batch_data[3][-1]]
                user_domain_neigh = [[neigh.to(self.device) for neigh in neighs] for neighs in batch_data[3][:-1]]
                item_common_neigh = [neigh.to(self.device) for neigh in batch_data[4][-1]]
                item_domain_neigh = [[neigh.to(self.device) for neigh in neighs] for neighs in batch_data[4][:-1]]

                prediction, prediction_common, prediction_domain, user_common_feature, item_common_feature, \
                user_domain_feature, item_domain_feature, user_domain_features = \
                    self.rs_mutual(users, items, domains, user_common_neigh, item_common_neigh, user_domain_neigh,
                                   item_domain_neigh)

                if visual_emb:
                    batch_size = domains.shape[0]
                    for idx in range(batch_size):
                        u_c = users[-1][idx].item()
                        if u_c not in users_common_emb:
                            users_common_emb[u_c] = user_common_feature[idx].cpu().numpy()
                        d = domains[idx].item()
                        u_s = users[d][idx].item()
                        if u_s not in users_domain_emb_lists[d]:
                            users_domain_emb_lists[d][u_s] = user_domain_feature[idx].cpu().numpy()
                        i_c = items[-1][idx][-1].item()
                        if i_c not in items_common_emb:
                            items_common_emb[i_c] = item_common_feature[idx].cpu().numpy()
                        i_s = items[d][idx][-1].item()
                        if i_s not in items_domain_emb_lists[d]:
                            items_domain_emb_lists[d][i_s] = item_domain_feature[idx].cpu().numpy()
                        # if u_c not in users_items[-1]:
                        #     users_items[-1][u_c] = [i_c]
                        # else:
                        #     users_items[-1][u_c].append(i_c)
                        if u_c not in users_items[d]:
                            users_items[d][u_c] = [i_c]
                        else:
                            users_items[d][u_c].append(i_c)


                pred_items = torch.topk(torch.sigmoid(prediction), self.topK, sorted=True, dim=1)[1]
                pred_items_common = torch.topk(torch.sigmoid(prediction_common), self.topK, sorted=True, dim=1)[1]
                pred_items_domain = torch.topk(torch.sigmoid(prediction_domain), self.topK, sorted=True, dim=1)[1]
                for i in range(batch_size):
                    user = users[-1][i].item()
                    # if user not in feeds_non:
                    #     jj += 1
                    #     continue
                    ii += 1
                    domain = domains[i]
                    pred = pred_items[i].tolist()
                    pred_common = pred_items_common[i].tolist()
                    pred_domain = pred_items_domain[i].tolist()

                    p_at_k_common = getP(pred_common, [99])
                    r_at_k_common = getR(pred_common, [99])
                    h_at_k_common = getHitRatio(pred_common, [99])
                    ndcg_at_k_common = getNDCG(pred_common, [99])
                    eval_p_common.append(p_at_k_common)
                    eval_r_common.append(r_at_k_common)
                    eval_h_common.append(h_at_k_common)
                    eval_ndcg_common.append(ndcg_at_k_common)
                    eval_len_common.append(1)

                    p_at_k_domain = getP(pred_domain, [99])
                    r_at_k_domain = getR(pred_domain, [99])
                    h_at_k_domain = getHitRatio(pred_domain, [99])
                    ndcg_at_k_domain = getNDCG(pred_domain, [99])
                    eval_p_domain.append(p_at_k_domain)
                    eval_r_domain.append(r_at_k_domain)
                    eval_h_domain.append(h_at_k_domain)
                    eval_ndcg_domain.append(ndcg_at_k_domain)
                    eval_len_domain.append(1)

                    p_at_k = getP(pred, [99])
                    r_at_k = getR(pred, [99])
                    h_at_k = getHitRatio(pred, [99])
                    ndcg_at_k = getNDCG(pred, [99])
                    if mode == 'valid':
                        eval_p.append(p_at_k)
                        eval_r.append(r_at_k)
                        eval_h.append(h_at_k)
                        eval_ndcg.append(ndcg_at_k)
                        eval_len.append(1)
                    else:
                        eval_p[domain].append(p_at_k)
                        eval_r[domain].append(r_at_k)
                        eval_h[domain].append(h_at_k)
                        eval_ndcg[domain].append(ndcg_at_k)
                        eval_len[domain].append(1)

        if mode == 'valid':
            mean_p = np.mean(eval_p)
            mean_r = np.mean(eval_r)
            mean_h = np.mean(eval_h)
            mean_ndcg = np.mean(eval_ndcg)
        else:
            mean_p = [np.mean(d) for d in eval_p]
            mean_r = [np.mean(d) for d in eval_r]
            mean_h = [np.sum(eval_h[d]) / np.sum(eval_len[d]) for d in range(self.n_domain)]
            mean_ndcg = [np.mean(d) for d in eval_ndcg]

        mean_p_common = np.mean(eval_p_common)
        mean_r_common = np.mean(eval_r_common)
        mean_h_common = np.sum(eval_h_common) / np.sum(eval_len_common)
        mean_ndcg_common = np.mean(eval_ndcg_common)

        mean_p_domain = np.mean(eval_p_domain)
        mean_r_domain = np.mean(eval_r_domain)
        mean_h_domain = np.sum(eval_h_domain) / np.sum(eval_len_domain)
        mean_ndcg_domain = np.mean(eval_ndcg_domain)

        mean_p_all = [mean_p_common, mean_p_domain, mean_p]
        mean_r_all = [mean_r_common, mean_r_domain, mean_r]
        mean_h_all = [mean_h_common, mean_h_domain, mean_h]
        mean_ndcg_all = [mean_ndcg_common, mean_ndcg_domain, mean_ndcg]
        if visual_emb:
            with open('user_item_embedding.pkl', 'wb') as f:
                pickle.dump(users_common_emb, f)
                pickle.dump(items_common_emb, f)
                pickle.dump(users_domain_emb_lists, f)
                pickle.dump(items_domain_emb_lists, f)
                pickle.dump(users_items, f)
            print('user item embedding saved!')
        print(ii, jj)
        return mean_p_all, mean_r_all, mean_h_all, mean_ndcg_all

    def valid(self):
        print('Start Valid')
        mode = 'valid'
        mean_p, mean_r, mean_h, mean_ndcg = self.eval(mode)
        print(f'Valid:\tprecision_c@{self.topK}:{mean_p[0]:.6f}, recall_c@{self.topK}:{mean_r[0]:.6f}, '
              f'hr_c@{self.topK}:{mean_h[0]:.6f}, ndcg_c@{self.topK}:{mean_ndcg[0]:.6f}')
        print(f'Valid:\tprecision_d@{self.topK}:{mean_p[1]:.6f}, recall_d@{self.topK}:{mean_r[1]:.6f}, '
              f'hr_d@{self.topK}:{mean_h[1]:.6f}, ndcg_d@{self.topK}:{mean_ndcg[1]:.6f}')
        print(f'Valid:\tprecision@{self.topK}:{mean_p[2]:.6f}, recall@{self.topK}:{mean_r[2]:.6f}, '
              f'hr@{self.topK}:{mean_h[2]:.6f}, ndcg@{self.topK}:{mean_ndcg[2]:.6f}')
        return mean_p, mean_r, mean_h, mean_ndcg

    def test(self, visual_att=False, visual_emb=False):
        print('Start Test')
        if not visual_att:
            mode = 'test'
            mean_p, mean_r, mean_h, mean_ndcg = self.eval(mode, visual_emb)
            print(f'Test:\tprecision_c@{self.topK}:{mean_p[0]:.6f}, recall_c@{self.topK}:{mean_r[0]:.6f}, '
                  f'hr_c@{self.topK}:{mean_h[0]:.6f}, ndcg_c@{self.topK}:{mean_ndcg[0]:.6f}')
            print(f'Test:\tprecision_d@{self.topK}:{mean_p[1]:.6f}, recall_d@{self.topK}:{mean_r[1]:.6f}, '
                  f'hr_d@{self.topK}:{mean_h[1]:.6f}, ndcg_d@{self.topK}:{mean_ndcg[1]:.6f}')
            for d in range(self.n_domain):
                print(f'Test domain_{d}:\tprecision@{self.topK}:{mean_p[2][d]:.6f}, recall@{self.topK}:{mean_r[2][d]:.6f}, '
                      f'hr@{self.topK}:{mean_h[2][d]:.6f}, ndcg@{self.topK}:{mean_ndcg[2][d]:.6f}')
        else:
            self.rs_mutual.eval()
            eval_data_loader = self.test_data_loader
            n_eval = len(eval_data_loader)
            # r_choice = np.random.randint(n_eval)
            # r_choice = 0
            user_common_h_atts = 0
            item_common_h_atts = 0
            user_domain_h_atts = 0
            item_domain_h_atts = 0
            n_att_s = 0.0
            n_att_c = 0.0
            with torch.no_grad():
                for step, batch_data in tqdm(enumerate(eval_data_loader), total=n_eval):
                    # if step == r_choice:
                    d = batch_data[1][0].item()
                    # print('domain:', str(d))
                    # domains = batch_data[1].to(self.device)
                    user_c = [batch_data[0][-1].to(self.device).squeeze(dim=-1)]
                    user_s = [batch_data[0][d].to(self.device).squeeze(dim=-1)]

                    item_c = [batch_data[2][-1][:, -1].to(self.device)]
                    item_s = [batch_data[2][d][:, -1].to(self.device)]

                    # print(user_c)
                    # print(item_c)

                    # print(batch_data[2][-1])
                    # print(item_c)
                    # print(item_s)
                    # print(user_c)
                    # time.sleep(100)
                    # for l in range(3):
                    #     print(batch_data[3][-1][l].shape)
                    #     print(batch_data[4][-1][l].shape)
                    # time.sleep(100)

                    user_common_neigh = user_c + [neigh.to(self.device) for neigh in batch_data[3][-1]]
                    user_domain_neigh = user_s + [neigh.to(self.device) for neigh in batch_data[3][d]]
                    item_common_neigh = item_c + [neigh[:, -1, :].to(self.device) for neigh in batch_data[4][-1]]
                    item_domain_neigh = item_s + [neigh[:, -1, :].to(self.device) for neigh in batch_data[4][d]]

                    users_common_neigh_es = []
                    items_common_neigh_es = []
                    users_domain_neigh_es = []
                    items_domain_neigh_es = []
                    for l in range(self.n_layer+1):
                        if l % 2 == 0:
                            users_common_neigh_es.append(self.rs_mutual.user_common_embedding(user_common_neigh[l]))
                            items_common_neigh_es.append(self.rs_mutual.item_common_embedding(item_common_neigh[l]))
                            users_domain_neigh_es.append(self.rs_mutual.user_domain_embedding_lists[d](user_domain_neigh[l]))
                            items_domain_neigh_es.append(self.rs_mutual.item_domain_embedding_lists[d](item_domain_neigh[l]))
                        if l % 2 == 1:
                            users_common_neigh_es.append(self.rs_mutual.item_common_embedding(user_common_neigh[l]))
                            items_common_neigh_es.append(self.rs_mutual.user_common_embedding(item_common_neigh[l]))
                            users_domain_neigh_es.append(self.rs_mutual.item_domain_embedding_lists[d](user_domain_neigh[l]))
                            items_domain_neigh_es.append(self.rs_mutual.user_domain_embedding_lists[d](item_domain_neigh[l]))

                    # for l in range(self.n_layer + 1):
                    #     # print(users_common_neigh_es[l].shape)
                    #     # print(items_common_neigh_es[l].shape)
                    #     print(users_domain_neigh_es[l].shape)
                    #     # print(items_domain_neigh_es[l].shape)

                    user_common_graph_layers = self.rs_mutual.user_common_graph.GTNLayers
                    user_common_pos_encoder = self.rs_mutual.user_common_graph.pos_encoder
                    user_common_cross_layers = self.rs_mutual.user_common_graph.CrossLayers

                    item_common_graph_layers = self.rs_mutual.item_common_graph.GTNLayers
                    item_common_pos_encoder = self.rs_mutual.item_common_graph.pos_encoder
                    item_common_cross_layers = self.rs_mutual.item_common_graph.CrossLayers

                    user_domain_graph_layers = self.rs_mutual.user_domain_graph_lists[d].GTNLayers
                    user_domain_pos_encoder = self.rs_mutual.user_domain_graph_lists[d].pos_encoder
                    user_domain_cross_layers = self.rs_mutual.user_domain_graph_lists[d].CrossLayers

                    item_domain_graph_layers = self.rs_mutual.item_domain_graph_lists[d].GTNLayers
                    item_domain_pos_encoder = self.rs_mutual.item_domain_graph_lists[d].pos_encoder
                    item_domain_cross_layers = self.rs_mutual.item_domain_graph_lists[d].CrossLayers

                    user_common_h_list = [users_common_neigh_es[0]]
                    item_common_h_list = [items_common_neigh_es[0]]
                    user_domain_h_list = [users_domain_neigh_es[0]]
                    item_domain_h_list = [items_domain_neigh_es[0]]
                    for l in range(self.n_layer):
                        user_common_h_list.append(
                            user_common_graph_layers[l](users_common_neigh_es[0], users_common_neigh_es[l+1]))
                        item_common_h_list.append(
                            item_common_graph_layers[l](items_common_neigh_es[0], items_common_neigh_es[l+1]))
                        user_domain_h_list.append(
                            user_domain_graph_layers[l](users_domain_neigh_es[0], users_domain_neigh_es[l+1]))
                        item_domain_h_list.append(
                            item_domain_graph_layers[l](items_domain_neigh_es[0], items_domain_neigh_es[l+1]))
                    user_common_h_list = torch.stack(user_common_h_list, dim=0)
                    user_common_h_list = user_common_h_list.view(self.n_layer+1, -1, self.cur_dim) * math.sqrt(self.cur_dim)
                    user_common_h_feature = user_common_pos_encoder(user_common_h_list)

                    item_common_h_list = torch.stack(item_common_h_list, dim=0)
                    item_common_h_list = item_common_h_list.view(self.n_layer+1, -1, self.cur_dim) * math.sqrt(self.cur_dim)
                    item_common_h_feature = item_common_pos_encoder(item_common_h_list)

                    user_domain_h_list = torch.stack(user_domain_h_list, dim=0)
                    user_domain_h_list = user_domain_h_list.view(self.n_layer+1, -1, self.cur_dim) * math.sqrt(self.cur_dim)
                    user_domain_h_feature = user_domain_pos_encoder(user_domain_h_list)

                    item_domain_h_list = torch.stack(item_domain_h_list, dim=0)
                    item_domain_h_list = item_domain_h_list.view(self.n_layer+1, -1, self.cur_dim) * math.sqrt(self.cur_dim)
                    item_domain_h_feature = item_domain_pos_encoder(item_domain_h_list)

                    # print(user_common_h_feature)
                    # print(user_common_h_feature.shape)
                    # time.sleep(100)
                    user_common_h_att = user_common_cross_layers.layers[-1].self_attn(
                        user_common_h_feature, user_common_h_feature, user_common_h_feature, need_weights=True)[1]

                    user_domain_h_att = user_domain_cross_layers.layers[-1].self_attn(
                        user_domain_h_feature, user_domain_h_feature, user_domain_h_feature, need_weights=True)[1]

                    item_common_h_att = item_common_cross_layers.layers[-1].self_attn(
                        item_common_h_feature, item_common_h_feature, item_common_h_feature, need_weights=True)[1]

                    item_domain_h_att = item_domain_cross_layers.layers[-1].self_attn(
                        item_domain_h_feature, item_domain_h_feature, item_domain_h_feature, need_weights=True)[1]

                    # torch.set_printoptions(precision=3, sci_mode=False)

                    if d == 1:
                        user_common_h_atts += user_common_h_att
                        item_common_h_atts += item_common_h_att
                        user_domain_h_atts += user_domain_h_att
                        item_domain_h_atts += item_domain_h_att

                        n_att_s += 1
                        n_att_c += 1

                    # print(user_common_h_att)
                    # print(item_common_h_att)
                    # print(user_domain_h_att)
                    # print(item_domain_h_att)
                    # break
                        # time.sleep(100)
            torch.set_printoptions(precision=3, sci_mode=False)
            print(user_common_h_atts/n_att_c)
            print(item_common_h_atts/n_att_c)
            print(user_domain_h_atts/n_att_s)
            print(item_domain_h_atts/n_att_s)







                        # prediction, prediction_common, prediction_domain, user_common_feature, user_domain_features = \
                        #     self.rs_mutual(users, items, domains, user_common_neigh, item_common_neigh, user_domain_neigh,
                        #                    item_domain_neigh)



    def save_model(self):
        with open(self.save + '_rs.mutual.pth', 'wb') as f:
            torch.save(self.rs_mutual.state_dict(), f)

    def load_model(self):
        self.rs_mutual.load_state_dict(torch.load(self.save + '_rs.mutual.pth'))
        self.rs_mutual.to(self.device)