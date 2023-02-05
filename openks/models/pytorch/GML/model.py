import torch
import torch.nn as nn
from GTN2 import GTN

class rs_mutual(nn.Module):
    def __init__(self, args):
        super(rs_mutual, self).__init__()
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.n_domain = args.n_domain
        self.n_domain_users = args.n_domain_users
        self.n_domain_items = args.n_domain_items
        self.item_domain_index_tables = args.item_domain_index_tables
        self.user_domain_index_tables = args.item_domain_index_tables

        self.cur_dim = args.cur_dim
        self.n_layer = args.n_layer

        self.temperature = args.temperature
        self.labelloss_weight = args.labelloss_weight
        self.hintloss_weight = args.hintloss_weight
        self.orthloss_weight = args.orthloss_weight
        self.device = args.device

        self.user_common_embedding = nn.Embedding(self.n_users, self.cur_dim, padding_idx=0).to(self.device)
        torch.nn.init.xavier_uniform_(self.user_common_embedding.weight)
        self.user_common_embedding.weight.data[0] = 0
        self.item_common_embedding = nn.Embedding(self.n_items, self.cur_dim, padding_idx=0).to(self.device)
        torch.nn.init.xavier_uniform_(self.item_common_embedding.weight)
        self.item_common_embedding.weight.data[0] = 0

        self.user_domain_embedding_lists = nn.ModuleList().to(self.device)
        self.item_domain_embedding_lists = nn.ModuleList().to(self.device)
        for d in range(self.n_domain):
            user_domain_embedding = nn.Embedding(self.n_domain_users[d], self.cur_dim, padding_idx=0).to(self.device)
            torch.nn.init.xavier_uniform_(user_domain_embedding.weight)
            user_domain_embedding.weight.data[0] = 0
            item_domain_embedding = nn.Embedding(self.n_domain_items[d], self.cur_dim, padding_idx=0).to(self.device)
            torch.nn.init.xavier_uniform_(item_domain_embedding.weight)
            item_domain_embedding.weight.data[0] = 0

            self.user_domain_embedding_lists.append(user_domain_embedding)
            self.item_domain_embedding_lists.append(item_domain_embedding)

        self.user_common_graph = GTN(self.cur_dim, args.n_neigh, args.n_head_layer, args.n_head_cross, args.n_cross,
                                     args.dropout, args.device).to(args.device)
        self.item_common_graph = GTN(self.cur_dim, args.n_neigh, args.n_head_layer, args.n_head_cross, args.n_cross,
                                     args.dropout, args.device).to(args.device)

        self.user_domain_graph_lists = nn.ModuleList().to(self.device)
        self.item_domain_graph_lists = nn.ModuleList().to(self.device)
        for d in range(self.n_domain):
            self.user_domain_graph_lists.append(GTN(self.cur_dim, args.n_neigh, args.n_head_layer, args.n_head_cross,
                                                    args.n_cross, args.dropout, args.device).to(args.device))
            self.item_domain_graph_lists.append(GTN(self.cur_dim, args.n_neigh, args.n_head_layer, args.n_head_cross,
                                                    args.n_cross, args.dropout, args.device).to(args.device))
        # self.item_domain_graph = GTN(self.cur_dim, args.n_neigh, args.n_head_layer, args.n_head_cross, args.n_cross,
        #                              args.dropout, args.device).to(args.device)

        # self.emb_proj_lists = nn.ModuleList()
        # for _ in range(self.n_domain + 1):
        #     self.emb_proj_lists.append(nn.Linear(self.cur_dim, self.cur_dim))

        self.gate = nn.Sequential(
            nn.Linear(self.cur_dim * (self.n_layer + 1) * 2, self.cur_dim * (self.n_layer + 1)),
            nn.ReLU(),
            nn.Linear(self.cur_dim * (self.n_layer + 1), 1)).to(self.device)

        self.W_common = nn.Sequential(
            nn.Linear(self.cur_dim * (self.n_layer + 1) * 2, self.cur_dim * (self.n_layer + 1)),
            nn.ReLU(),
            nn.Linear(self.cur_dim * (self.n_layer + 1), 1)).to(self.device)

        self.W_domain_lists = nn.ModuleList().to(self.device)
        for d in range(self.n_domain):
            self.W_domain_lists.append(nn.Sequential(
                nn.Linear(self.cur_dim * (self.n_layer + 1) * 2, self.cur_dim * (self.n_layer + 1)),
                nn.ReLU(),
                nn.Linear(self.cur_dim * (self.n_layer + 1), 1)).to(self.device))

        self.loss_fn = nn.BCELoss(reduction='none').to(self.device)
        self.loss_cos = nn.CosineEmbeddingLoss(reduction='none').to(self.device)

    def forward(self, users, items, domains, users_common_neigh, items_common_neigh, users_domain_neigh, items_domain_neigh):
        batch_size = users[0].shape[0]
        users_common_neigh_es = []
        items_common_neigh_es = []
        users_common_neigh = [users[-1]] + users_common_neigh
        items_common_neigh = [items[-1]] + items_common_neigh

        users_domain_neigh_es = [[] for _ in range(self.n_domain)]
        items_domain_neigh_es = [[] for _ in range(self.n_domain)]

        for l in range(self.n_layer+1):
            if l % 2 == 0:
                users_common_neigh_es.append(self.user_common_embedding(users_common_neigh[l]))
                items_common_neigh_es.append(self.item_common_embedding(items_common_neigh[l]))
                # item_domain_l_es = []
                # for d in range(self.n_domain):
                #     item_domain_l_es.append(self.item_domain_embedding_lists[d](items_domain_neigh_[l]))
                # items_domain_neigh_es.append(torch.stack(item_domain_l_es, dim=1)[torch.arange(batch_size), domains, :])
            if l % 2 == 1:
                users_common_neigh_es.append(self.item_common_embedding(users_common_neigh[l]))
                items_common_neigh_es.append(self.user_common_embedding(items_common_neigh[l]))
                # item_domain_l_es = []
                # for d in range(self.n_domain):
                #     items_domain_neigh_ = [items[d]] + items_domain_neigh[d]
                #     item_domain_l_es.append(self.user_domain_embedding_lists[d](items_domain_neigh_[l]))
                # items_domain_neigh_es.append(torch.stack(item_domain_l_es, dim=1)[torch.arange(batch_size), domains, :])

        for d in range(self.n_domain):
            users_domain_neigh_ = [users[d]] + users_domain_neigh[d]
            items_domain_neigh_ = [items[d]] + items_domain_neigh[d]
            for l in range(self.n_layer + 1):
                if l % 2 == 0:
                    users_domain_neigh_es[d].append(self.user_domain_embedding_lists[d](users_domain_neigh_[l]))
                    items_domain_neigh_es[d].append(self.item_domain_embedding_lists[d](items_domain_neigh_[l]))
                if l % 2 == 1:
                    users_domain_neigh_es[d].append(self.item_domain_embedding_lists[d](users_domain_neigh_[l]))
                    items_domain_neigh_es[d].append(self.user_domain_embedding_lists[d](items_domain_neigh_[l]))

        user_common_feature = self.user_common_graph(users_common_neigh_es[0], users_common_neigh_es[1:]).view(batch_size, -1)
        item_common_feature = self.item_common_graph(items_common_neigh_es[0], items_common_neigh_es[1:])
        item_common_feature = item_common_feature.view(batch_size, items[0].shape[1], -1)
        user_common_feature_ = torch.stack([user_common_feature] * items[0].shape[1], dim=1)

        user_domain_features = []
        item_domain_features = []
        for d in range(self.n_domain):
            user_feature = self.user_domain_graph_lists[d](users_domain_neigh_es[d][0], users_domain_neigh_es[d][1:])
            item_feature = self.item_domain_graph_lists[d](items_domain_neigh_es[d][0], items_domain_neigh_es[d][1:])
            user_domain_features.append(user_feature.view(batch_size, -1))
            item_domain_features.append(item_feature.view(batch_size, items[0].shape[1], -1))
        user_domain_features = torch.stack(user_domain_features, dim=1)
        item_domain_features = torch.stack(item_domain_features, dim=1)
        user_domain_feature = user_domain_features[torch.arange(batch_size), domains, :]
        item_domain_feature = item_domain_features[torch.arange(batch_size), domains, :]

        user_domain_feature_ = torch.stack([user_domain_feature] * items[0].shape[1], dim=1)

        # user_feature = self.proj(torch.cat([user_common_feature, user_domain_feature], dim=-1))
        # user_feature_ = torch.stack([user_feature] * items.shape[1], dim=1)
        user_c = self.gate(torch.cat([user_common_feature, user_domain_feature], dim=-1))
        user_c = torch.sigmoid(user_c)

        prediction_common = self.W_common(torch.cat([user_common_feature_, item_common_feature], dim=-1)).squeeze(-1)
        prediction_domains = []
        for d in range(self.n_domain):
            prediction_domains.append(self.W_domain_lists[d](torch.cat([user_domain_feature_, item_domain_feature],
                                                                       dim=-1)).squeeze(-1))
        prediction_domains = torch.stack(prediction_domains, dim=1)
        prediction_domain = prediction_domains[torch.arange(batch_size), domains]
        # prediction_common = (user_common_feature_ * item_common_feature).sum(-1)
        # prediction_domain = (user_domain_feature_ * item_domain_feature).sum(-1)

        # prediction_common_ = torch.sigmoid(prediction_common)
        # prediction_domain_ = torch.sigmoid(prediction_domain)

        prediction = user_c*prediction_common + (1-user_c)*prediction_domain

        return prediction, prediction_common, prediction_domain, user_common_feature, item_common_feature[:, -1, :],\
               user_domain_feature, item_domain_feature[:, -1, :], user_domain_features

    def get_loss(self, prediction, prediction_common, prediction_domain, label,
                 user_common_feature, user_domain_features):
        batch_size = label.shape[0]
        GTloss = self.loss_fn(torch.sigmoid(prediction), label).sum(-1)
        common_GTloss = self.loss_fn(torch.sigmoid(prediction_common), label).sum(-1)
        domain_GTloss = self.loss_fn(torch.sigmoid(prediction_domain), label).sum(-1)

        hintLoss = 0
        for d in range(self.n_domain):
            domain_feature = user_domain_features[:, d, :]
            hintLoss += self.loss_cos(user_common_feature, torch.autograd.Variable(domain_feature),
                                      torch.ones(batch_size).to(self.device))

        out = torch.matmul(user_domain_features, user_domain_features.transpose(1, 2))
        deno = torch.max(out, dim=-1)[0]
        deno.masked_fill_(deno == 0, 1)
        out = out / deno.unsqueeze(-2)
        i_m = torch.eye(user_domain_features.shape[-2]).unsqueeze(0).to(self.device)
        orthLoss = torch.abs(out - i_m).view(batch_size, -1).sum(-1)

        labelLoss1 = self.loss_fn(torch.sigmoid(prediction_common/self.temperature),
                                 torch.autograd.Variable(torch.sigmoid(prediction/self.temperature))).sum(-1)

        labelLoss2 = self.loss_fn(torch.sigmoid(prediction_domain / self.temperature),
                                  torch.autograd.Variable(torch.sigmoid(prediction/self.temperature))).sum(-1)

        loss = torch.mean(GTloss + common_GTloss + domain_GTloss +
                                 self.hintloss_weight * hintLoss + self.orthloss_weight * orthLoss +
                                 pow(self.temperature, 2) * self.labelloss_weight * (labelLoss1+labelLoss2))
        return loss

# class rs_inter(nn.Module):
#     def __init__(self, args):
#         super(rs_inter, self).__init__()
#         self.n_users = args.n_users
#         self.n_items = args.n_items
#         self.cur_dim = args.cur_dim
#         self.n_layer = args.n_layer
#         self.n_domain = args.n_domain
#         self.orth_weight = args.orth_weight
#         # self.kl_weight = args.kl_weight
#         self.temperature = args.temperature
#         self.labelloss_weight = args.labelloss_weight
#         # self.hintloss_weight = args.hintloss_weight
#         self.device = args.device
#
#         self.user_embedding = nn.Embedding(self.n_users, self.cur_dim, padding_idx=0).to(self.device)
#         torch.nn.init.xavier_uniform_(self.user_embedding.weight)
#         self.user_embedding.weight.data[0] = 0
#         self.item_embedding = nn.Embedding(self.n_items, self.cur_dim, padding_idx=0).to(self.device)
#         torch.nn.init.xavier_uniform_(self.item_embedding.weight)
#         self.item_embedding.weight.data[0] = 0
#         self.domain_graph_lists = nn.ModuleList().to(self.device)
#         for d in range(self.n_domain):
#             self.domain_graph_lists.append(GTN(self.cur_dim, args.n_neigh, args.n_head_layer, args.n_head_cross,
#                                                     args.n_cross, args.dropout, args.device).to(args.device))
#
#         self.emb_proj_lists = nn.ModuleList()
#         for _ in range(self.n_domain + 1):
#             self.emb_proj_lists.append(nn.Linear(self.cur_dim, self.cur_dim))
#
#         # self.user_embedding = nn.Embedding(self.n_users, self.cur_dim, padding_idx=0)
#         # self.item_embedding = nn.Embedding(self.n_items, self.cur_dim, padding_idx=0)
#         #
#         # torch.nn.init.xavier_uniform_(self.user_embedding.weight)
#         # self.user_embedding.weight.data[0] = 0
#         # torch.nn.init.xavier_uniform_(self.item_embedding.weight)
#         # self.item_embedding.weight.data[0] = 0
#         #
#         # self.user_domain_graph_lists = nn.ModuleList([graph_net(args, self.user_embedding, self.item_embedding,
#         #                                                         tgt_type='user', gtype='domain')
#         #                                               for _ in range(self.n_domain)]).to(self.device)
#         # self.item_graph = graph_net(args, self.user_embedding, self.item_embedding,
#         #                             tgt_type='item', gtype='domain').to(self.device)
#         # self.W = nn.Linear(self.cur_dim * (self.n_layer + 1) * 3, 1)
#         # self.W = nn.Sequential(
#         #     nn.Linear(self.cur_dim * (self.n_layer + 1) * 3, self.cur_dim * (self.n_layer + 1)),
#         #     nn.ReLU(),
#         #     nn.Linear(self.cur_dim * (self.n_layer + 1), 1))
#
#         # self.affine_output = nn.Linear(self.cur_dim * (self.n_layer + 1) * 2, 1)
#
#         self.proj = nn.Sequential(
#             nn.Linear(self.cur_dim * (self.n_layer + 1) * 2, self.cur_dim * (self.n_layer + 1)),
#             nn.ReLU(),
#             nn.Linear(self.cur_dim * (self.n_layer + 1), self.cur_dim * (self.n_layer + 1)))
#
#         self.W = nn.Sequential(
#             nn.Linear(self.cur_dim * (self.n_layer + 1) * 2, self.cur_dim * (self.n_layer + 1)),
#             nn.ReLU(),
#             nn.Linear(self.cur_dim * (self.n_layer + 1), 1))
#
#         # self.critic = Discriminator(self.n_layer, self.cur_dim, self.n_domain)
#
#         self.loss_fn = nn.BCELoss(reduction='none').to(self.device)
#         # self.loss_kl = nn.KLDivLoss(reduction='none').to(self.device)
#         # self.loss_mse = nn.MSELoss(reduction='none').to(self.device)
#         # self.loss_cos = nn.CosineEmbeddingLoss(reduction='none').to(self.device)
#         # self.loss_criterion = nn.CrossEntropyLoss(reduction='none')
#
#     def forward(self, users, items, domains, users_domain_neigh, items_neigh, user_common_feature):
#         batch_size = users.shape[0]
#         users_domain_neigh_es = [[] for _ in range(self.n_domain)]
#         items_domain_neigh_es = []
#         items_neigh = [items] + items_neigh
#         for l in range(self.n_layer + 1):
#             if l % 2 == 0:
#                 items_domain_neigh_es.append(self.item_embedding(items_neigh[l]))
#             if l % 2 == 1:
#                 item_domain_l_es = []
#                 for d in range(self.n_domain):
#                     item_domain_l_es.append(self.emb_proj_lists[d](self.user_embedding(items_neigh[l])))
#                 items_domain_neigh_es.append(torch.stack(item_domain_l_es, dim=1)[torch.arange(batch_size), domains, :])
#
#         for d in range(self.n_domain):
#             users_domain_neigh_ = [users] + users_domain_neigh[d]
#             for l in range(self.n_layer + 1):
#                 if l % 2 == 0:
#                     users_domain_neigh_es[d].append(self.emb_proj_lists[d](self.user_embedding(users_domain_neigh_[l])))
#                 if l % 2 == 1:
#                     users_domain_neigh_es[d].append(self.item_embedding(users_domain_neigh_[l]))
#
#         user_domain_features = []
#         item_domain_features = []
#         for d in range(self.n_domain):
#             user_feature = self.domain_graph_lists[d](users_domain_neigh_es[d][0], users_domain_neigh_es[d][1:])
#             item_feature = self.domain_graph_lists[d](items_domain_neigh_es[0], items_domain_neigh_es[1:])
#             user_domain_features.append(user_feature.view(batch_size, -1))
#             item_domain_features.append(item_feature.view(batch_size, items.shape[1], -1))
#         user_domain_features = torch.stack(user_domain_features, dim=1)
#         item_domain_features = torch.stack(item_domain_features, dim=1)
#         user_domain_feature = user_domain_features[torch.arange(batch_size), domains, :]
#         item_domain_feature = item_domain_features[torch.arange(batch_size), domains, :]
#         user_domain_feature_ = torch.stack([user_domain_feature] * items.shape[1], dim=1)
#
#         user_feature = self.proj(torch.cat([torch.autograd.Variable(user_common_feature), user_domain_feature], dim=-1))
#         user_feature_ = torch.stack([user_feature] * items.shape[1], dim=1)
#
#         prediction_domain = self.W(torch.cat([user_domain_feature_, item_domain_feature], dim=-1)).squeeze(-1)
#         prediction = self.W(torch.cat([user_feature_, item_domain_feature], dim=-1)).squeeze(-1)
#         return prediction, prediction_domain, user_domain_features
#
#     def get_loss(self, prediction_intra, prediction_inter, prediction_domain, label, user_domain_features):
#         batch_size = label.shape[0]
#         domain_GTloss = self.loss_fn(torch.sigmoid(prediction_domain), label).sum(-1)
#         GTloss = self.loss_fn(torch.sigmoid(prediction_inter), label).sum(-1)
#
#         # domain_pred = self.critic(user_domain_feature_mlps)
#         # domain_label = torch.zeros(batch_size, self.n_domain).to(self.device).scatter_(1, domains.unsqueeze(-1), 1).long()
#         # loss_critic = self.loss_criterion(domain_pred, domain_label).sum(dim=-1)
#         out = torch.matmul(user_domain_features, user_domain_features.transpose(1, 2))
#         deno = torch.max(out, dim=-1)[0]
#         deno.masked_fill_(deno == 0, 1)
#         out = out / deno.unsqueeze(-2)
#         i_m = torch.eye(user_domain_features.shape[-2]).unsqueeze(0).to(self.device)
#         orthLoss = torch.abs(out - i_m).view(batch_size, -1).sum(-1)
#
#         # Labelloss = self.loss_kl(user_domain_feature_mlp.softmax(dim=-1).log(),
#         #                       torch.autograd.Variable(user_common_feature_mlp.softmax(dim=-1))).sum(-1)
#         # KLloss = torch.sum(KLloss, -1)
#         # Hintloss = self.loss_mse(user_domain_feature_mlp, torch.autograd.Variable(user_common_feature_mlp)).sum(-1)
#         # Hintloss = self.loss_cos(user_domain_feature_mlp, torch.autograd.Variable(user_common_feature_mlp),
#         #                          torch.ones(batch_size).to(self.device))
#         labelLoss = self.loss_fn(torch.sigmoid(prediction_inter/self.temperature),
#                                  torch.autograd.Variable(torch.sigmoid(prediction_intra/self.temperature))).sum(-1)
#         # domain_loss = torch.mean(GTloss + domain_GTloss + self.orth_weight * orthLoss +
#         #                          pow(self.temperature, 2) * self.labelloss_weight * labelLoss)
#         domain_loss = torch.mean(GTloss + domain_GTloss +
#                                  self.orth_weight * orthLoss +
#                                  pow(self.temperature, 2) * self.labelloss_weight * labelLoss)
#         return domain_loss

# class graph_net(nn.Module):
#     def __init__(self, args, user_embedding_common, user_embedding_domains, item_embedding):
#         super(graph_net, self).__init__()
#         self.cur_dim = args.cur_dim
#         self.n_layer = args.n_layer
#         self.device = args.device
#         self.user_embedding_common = user_embedding_common
#         self.user_embedding_domains = user_embedding_domains
#         self.item_embedding = item_embedding
#         # self.gnn_model = graphSage.GraphSage(self.cur_dim, [100, 100, 100], args.n_neigh)
#         self.gnn_model = GTN(self.cur_dim, args.n_neigh, args.n_head_layer, args.n_head_cross, args.n_cross,
#                              args.dropout, args.device).to(args.device)
#         # if self.gtype == 'domain':
#         #     self.w_f = nn.Linear(self.cur_dim, self.cur_dim).to(self.device)
#
#     def forward(self, neigh, tgt_type, gtype):
#         layer_features = []
#         for l in range(self.n_layer+1):
#             neigh[l] = neigh[l].to(self.device)
#             if tgt_type == 'user':
#                 if l % 2 == 0:
#                     if gtype == 'domain':
#                         trans_feature = self.user_embedding(neigh[l])
#                     else:
#                         trans_feature = self.user_embedding(neigh[l])
#                     # trans_feature = F.normalize(trans_feature, dim=1)
#                     layer_features.append(trans_feature)
#                 else:
#                     layer_features.append(self.item_embedding(neigh[l]))
#             elif self.tgt_type == 'item':
#                 if l % 2 == 0:
#                     layer_features.append(self.item_embedding(neigh[l]))
#                 else:
#                     if self.gtype == 'domain':
#                         trans_feature = self.w_f(self.user_embedding(neigh[l]))
#                     else:
#                         trans_feature = self.user_embedding(neigh[l])
#                     # trans_feature = F.normalize(trans_feature, dim=1)
#                     layer_features.append(trans_feature)
#             else:
#                 print('Error!')
#         output = self.gnn_model(layer_features[0], layer_features[1:])
#         # output = self.gnn_model(neigh_features)
#         # output = output.view(output.shape[0], -1) #for graphsage
#         return output
