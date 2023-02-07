import torch
import torch.nn as nn
import torch.nn.functional as fn

# class HomoAggregate_attention(nn.Module):
#     def __init__(self, cur_dim, n_facet, dropout, training):
#         super(HomoAggregate_attention, self).__init__()
#         self.cur_dim = cur_dim
#         self.n_facet = n_facet
#         self.emb_dim = cur_dim//n_facet
#         self.dropout = dropout
#         self.training = training
#
#     def forward(self, weight_p, feature_q, neigh_feature_k, neigh_feature_v):
#         batch_size = feature_q.shape[0]
#         n_with_neg = feature_q.shape[1]
#         neigh_size = neigh_feature_k.shape[2]
#         q_ = feature_q.view(batch_size, n_with_neg, 1, self.n_facet, self.emb_dim)
#         k_ = neigh_feature_k.view(batch_size, n_with_neg, neigh_size, self.n_facet, self.emb_dim)
#         v_ = neigh_feature_v.view(batch_size, n_with_neg, neigh_size, self.n_facet*self.emb_dim)
#         p = torch.sum(k_*q_, dim=4)
#         p = torch.sum(p*weight_p.view(batch_size, n_with_neg, 1, self.n_facet), dim=3)
#         p = fn.softmax(p, dim=2)
#         p = fn.dropout(p, self.dropout, self.training)
#         relation_feature = torch.sum(v_*p.view(batch_size, n_with_neg, neigh_size, 1), dim=2)
#         return relation_feature

class HomoAggregate_attention(nn.Module):
    def __init__(self, cur_dim, n_facet, dropout, training):
        super(HomoAggregate_attention, self).__init__()
        self.cur_dim = cur_dim
        self.n_facet = n_facet
        self.emb_dim = cur_dim//n_facet
        self.dropout = dropout
        self.training = training
        self.weight_cat = nn.Parameter(torch.randn(2*self.emb_dim, dtype=torch.float), requires_grad=True)

    def forward(self, weight_p, feature, neigh_feature):
        batch_size = neigh_feature.shape[0]
        n_with_neg = neigh_feature.shape[1]
        neigh_size = neigh_feature.shape[2]
        feature = feature.view(batch_size, n_with_neg, 1, self.n_facet, self.emb_dim).repeat(1, 1, neigh_size, 1, 1)
        neigh_feature = neigh_feature.view(batch_size, n_with_neg, neigh_size, self.n_facet, self.emb_dim)
        cat_feature = torch.cat([feature, neigh_feature], dim=4)
        p = torch.relu(torch.sum(cat_feature*self.weight_cat.view(1, 1, 1, 1, 2*self.emb_dim), dim=4))

        # p = torch.sum(feature*neigh_feature, dim=4)

        p = torch.sum(p*weight_p.view(batch_size, n_with_neg, 1, self.n_facet), dim=3)
        p = fn.softmax(p, dim=2)
        p = fn.dropout(p, self.dropout, self.training)
        neigh_feature = neigh_feature.view(batch_size, n_with_neg, neigh_size, self.n_facet*self.emb_dim)
        feature = torch.relu(torch.sum(neigh_feature*p.view(batch_size, n_with_neg, neigh_size, 1), dim=2))
        # feature = torch.sum(neigh_feature*p.view(batch_size, n_with_neg, neigh_size, 1), dim=2)
        return feature


class HeteAttention(nn.Module):
    def __init__(self, cur_dim, n_facet, n_relation):
        super(HeteAttention, self).__init__()
        self.cur_dim = cur_dim
        self.n_facet = n_facet
        self.emb_dim = cur_dim//n_facet
        self.n_relation = n_relation
        self.q_encoder = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.weight_q = nn.Parameter(torch.randn(self.n_relation, self.emb_dim, dtype=torch.float), requires_grad=True)

    def forward(self, feature, relation_feature):
        batch_size = feature.shape[0]
        n_with_neg = feature.shape[1]
        relation_feature = relation_feature.view(batch_size, n_with_neg, self.n_relation, self.n_facet, self.emb_dim)
        x = feature.view(batch_size, n_with_neg, self.n_facet, self.emb_dim)
        val = self.q_encoder(relation_feature)
        q = torch.tanh(val)
        f = torch.sum(q*self.weight_q.view(1, 1, self.n_relation, 1, self.emb_dim), dim=4)
        f = fn.softmax(f, dim=3)
        # f = fn.dropout(f, self.dropout, self.training)
        u = torch.sum(val*f.view(batch_size, n_with_neg, self.n_relation, self.n_facet, 1), dim=2)
        u += x
        # u = torch.relu(u)
        u = fn.normalize(u, dim=3)
        fs = [i.squeeze(dim=2) for i in f.split(1, dim=2)]
        return u.view(batch_size, n_with_neg, self.n_facet*self.emb_dim), fs

# class UserItemAttention(nn.Module):
#     def __init__(self, cur_dim, n_facet):
#         super(UserItemAttention, self).__init__()
#         self.cur_dim = cur_dim
#         self.n_facet = n_facet
#         self.emb_dim = cur_dim // n_facet
#         atten_models = []
#         atten_models.append(nn.Linear(self.emb_dim, self.emb_dim))
#         atten_models.append(nn.Tanh())
#         atten_models.append(nn.Linear(self.emb_dim, 1, bias=False))
#         self.atten_layers = nn.Sequential(*atten_models)
#         self.predic_layer = nn.Linear(self.emb_dim, 1, bias=False)
#
#     def forward(self, user_emb, item_emb):
#         batch_size = item_emb.shape[0]
#         n_with_neg = item_emb.shape[1]
#         # ###user self attention
#         # u_emb = user_emb.view(batch_size, 1, self.n_facet, self.emb_dim).repeat(1, n_with_neg, 1, 1)
#         u_emb = user_emb.view(batch_size, 1, self.n_facet, self.emb_dim)
#         i_emb = item_emb.view(batch_size, n_with_neg, self.n_facet, self.emb_dim)
#         # u_emb = u_emb.view(batch_size, n_with_neg, self.n_facet, 1, self.emb_dim)
#         # i_emb = i_emb.view(batch_size, n_with_neg, 1, self.n_facet, self.emb_dim)
#         out = u_emb*i_emb
#         p_out = fn.softmax(self.atten_layers(out).view(batch_size, n_with_neg, self.n_facet, 1), dim=2)
#         out = torch.sum(p_out*out.view(batch_size, n_with_neg, self.n_facet, self.emb_dim), dim=2)
#         out = self.predic_layer(out)
#         return torch.squeeze(torch.sigmoid(out))

# class UserItemAttention(nn.Module):
#     def __init__(self, cur_dim, n_facet, dropout, training):
#         super(UserItemAttention, self).__init__()
#         self.cur_dim = cur_dim
#         self.n_facet = n_facet
#         self.emb_dim = cur_dim // n_facet
#
#         # atten_models = []
#         # atten_models.append(nn.Linear(self.emb_dim, self.emb_dim))
#         # atten_models.append(nn.Tanh())
#         # atten_models.append(nn.Linear(self.emb_dim, 1, bias=False))
#         # self.atten_layers = nn.Sequential(*atten_models)
#         # self.predic_layer = nn.Linear(self.emb_dim, 1, bias=False)
#         #
#         # # self.weight_h = nn.Parameter(torch.randn(n_facet*n_facet, dtype=torch.float).to(device), requires_grad=True)
#         # # self.weight_h = nn.Linear(n_facet * n_facet, 1).to(device)
#         mlp_modules = []
#         mlp_modules.append(nn.Linear(self.emb_dim, self.emb_dim))
#         mlp_modules.append(nn.ReLU())
#         mlp_modules.append(nn.Dropout(p=dropout))
#         mlp_modules.append(nn.Linear(self.emb_dim, self.emb_dim // 2))
#         mlp_modules.append(nn.ReLU())
#         self.mlp_layers = nn.Sequential(*mlp_modules)
#         self.predic_layer = nn.Linear(self.emb_dim // 2, 1)
#
#
#     def forward(self, user_emb, item_emb):
#         batch_size = item_emb.shape[0]
#         n_with_neg = item_emb.shape[1]
#         u_emb = user_emb.view(batch_size, 1, self.n_facet, self.emb_dim)
#         i_emb = item_emb.view(batch_size, n_with_neg, self.n_facet, self.emb_dim)
#         out = torch.sum(u_emb*i_emb, dim=2)
#         # p_out = fn.softmax(self.atten_layers(out).view(batch_size, n_with_neg, self.n_facet, 1), dim=2)
#         # out = torch.sum(p_out*out.view(batch_size, n_with_neg, self.n_facet, self.emb_dim), dim=2)
#         out = self.mlp_layers(out)
#         out = self.predic_layer(out)
#         return torch.squeeze(torch.sigmoid(out))



# class HeteAttention(nn.Module):
#     def __init__(self, cur_dim, n_facet, n_path, dropout, device, training):
#         super(HeteAttention, self).__init__()
#         self.cur_dim = cur_dim
#         self.n_facet = n_facet
#         self.emb_dim = cur_dim//n_facet
#         self.n_path = n_path
#         self.dropout = dropout
#         self.training = training
#         self.weight_k = [nn.Linear(self.emb_dim, self.emb_dim, bias=False).to(device) for _ in range(self.n_facet)]
#         self.weight_q = nn.Parameter(torch.randn(n_path, self.emb_dim, dtype=torch.float).to(device), requires_grad=True)
#         self.weight_v = [nn.Linear(self.emb_dim, self.emb_dim, bias=False).to(device) for _ in range(self.n_facet)]
#
#     def forward(self, feature, metapath_feature):
#         batch_size = feature.shape[0]
#         n_with_neg = feature.shape[1]
#         x = feature.view(batch_size, n_with_neg, self.n_facet, self.emb_dim)
#         metapaths = metapath_feature.split(self.emb_dim, dim=3)
#         metapath_feature = torch.stack([self.weight_k[i](metapaths[i]) for i in range(self.n_facet)], dim=3)
#         metapath_value = torch.stack([self.weight_v[i](metapaths[i]) for i in range(self.n_facet)], dim=3)
#         z = torch.tanh(metapath_feature)
#         v = metapath_value
#         p = torch.sum(z*self.weight_q.view(1, 1, self.n_path, 1, self.emb_dim), dim=4)
#         p = fn.softmax(p, dim=3)
#         p = fn.dropout(p, self.dropout, self.training)
#         u = torch.sum(v*p.view(batch_size, n_with_neg, self.n_path, self.n_facet, 1), dim=2)
#         u += x
#         u = fn.normalize(u, dim=3)
#         p = [i.squeeze(dim=2) for i in p.split(1, dim=2)]
#         return u.view(batch_size, n_with_neg, self.n_facet*self.emb_dim), p



# class HeteAttention(nn.Module):
#     def __init__(self, cur_dim, n_facet, n_path, dropout, device, training):
#         super(HeteAttention, self).__init__()
#         self.cur_dim = cur_dim
#         self.n_facet = n_facet
#         self.emb_dim = cur_dim//n_facet
#         self.n_path = n_path
#         self.dropout = dropout
#         self.training = training
#         self.weight_k = nn.Linear(self.emb_dim, self.emb_dim, bias=False).to(device)
#         self.weight_q = nn.Parameter(torch.randn(n_path, self.emb_dim, dtype=torch.float).to(device), requires_grad=True)
#         # self.weight_v = nn.Linear(self.cur_dim, self.cur_dim, bias=False).to(device)
#
#     def forward(self, feature, metapath_feature):
#         batch_size = feature.shape[0]
#         n_with_neg = feature.shape[1]
#         # feature = feature.view(batch_size, n_with_neg, self.n_facet, self.emb_dim)
#         metapath_feature = metapath_feature.view(batch_size, n_with_neg, self.n_path, self.n_facet, self.emb_dim)
#         x = feature.view(batch_size, n_with_neg, self.n_facet, self.emb_dim)
#         path_val = self.weight_k(metapath_feature)
#         z = torch.tanh(path_val)
#         # z = torch.tanh(self.weight_k(metapath_feature))
#         # v = metapath_feature
#         v = path_val
#         p = torch.sum(z*self.weight_q.view(1, 1, self.n_path, 1, self.emb_dim), dim=4)
#         p = fn.softmax(p, dim=3)
#         p = fn.dropout(p, self.dropout, self.training)
#         u = torch.sum(v*p.view(batch_size, n_with_neg, self.n_path, self.n_facet, 1), dim=2)
#         u += x
#         # u = torch.nn.LeakyReLU()(u)
#         # u = torch.relu(u)
#         u = fn.normalize(u, dim=3)
#         p = [i.squeeze(dim=2) for i in p.split(1, dim=2)]
#         return u.view(batch_size, n_with_neg, self.n_facet*self.emb_dim), p
