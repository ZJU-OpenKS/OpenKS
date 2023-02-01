import numpy as np
import torch
import torch.nn as nn
from aggregator import HomoAggregate_attention, HeteAttention, UserItemAttention

class multi_HAN(nn.Module):
    def __init__(self, n_nodes_list, n_paths_type, neighs_type, args):
        super(multi_HAN, self).__init__()
        user_cuda = torch.cuda.is_available() and args.cuda
        self.device = torch.device('cuda' if user_cuda else 'cpu')
        self.dataset = args.dataset
        self.cur_dim = args.cur_dim
        self.n_facet = args.n_facet
        self.n_iter = args.n_iter
        self.n_layer = args.n_layer
        self.n_neigh = args.n_neigh
        # self.reg = args.reg
        self.dropout = args.dropout
        self.n_nodes_list = n_nodes_list
        self.n_paths_type = n_paths_type
        self.neighs_type = neighs_type
        if args.mode == 'train':
            self.training = True
        else:
            self.training = False
        if self.dataset == 'yelp':
            n_users, n_businesses, n_cities, n_categories = self.n_nodes_list
            self.user_emb_init = nn.Embedding(n_users+1, self.cur_dim, padding_idx=n_users)
            self.business_emb_init = nn.Embedding(n_businesses+1, self.cur_dim, padding_idx=n_businesses)
            self.city_emb_init = nn.Embedding(n_cities+1, self.cur_dim, padding_idx=n_cities)
            self.category_emb_init = nn.Embedding(n_categories+1, self.cur_dim, padding_idx=n_categories)
            stdv = 1. / np.sqrt(self.cur_dim)
            # torch.nn.init.xavier_uniform_(self.user_emb_init.weight)
            # torch.nn.init.xavier_uniform_(self.business_emb_init.weight)
            # torch.nn.init.xavier_uniform_(self.city_emb_init.weight)
            # torch.nn.init.xavier_uniform_(self.category_emb_init.weight)
            torch.nn.init.uniform_(self.user_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.business_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.city_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.category_emb_init.weight, -stdv, stdv)
            self.user_emb_init.weight.data[n_users] = 0
            self.business_emb_init.weight.data[n_businesses] = 0
            self.city_emb_init.weight.data[n_cities] = 0
            self.category_emb_init.weight.data[n_categories] = 0
            # self.emb_init = [self.user_emb_init, self.business_emb_init, self.city_emb_init, self.category_emb_init]
            self.emb_init = nn.ModuleList([self.user_emb_init, self.business_emb_init, self.city_emb_init, self.category_emb_init])
        elif self.dataset == 'amazon':
            n_users, n_items, n_brands, n_categories = self.n_nodes_list
            self.user_emb_init = nn.Embedding(n_users + 1, self.cur_dim, padding_idx=n_users)
            self.item_emb_init = nn.Embedding(n_items + 1, self.cur_dim, padding_idx=n_items)
            self.brand_emb_init = nn.Embedding(n_brands + 1, self.cur_dim, padding_idx=n_brands)
            self.category_emb_init = nn.Embedding(n_categories + 1, self.cur_dim, padding_idx=n_categories)
            stdv = 1. / np.sqrt(self.cur_dim)
            torch.nn.init.uniform_(self.user_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.item_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.brand_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.category_emb_init.weight, -stdv, stdv)
            self.user_emb_init.weight.data[n_users] = 0
            self.item_emb_init.weight.data[n_items] = 0
            self.brand_emb_init.weight.data[n_brands] = 0
            self.category_emb_init.weight.data[n_categories] = 0
            # self.emb_init = [self.user_emb_init, self.item_emb_init, self.brand_emb_init, self.category_emb_init]
            self.emb_init = nn.ModuleList([self.user_emb_init, self.item_emb_init, self.brand_emb_init, self.category_emb_init])
        elif self.dataset == 'movielens':
            n_users, n_movies, n_actors, n_directors, n_countries, n_genres = self.n_nodes_list
            self.user_emb_init = nn.Embedding(n_users + 1, self.cur_dim, padding_idx=n_users)
            self.movie_emb_init = nn.Embedding(n_movies + 1, self.cur_dim, padding_idx=n_movies)
            self.actor_emb_init = nn.Embedding(n_actors + 1, self.cur_dim, padding_idx=n_actors)
            self.director_emb_init = nn.Embedding(n_directors + 1, self.cur_dim, padding_idx=n_directors)
            self.country_emb_init = nn.Embedding(n_countries + 1, self.cur_dim, padding_idx=n_countries)
            self.genre_emb_init = nn.Embedding(n_genres + 1, self.cur_dim, padding_idx=n_genres)
            stdv = 1. / np.sqrt(self.cur_dim)
            torch.nn.init.uniform_(self.user_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.movie_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.actor_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.director_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.country_emb_init.weight, -stdv, stdv)
            torch.nn.init.uniform_(self.genre_emb_init.weight, -stdv, stdv)
            self.user_emb_init.weight.data[n_users] = 0
            self.movie_emb_init.weight.data[n_movies] = 0
            self.actor_emb_init.weight.data[n_actors] = 0
            self.director_emb_init.weight.data[n_directors] = 0
            self.country_emb_init.weight.data[n_countries] = 0
            self.genre_emb_init.weight.data[n_genres] = 0
            # self.emb_init = [self.user_emb_init, self.movie_emb_init, self.actor_emb_init, self.director_emb_init, self.country_emb_init, self.genre_emb_init]
            self.emb_init = nn.ModuleList([self.user_emb_init, self.movie_emb_init, self.actor_emb_init, self.director_emb_init, self.country_emb_init, self.genre_emb_init])
        else:
            print('dataset wrong!')

        self.homo_encoders = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.ModuleList([HomoAggregate_attention(self.cur_dim, self.n_facet[l], self.dropout, self.training)
                                                                    for _ in range(n_paths_type[type][i])])
                                                                    for i in range(len(n_paths_type[type]))])
                                                                    for type in range(len(self.n_nodes_list))])
                                                                    for l in range(self.n_layer)])

        self.n_paths = []
        for type in range(len(self.n_nodes_list)):
            path_num = 0
            for i in range(len(n_paths_type[type])):
                path_num += n_paths_type[type][i]
            self.n_paths.append(path_num)

        self.hete_encoders = nn.ModuleList([nn.ModuleList([HeteAttention(self.cur_dim, self.n_facet[l], self.n_paths[i], self.dropout, self.training)
                                                                            for i in range(len(self.n_nodes_list))])
                                                                            for l in range(self.n_layer)])

        self.pre_encoders = nn.ModuleList([nn.ModuleList([nn.Linear(self.cur_dim, self.cur_dim, bias=False)
                                                            for _ in range(len(self.n_nodes_list))])
                                                            for _ in range(self.n_layer)])

        self.user_item_fusion_layer = UserItemAttention(self.cur_dim, self.n_facet[0])

    def recur_aggregate(self, hidden, next_hidden, hidden_type, current_layer):
        if isinstance(hidden, list):
            updated_neighs_list = []
            for type_index in range(len(hidden)):
                next_type = self.neighs_type[hidden_type][type_index]
                updated_neighs = []
                for path_index in range(len(hidden[type_index])):
                    neigh = hidden[type_index][path_index]
                    next_neigh = next_hidden[type_index][path_index]
                    updated_hidden = self.recur_aggregate(neigh, next_neigh, next_type, current_layer)
                    updated_neighs.append(updated_hidden)
                updated_neighs_list.append(updated_neighs)
        else:
            batch_size, n_neg = hidden.shape[0], hidden.shape[1]
            support_size = n_neg
            for l in range(current_layer):
                support_size = support_size * self.n_neigh[l]
            hidden_ = hidden.view(batch_size, support_size, self.cur_dim)
            if current_layer > 0:
                hidden_ = self.pre_encoders[current_layer][hidden_type](hidden_)
            weight_p = [torch.ones(batch_size, support_size, self.n_facet[current_layer]).to(self.device) for _ in range(self.n_paths[hidden_type])]
            for clus_iter in range(self.n_iter):
                path_n = 0
                neigh_encodes = []
                for type_index in range(len(next_hidden)):
                    next_type = self.neighs_type[hidden_type][type_index]
                    for path_index in range(len(next_hidden[type_index])):
                        neigh_hidden_ = next_hidden[type_index][path_index].view(batch_size, support_size, self.n_neigh[current_layer], self.cur_dim)
                        neigh_hidden_ = self.pre_encoders[current_layer][next_type](neigh_hidden_)
                        neigh_homo = self.homo_encoders[current_layer][hidden_type][next_type][path_index](weight_p[path_n], hidden_, neigh_hidden_)
                        neigh_encodes.append(neigh_homo)
                        path_n += 1
                hidden_, weight_p = self.hete_encoders[current_layer][hidden_type](hidden_, torch.stack(neigh_encodes, dim=2))
            if current_layer > 0:
                hidden_ = torch.relu(hidden_)
            updated_neighs_list = hidden_.view_as(hidden)
        return updated_neighs_list

    def recur_emb(self, neighs_layer, current_type):
        if isinstance(neighs_layer, list):
            neighs_layer_emb = []
            for type_index in range(len(neighs_layer)):
                next_type = self.neighs_type[current_type][type_index]
                neighs_emb = []
                for path_index in range(len(neighs_layer[type_index])):
                    neigh_emb = self.recur_emb(neighs_layer[type_index][path_index], next_type)
                    neighs_emb.append(neigh_emb)
                neighs_layer_emb.append(neighs_emb)
            return neighs_layer_emb
        else:
            return self.emb_init[current_type](neighs_layer)

    def forward(self, user_neighs_layers, item_neighs_layers):
        user_hidden = []
        item_hidden = []
        for l in range(self.n_layer+1):
            user_hidden.append(self.recur_emb(user_neighs_layers[l], 0))
            item_hidden.append(self.recur_emb(item_neighs_layers[l], 1))
        user_h = user_hidden[self.n_layer]
        item_h = item_hidden[self.n_layer]
        for l in range(self.n_layer-1, -1, -1):
            user_h = self.recur_aggregate(user_hidden[l], user_h, 0, l)
            item_h = self.recur_aggregate(item_hidden[l], item_h, 1, l)
        logit = self.autocross(user_h, item_h)
        return logit

    def autocross(self, user_emb, business_emb):
        # logit = torch.sum(user_emb*business_emb, dim=2)
        # return torch.squeeze(torch.sigmoid(logit))
        logit = self.user_item_fusion_layer(user_emb, business_emb)
        return logit
