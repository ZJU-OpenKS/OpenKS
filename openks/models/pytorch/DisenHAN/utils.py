import numpy as np
from torch.utils.data import Dataset
import pickle
from copy import deepcopy
import time

def sample_neg_item_for_user(user, n_items, negative_size, adj_UI):
    neg_items = []
    while True:
        if len(neg_items) == negative_size:
            break
        neg_item = np.random.choice(range(n_items), size=1)[0]
        if (adj_UI[user, neg_item] == 0) and neg_item not in neg_items:
            neg_items.append(neg_item)
    return neg_items

class YelpDataset(Dataset):
    def __init__(self, n_nodes_list, data_path, adj_paths, n_layer, n_neigh, n_neg, train_prop, mode):
        self.n_layer = n_layer
        self.n_neigh = n_neigh
        self.n_neg = n_neg
        self.mode = mode
        self.n_nodes_list = n_nodes_list
        self.n_users, self.n_businesses, self.n_cities, self.n_categories = self.n_nodes_list
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f, encoding='iso-8859-1')
        # adj_UU, adj_UB, adj_BCi, adj_BCa, adj_UUB, adj_UBU, adj_UBUB, adj_UBCi, adj_UBCa, adj_UBCiB, adj_UBCaB, adj_BCaB, adj_BCiB
        self.adjs = []
        for adj_path in adj_paths:
            with open(adj_path, 'rb') as f:
                self.adjs.append(pickle.load(f, encoding='iso-8859-1'))

        user_neigh_adjs = [self.adjs[0], self.adjs[1]]
        business_neigh_adjs = [self.adjs[1].T, self.adjs[2], self.adjs[3]]
        city_neigh_adjs = [self.adjs[2].T]
        category_neigh_adjs = [self.adjs[3].T]

        self.user_neighs_dict_list = []
        for adj in user_neigh_adjs:
            neigh_dict = {u: np.nonzero(adj[u])[0] for u in range(len(adj))}
            self.user_neighs_dict_list.append(neigh_dict)

        self.business_neighs_dict_list = []
        for adj in business_neigh_adjs:
            neigh_dict = {b: np.nonzero(adj[b])[0] for b in range(len(adj))}
            self.business_neighs_dict_list.append(neigh_dict)

        self.city_neighs_dict_list = []
        for adj in city_neigh_adjs:
            neigh_dict = {ci: np.nonzero(adj[ci])[0] for ci in range(len(adj))}
            self.city_neighs_dict_list.append(neigh_dict)

        self.category_neighs_dict_list = []
        for adj in category_neigh_adjs:
            neigh_dict = {ca: np.nonzero(adj[ca])[0] for ca in range(len(adj))}
            self.category_neighs_dict_list.append(neigh_dict)

        self.neighs_dict_list = [self.user_neighs_dict_list, self.business_neighs_dict_list, self.city_neighs_dict_list, self.category_neighs_dict_list]
        user_neighs_type = [0, 1]
        business_neighs_type = [0, 2, 3]
        city_neighs_type = [1]
        category_neighs_type = [1]
        self.neighs_type = [user_neighs_type, business_neighs_type, city_neighs_type, category_neighs_type]

    def get_neighbor(self, nodes, n_neigh, current_type):
        nodes_shape = list(nodes.shape)
        nodes = list(nodes.flatten())
        neighs_dict_list = self.neighs_dict_list[current_type]
        neighs_list = []
        for type_index in range(len(neighs_dict_list)):
            neighbors = []
            next_type = self.neighs_type[current_type][type_index]
            for node in nodes:
                try:
                    neighbor = neighs_dict_list[type_index][node]
                except Exception:
                    neighbor = np.array([self.n_nodes_list[next_type]])
                if len(neighbor) < n_neigh:
                    # if len(neighbor) == 0:
                    neighbor = np.append(neighbor, np.array([self.n_nodes_list[next_type]]*(n_neigh-len(neighbor))), axis=0)
                    # else:
                    #     neighbor = np.random.choice(neighbor, size=n_neigh, replace=True)
                else:
                    neighbor = np.random.choice(neighbor, size=n_neigh, replace=False)
                neighbors.append(neighbor)
            neighbors = np.stack(neighbors, axis=0)
            new_nodes_shape = deepcopy(nodes_shape)
            new_nodes_shape.append(n_neigh)
            neighbors = neighbors.reshape(new_nodes_shape)
            neighs_list.append(neighbors)
        return neighs_list

    def recur_get_neighbor(self, nodes, current_type, current_layer):
        if isinstance(nodes, list):
            neighs_list = []
            for type_index in range(len(nodes)):
                next_type = self.neighs_type[current_type][type_index]
                neighs_list.append(self.recur_get_neighbor(nodes[type_index], next_type, current_layer+1))
            return neighs_list
        else:
            return self.get_neighbor(nodes, self.n_neigh[current_layer], current_type)

    def getitem(self, user, pos_businesses, neg_businesses):
        businesses = pos_businesses + neg_businesses
        user_neighs_layers = [np.array([user], dtype=np.int)]
        business_neighs_layers = [np.array(businesses, dtype=np.int)]
        for l in range(0, self.n_layer):
            user_neighs_lists = self.recur_get_neighbor(user_neighs_layers[-1], 0, 0)
            user_neighs_layers.append(user_neighs_lists)
            business_neighs_lists = self.recur_get_neighbor(business_neighs_layers[-1], 1, 0)
            business_neighs_layers.append(business_neighs_lists)
        label = np.zeros([len(businesses)], dtype=np.float32)
        label[range(len(pos_businesses))] = 1.0
        return label, user_neighs_layers, business_neighs_layers

    def __getitem__(self, index):
        if self.mode == 'train':
            user = self.data[index]['user_id']
            pos_businesses = [self.data[index]['business_id']]
            neg_businesses = sample_neg_item_for_user(user, self.n_businesses, self.n_neg, self.adjs[1])
            return self.getitem(user, pos_businesses, neg_businesses)
        else:
            user = self.data[index]['user_id']
            pos_businesses = self.data[index]['pos_business_id']
            neg_businesses = self.data[index]['neg_business_id']
            return self.getitem(user, pos_businesses, neg_businesses)

    def __len__(self):
        return len(self.data)

class AmazonDataset(Dataset):
    def __init__(self, n_nodes_list, data_path, adj_paths, n_layer, n_neigh, n_neg, train_prop, mode):
        self.n_layer = n_layer
        self.n_neigh = n_neigh
        self.n_neg = n_neg
        self.mode = mode
        self.n_nodes_list = n_nodes_list
        self.n_users, self.n_items, self.n_brands, self.n_categories = self.n_nodes_list
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        # adj_UI, adj_II, adj_IBr, adj_ICa
        self.adjs = []
        for adj_path in adj_paths:
            with open(adj_path, 'rb') as f:
                self.adjs.append(pickle.load(f))

        if train_prop<1.0 and mode=='train':
            train_len = int(len(self.data)*train_prop)
            print(train_len)
            self.data = self.data[:train_len]
            adj_UI = np.zeros([self.n_users, self.n_items])
            for i in self.data:
                adj_UI[i['user_id'], i['item_id']] = 1
            self.adjs[0] = adj_UI

        user_neigh_adjs = [self.adjs[0]]
        item_neigh_adjs = [self.adjs[0].T, self.adjs[1], self.adjs[2], self.adjs[3]]
        brand_neigh_adjs = [self.adjs[2].T]
        category_neigh_adjs = [self.adjs[3].T]

        self.user_neighs_dict_list = []
        for adj in user_neigh_adjs:
            neigh_dict = {u: np.nonzero(adj[u])[0] for u in range(len(adj))}
            self.user_neighs_dict_list.append(neigh_dict)

        self.item_neighs_dict_list = []
        for adj in item_neigh_adjs:
            neigh_dict = {i: np.nonzero(adj[i])[0] for i in range(len(adj))}
            self.item_neighs_dict_list.append(neigh_dict)

        self.brand_neighs_dict_list = []
        for adj in brand_neigh_adjs:
            neigh_dict = {br: np.nonzero(adj[br])[0] for br in range(len(adj))}
            self.brand_neighs_dict_list.append(neigh_dict)

        self.category_neighs_dict_list = []
        for adj in category_neigh_adjs:
            neigh_dict = {ca: np.nonzero(adj[ca])[0] for ca in range(len(adj))}
            self.category_neighs_dict_list.append(neigh_dict)

        self.neighs_dict_list = [self.user_neighs_dict_list, self.item_neighs_dict_list, self.brand_neighs_dict_list, self.category_neighs_dict_list]
        user_neighs_type = [1]
        item_neighs_type = [0, 1, 2, 3]
        brand_neighs_type = [1]
        category_neighs_type = [1]
        self.neighs_type = [user_neighs_type, item_neighs_type, brand_neighs_type, category_neighs_type]

    def get_neighbor(self, nodes, n_neigh, current_type, current_layer, exps):
        nodes_shape = list(nodes.shape)
        nodes = list(nodes.flatten())
        neighs_dict_list = self.neighs_dict_list[current_type]

        # neighs_dict_list = deepcopy(self.neighs_dict_list[current_type])
        # if current_layer == 0 and current_type == 0:
        #     for node in nodes:
        #         neighs_lists = list(neighs_dict_list[0][node])
        #         for exp in exps:
        #             try:
        #                 neighs_lists.remove(exp)
        #             except:
        #                 continue
        #         neighs_dict_list[0][node] = np.asarray(neighs_lists, dtype=int)
        #         # print(self.neighs_dict_list[current_type][0][node])
        #         # print(neighs_dict_list[0][node])
        #         # time.sleep(10)
        # if current_layer == 0 and current_type == 1:
        #     for node in nodes:
        #         neighs_lists = list(neighs_dict_list[0][node])
        #         for exp in exps:
        #             try:
        #                 neighs_lists.remove(exp)
        #             except:
        #                 continue
        #         neighs_dict_list[0][node] = np.asarray(neighs_lists, dtype=int)

        neighs_list = []
        for type_index in range(len(neighs_dict_list)):
            neighbors = []
            next_type = self.neighs_type[current_type][type_index]
            for node in nodes:
                try:
                    neighbor = neighs_dict_list[type_index][node]
                    if current_layer == 0 and (current_type in [0,1]):
                        neighbor = list(neighbor)
                        for exp in exps:
                            try:
                                neighbor.remove(exp)
                            except:
                                continue
                        neighbor = np.asarray(neighbor, dtype=int)
                except Exception:
                    neighbor = np.array([self.n_nodes_list[next_type]])
                if len(neighbor) < n_neigh:
                    # if len(neighbor) == 0:
                    neighbor = np.append(neighbor, np.array([self.n_nodes_list[next_type]]*(n_neigh-len(neighbor))), axis=0)
                    # else:
                    #     neighbor = np.random.choice(neighbor, size=n_neigh, replace=True)
                else:
                    neighbor = np.random.choice(neighbor, size=n_neigh, replace=False)
                neighbors.append(neighbor)
            neighbors = np.stack(neighbors, axis=0)
            new_nodes_shape = deepcopy(nodes_shape)
            new_nodes_shape.append(n_neigh)
            neighbors = neighbors.reshape(new_nodes_shape)
            neighs_list.append(neighbors)
        return neighs_list

    def recur_get_neighbor(self, nodes, current_type, current_layer, neigh_exp):
        if isinstance(nodes, list):
            neighs_list = []
            for type_index in range(len(nodes)):
                next_type = self.neighs_type[current_type][type_index]
                neighs_list.append(self.recur_get_neighbor(nodes[type_index], next_type, current_layer+1, neigh_exp))
            return neighs_list
        else:
            return self.get_neighbor(nodes, self.n_neigh[current_layer], current_type, current_layer, neigh_exp)

    def getitem(self, user, pos_items, neg_items):
        items = pos_items + neg_items
        user_neighs_layers = [np.array([user], dtype=np.int)]
        items_neighs_layers = [np.array(items, dtype=np.int)]
        for l in range(0, self.n_layer):
            user_neighs_lists = self.recur_get_neighbor(user_neighs_layers[-1], 0, 0, items)
            user_neighs_layers.append(user_neighs_lists)
            items_neighs_lists = self.recur_get_neighbor(items_neighs_layers[-1], 1, 0, [user])
            items_neighs_layers.append(items_neighs_lists)
        label = np.zeros([len(items)], dtype=np.float32)
        label[range(len(pos_items))] = 1.0
        return label, user_neighs_layers, items_neighs_layers

    def __getitem__(self, index):
        if self.mode == 'train':
            user = self.data[index]['user_id']
            pos_items = [self.data[index]['item_id']]
            neg_items = sample_neg_item_for_user(user, self.n_items, self.n_neg, self.adjs[0])
            return self.getitem(user, pos_items, neg_items)
        else:
            user = self.data[index]['user_id']
            pos_items = self.data[index]['pos_item_id']
            neg_items = self.data[index]['neg_item_id']
            return self.getitem(user, pos_items, neg_items)

    def __len__(self):
        return len(self.data)

class MovielensDataset(Dataset):
    def __init__(self, n_nodes_list, data_path, adj_paths, n_layer, n_neigh, n_neg, train_prop, mode):
        self.n_layer = n_layer
        self.n_neigh = n_neigh
        self.n_neg = n_neg
        self.mode = mode
        self.n_nodes_list = n_nodes_list
        self.n_users, self.n_movies, self.n_actors, self.n_directors, self.n_countries, self.n_genres = self.n_nodes_list
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        # adj_UI, adj_IA, adj_ID, adj_IC, adj_IG
        self.adjs = []
        for adj_path in adj_paths:
            with open(adj_path, 'rb') as f:
                self.adjs.append(pickle.load(f))

        user_neigh_adjs = [self.adjs[0]]
        movie_neigh_adjs = [self.adjs[0].T, self.adjs[1], self.adjs[2], self.adjs[3], self.adjs[4]]
        actor_neigh_adjs = [self.adjs[1].T]
        director_neigh_adjs = [self.adjs[2].T]
        country_neigh_adjs = [self.adjs[3].T]
        genre_neigh_adjs = [self.adjs[4].T]

        self.user_neighs_dict_list = []
        for adj in user_neigh_adjs:
            neigh_dict = {u: np.nonzero(adj[u])[0] for u in range(len(adj))}
            self.user_neighs_dict_list.append(neigh_dict)

        self.movie_neighs_dict_list = []
        for adj in movie_neigh_adjs:
            neigh_dict = {m: np.nonzero(adj[m])[0] for m in range(len(adj))}
            self.movie_neighs_dict_list.append(neigh_dict)

        self.actor_neighs_dict_list = []
        for adj in actor_neigh_adjs:
            neigh_dict = {a: np.nonzero(adj[a])[0] for a in range(len(adj))}
            self.actor_neighs_dict_list.append(neigh_dict)

        self.director_neighs_dict_list = []
        for adj in director_neigh_adjs:
            neigh_dict = {d: np.nonzero(adj[d])[0] for d in range(len(adj))}
            self.director_neighs_dict_list.append(neigh_dict)

        self.country_neighs_dict_list = []
        for adj in country_neigh_adjs:
            neigh_dict = {c: np.nonzero(adj[c])[0] for c in range(len(adj))}
            self.country_neighs_dict_list.append(neigh_dict)

        self.genre_neighs_dict_list = []
        for adj in genre_neigh_adjs:
            neigh_dict = {g: np.nonzero(adj[g])[0] for g in range(len(adj))}
            self.genre_neighs_dict_list.append(neigh_dict)

        self.neighs_dict_list = [self.user_neighs_dict_list, self.movie_neighs_dict_list, self.actor_neighs_dict_list,
                                 self.director_neighs_dict_list, self.country_neighs_dict_list, self.genre_neighs_dict_list]
        user_neighs_type = [1]
        movie_neighs_type = [0, 2, 3, 4, 5]
        actor_neighs_type = [1]
        director_neighs_type = [1]
        country_neighs_type = [1]
        genre_neighs_type = [1]
        self.neighs_type = [user_neighs_type, movie_neighs_type, actor_neighs_type, director_neighs_type,
                       country_neighs_type, genre_neighs_type]

    def get_neighbor(self, nodes, n_neigh, current_type):
        nodes_shape = list(nodes.shape)
        nodes = list(nodes.flatten())
        neighs_dict_list = self.neighs_dict_list[current_type]
        neighs_list = []
        for type_index in range(len(neighs_dict_list)):
            neighbors = []
            next_type = self.neighs_type[current_type][type_index]
            for node in nodes:
                try:
                    neighbor = neighs_dict_list[type_index][node]
                except Exception:
                    neighbor = np.array([self.n_nodes_list[next_type]])
                if len(neighbor) < n_neigh:
                    # if len(neighbor) == 0:
                    neighbor = np.append(neighbor, np.array([self.n_nodes_list[next_type]]*(n_neigh-len(neighbor))), axis=0)
                    # else:
                    #     neighbor = np.random.choice(neighbor, size=n_neigh, replace=True)
                else:
                    neighbor = np.random.choice(neighbor, size=n_neigh, replace=False)
                neighbors.append(neighbor)
            neighbors = np.stack(neighbors, axis=0)
            new_nodes_shape = deepcopy(nodes_shape)
            new_nodes_shape.append(n_neigh)
            neighbors = neighbors.reshape(new_nodes_shape)
            neighs_list.append(neighbors)
        return neighs_list

    def recur_get_neighbor(self, nodes, current_type, current_layer):
        if isinstance(nodes, list):
            neighs_list = []
            for type_index in range(len(nodes)):
                next_type = self.neighs_type[current_type][type_index]
                neighs_list.append(self.recur_get_neighbor(nodes[type_index], next_type, current_layer+1))
            return neighs_list
        else:
            return self.get_neighbor(nodes, self.n_neigh[current_layer], current_type)

    def getitem(self, user, pos_items, neg_items):
        items = pos_items + neg_items
        user_neighs_layers = [np.array([user], dtype=np.int)]
        items_neighs_layers = [np.array(items, dtype=np.int)]
        for l in range(0, self.n_layer):
            user_neighs_lists = self.recur_get_neighbor(user_neighs_layers[-1], 0, 0)
            user_neighs_layers.append(user_neighs_lists)
            items_neighs_lists = self.recur_get_neighbor(items_neighs_layers[-1], 1, 0)
            items_neighs_layers.append(items_neighs_lists)
        label = np.zeros([len(items)], dtype=np.float32)
        label[range(len(pos_items))] = 1.0
        return label, user_neighs_layers, items_neighs_layers

    def __getitem__(self, index):
        if self.mode == 'train':
            user = self.data[index]['user_id']
            pos_items = [self.data[index]['item_id']]
            neg_items = sample_neg_item_for_user(user, self.n_movies, self.n_neg, self.adjs[0])
            return self.getitem(user, pos_items, neg_items)
        else:
            user = self.data[index]['user_id']
            pos_items = self.data[index]['pos_item_id']
            neg_items = self.data[index]['neg_item_id']
            return self.getitem(user, pos_items, neg_items)

    def __len__(self):
        return len(self.data)

