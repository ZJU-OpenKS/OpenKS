import numpy as np
from torch.utils.data import Dataset
import pickle

def sample_neg_item_for_user(user, n_businesses, negative_size, adj_UI):
    neg_businesses = []
    while True:
        if len(neg_businesses) == negative_size:
            break
        neg_business = np.random.choice(range(n_businesses), size=1)[0]
        if (adj_UI[user, neg_business] == 0) and neg_business not in neg_businesses:
            neg_businesses.append(neg_business)
    return neg_businesses

class YelpDataset(Dataset):
    def __init__(self, n_nodes_list, data_path, adj_paths, n_layer, n_neigh, n_neg, mode):
        self.n_layer = n_layer
        self.n_neigh = n_neigh
        self.n_neg = n_neg
        self.mode = mode
        self.n_nodes_list = n_nodes_list
        self.n_users, self.n_businesses, self.n_cities, self.n_categories = self.n_nodes_list
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        # adj_UU, adj_UB, adj_BCi, adj_BCa, adj_UUB, adj_UBU, adj_UBUB, adj_UBCi, adj_UBCa, adj_UBCiB, adj_UBCaB, adj_BCaB, adj_BCiB
        self.adjs = []
        for adj_path in adj_paths:
            with open(adj_path, 'rb') as f:
                self.adjs.append(pickle.load(f))

        user_user_adjs = [self.adjs[0]]
        user_business_adjs = [self.adjs[1]]
        user_neigh_adjs = [user_user_adjs, user_business_adjs]

        business_user_adjs = [self.adjs[1].T]
        business_city_adjs = [self.adjs[2]]
        business_category_adjs = [self.adjs[3]]
        business_neigh_adjs = [business_user_adjs, business_city_adjs, business_category_adjs]

        city_business_adjs = [self.adjs[2].T]
        city_neigh_adjs = [city_business_adjs]

        category_business_adjs = [self.adjs[3].T]
        category_neigh_adjs = [category_business_adjs]

        self.user_neighs_dict_list = []
        for adjs_index in range(len(user_neigh_adjs)):
            adjs = user_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {u: np.nonzero(adj[u])[0] for u in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.user_neighs_dict_list.append(neighs_dict)

        self.business_neighs_dict_list = []
        for adjs_index in range(len(business_neigh_adjs)):
            adjs = business_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {b: np.nonzero(adj[b])[0] for b in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.business_neighs_dict_list.append(neighs_dict)

        self.city_neighs_dict_list = []
        for adjs_index in range(len(city_neigh_adjs)):
            adjs = city_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {ci: np.nonzero(adj[ci])[0] for ci in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.city_neighs_dict_list.append(neighs_dict)

        self.category_neighs_dict_list = []
        for adjs_index in range(len(category_neigh_adjs)):
            adjs = category_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {ca: np.nonzero(adj[ca])[0] for ca in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.category_neighs_dict_list.append(neighs_dict)

        self.neighs_dict_list = [self.user_neighs_dict_list, self.business_neighs_dict_list, self.city_neighs_dict_list, self.category_neighs_dict_list]
        user_neighs_type = [0, 1]
        business_neighs_type = [0, 2, 3]
        city_neighs_type = [1]
        category_neighs_type = [1]
        self.neighs_type = [user_neighs_type, business_neighs_type, city_neighs_type, category_neighs_type]

    def get_neighbor(self, nodes, n_neigh, current_type):
        nodes_shape = list(nodes.shape)
        nodes_shape.append(n_neigh)
        nodes = list(nodes.flatten())
        neighs_dict_list = self.neighs_dict_list[current_type]
        neighs_list = []
        for type_index in range(len(neighs_dict_list)):
            neigh_list = []
            next_type = self.neighs_type[current_type][type_index]
            for path_index in range(len(neighs_dict_list[type_index])):
                neighbors = []
                for node in nodes:
                    try:
                        neighbor = neighs_dict_list[type_index][path_index][node]
                    except Exception:
                        neighbor = np.array([self.n_nodes_list[next_type]])
                    if len(neighbor) < n_neigh:
                        neighbor = np.append(neighbor, np.array([self.n_nodes_list[next_type]]*(n_neigh-len(neighbor))), axis=0)
                    else:
                        neighbor = np.random.choice(neighbor, size=n_neigh, replace=False)
                    neighbors.append(neighbor)
                neighbors = np.stack(neighbors, axis=0)
                neighbors = neighbors.reshape(nodes_shape)
                neigh_list.append(neighbors)
            neighs_list.append(neigh_list)
        return neighs_list

    def recur_get_neighbor(self, nodes, current_type, current_layer):
        if isinstance(nodes, list):
            neighs_list = []
            for type_index in range(len(nodes)):
                neighs = []
                next_type = self.neighs_type[current_type][type_index]
                for path_index in range(len(nodes[type_index])):
                    neighs.append(self.recur_get_neighbor(nodes[type_index][path_index], next_type, current_layer+1))
                neighs_list.append(neighs)
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
    def __init__(self, n_nodes_list, data_path, adj_paths, n_layer, n_neigh, n_neg, mode):
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

        user_item_adjs = [self.adjs[0]]
        user_neigh_adjs = [user_item_adjs]

        item_user_adjs = [self.adjs[0].T]
        item_item_adjs = [self.adjs[1]]
        item_brand_adjs = [self.adjs[2]]
        item_category_adjs = [self.adjs[3]]
        item_neigh_adjs = [item_user_adjs, item_item_adjs, item_brand_adjs, item_category_adjs]

        brand_item_adjs = [self.adjs[2].T]
        brand_neigh_adjs = [brand_item_adjs]

        category_item_adjs = [self.adjs[3].T]
        category_neigh_adjs = [category_item_adjs]

        self.user_neighs_dict_list = []
        for adjs_index in range(len(user_neigh_adjs)):
            adjs = user_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {u: np.nonzero(adj[u])[0] for u in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.user_neighs_dict_list.append(neighs_dict)

        self.item_neighs_dict_list = []
        for adjs_index in range(len(item_neigh_adjs)):
            adjs = item_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {b: np.nonzero(adj[b])[0] for b in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.item_neighs_dict_list.append(neighs_dict)

        self.brand_neighs_dict_list = []
        for adjs_index in range(len(brand_neigh_adjs)):
            adjs = brand_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {ci: np.nonzero(adj[ci])[0] for ci in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.brand_neighs_dict_list.append(neighs_dict)

        self.category_neighs_dict_list = []
        for adjs_index in range(len(category_neigh_adjs)):
            adjs = category_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {ca: np.nonzero(adj[ca])[0] for ca in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.category_neighs_dict_list.append(neighs_dict)

        self.neighs_dict_list = [self.user_neighs_dict_list, self.item_neighs_dict_list, self.brand_neighs_dict_list, self.category_neighs_dict_list]
        user_neighs_type = [1]
        item_neighs_type = [0, 1, 2, 3]
        brand_neighs_type = [1]
        category_neighs_type = [1]
        self.neighs_type = [user_neighs_type, item_neighs_type, brand_neighs_type, category_neighs_type]

    def get_neighbor(self, nodes, n_neigh, current_type):
        nodes_shape = list(nodes.shape)
        nodes_shape.append(n_neigh)
        nodes = list(nodes.flatten())
        neighs_dict_list = self.neighs_dict_list[current_type]
        neighs_list = []
        for type_index in range(len(neighs_dict_list)):
            neigh_list = []
            next_type = self.neighs_type[current_type][type_index]
            for path_index in range(len(neighs_dict_list[type_index])):
                neighbors = []
                for node in nodes:
                    try:
                        neighbor = neighs_dict_list[type_index][path_index][node]
                    except Exception:
                        neighbor = np.array([self.n_nodes_list[next_type]])
                    if len(neighbor) < n_neigh:
                        neighbor = np.append(neighbor, np.array([self.n_nodes_list[next_type]]*(n_neigh-len(neighbor))), axis=0)
                    else:
                        neighbor = np.random.choice(neighbor, size=n_neigh, replace=False)
                    neighbors.append(neighbor)
                neighbors = np.stack(neighbors, axis=0)
                neighbors = neighbors.reshape(nodes_shape)
                neigh_list.append(neighbors)
            neighs_list.append(neigh_list)
        return neighs_list

    def recur_get_neighbor(self, nodes, current_type, current_layer):
        if isinstance(nodes, list):
            neighs_list = []
            for type_index in range(len(nodes)):
                neighs = []
                next_type = self.neighs_type[current_type][type_index]
                for path_index in range(len(nodes[type_index])):
                    neighs.append(self.recur_get_neighbor(nodes[type_index][path_index], next_type, current_layer+1))
                neighs_list.append(neighs)
            return neighs_list
        else:
            return self.get_neighbor(nodes, self.n_neigh[current_layer][current_type], current_type)

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
    def __init__(self, n_nodes_list, data_path, adj_paths, n_layer, n_neigh, n_neg, mode):
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

        user_movie_adjs = [self.adjs[0]]
        user_neigh_adjs = [user_movie_adjs]

        movie_user_adjs = [self.adjs[0].T]
        movie_actor_adjs = [self.adjs[1]]
        movie_director_adjs = [self.adjs[2]]
        movie_country_adjs = [self.adjs[3]]
        movie_genre_adjs = [self.adjs[4]]
        movie_neigh_adjs = [movie_user_adjs, movie_actor_adjs, movie_director_adjs, movie_country_adjs, movie_genre_adjs]

        actor_movie_adjs = [self.adjs[1].T]
        actor_neigh_adjs = [actor_movie_adjs]

        director_movie_adjs = [self.adjs[2].T]
        director_neigh_adjs = [director_movie_adjs]

        country_movie_adjs = [self.adjs[3].T]
        country_neigh_adjs = [country_movie_adjs]

        genre_movie_adjs = [self.adjs[4].T]
        genre_neigh_adjs = [genre_movie_adjs]

        self.user_neighs_dict_list = []
        for adjs_index in range(len(user_neigh_adjs)):
            adjs = user_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {u: np.nonzero(adj[u])[0] for u in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.user_neighs_dict_list.append(neighs_dict)

        self.movie_neighs_dict_list = []
        for adjs_index in range(len(movie_neigh_adjs)):
            adjs = movie_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {m: np.nonzero(adj[m])[0] for m in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.movie_neighs_dict_list.append(neighs_dict)

        self.actor_neighs_dict_list = []
        for adjs_index in range(len(actor_neigh_adjs)):
            adjs = actor_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {a: np.nonzero(adj[a])[0] for a in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.actor_neighs_dict_list.append(neighs_dict)

        self.director_neighs_dict_list = []
        for adjs_index in range(len(director_neigh_adjs)):
            adjs = director_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {d: np.nonzero(adj[d])[0] for d in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.director_neighs_dict_list.append(neighs_dict)

        self.country_neighs_dict_list = []
        for adjs_index in range(len(country_neigh_adjs)):
            adjs = country_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {c: np.nonzero(adj[c])[0] for c in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.country_neighs_dict_list.append(neighs_dict)

        self.genre_neighs_dict_list = []
        for adjs_index in range(len(genre_neigh_adjs)):
            adjs = genre_neigh_adjs[adjs_index]
            neighs_dict = []
            for adj in adjs:
                neigh_dict = {g: np.nonzero(adj[g])[0] for g in range(len(adj))}
                neighs_dict.append(neigh_dict)
            self.genre_neighs_dict_list.append(neighs_dict)


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
        nodes_shape.append(n_neigh)
        nodes = list(nodes.flatten())
        neighs_dict_list = self.neighs_dict_list[current_type]
        neighs_list = []
        for type_index in range(len(neighs_dict_list)):
            neigh_list = []
            next_type = self.neighs_type[current_type][type_index]
            for path_index in range(len(neighs_dict_list[type_index])):
                neighbors = []
                for node in nodes:
                    try:
                        neighbor = neighs_dict_list[type_index][path_index][node]
                    except Exception:
                        neighbor = np.array([self.n_nodes_list[next_type]])
                    if len(neighbor) < n_neigh:
                        neighbor = np.append(neighbor, np.array([self.n_nodes_list[next_type]]*(n_neigh-len(neighbor))), axis=0)
                    else:
                        neighbor = np.random.choice(neighbor, size=n_neigh, replace=False)
                    neighbors.append(neighbor)
                neighbors = np.stack(neighbors, axis=0)
                neighbors = neighbors.reshape(nodes_shape)
                neigh_list.append(neighbors)
            neighs_list.append(neigh_list)
        return neighs_list

    def recur_get_neighbor(self, nodes, current_type, current_layer):
        if isinstance(nodes, list):
            neighs_list = []
            for type_index in range(len(nodes)):
                neighs = []
                next_type = self.neighs_type[current_type][type_index]
                for path_index in range(len(nodes[type_index])):
                    neighs.append(self.recur_get_neighbor(nodes[type_index][path_index], next_type, current_layer+1))
                neighs_list.append(neighs)
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

