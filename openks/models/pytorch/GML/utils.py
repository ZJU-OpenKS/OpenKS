from torch.utils.data import Dataset
import numpy as np
import pickle
import torch

class CrossDomainDataset(Dataset):
    def __init__(self, datapath, n_layer, n_neigh, n_neg, n_domain,
                 user_domain_index_tables, item_domain_index_tables, mode):
        super(CrossDomainDataset, self).__init__()
        self.datapath = datapath
        self.n_layer = n_layer
        self.n_neigh = n_neigh
        self.n_neg = n_neg
        self.n_domain = n_domain
        self.user_domain_index_tables = user_domain_index_tables
        self.item_domain_index_tables = item_domain_index_tables
        self.mode = mode
        # self.rates = '0.6_'
        self.rates = ''

        if self.mode == 'test':
            with open(self.datapath + 'test_data_file.pkl', 'rb') as f:
                self.data = pickle.load(f)
            with open(self.datapath + self.rates + 'dict_data.pkl', 'rb') as f:
                domain_neighs_Dict = pickle.load(f)
                self.all_neighs_Dict = pickle.load(f)
            self.users = self.data['reviewerID']
            self.pos_items = self.data['asin']
            self.neg_items = self.data['neg_asin']
            self.domains = self.data['domain']
        elif self.mode == 'valid':
            with open(self.datapath + 'valid_data_file.pkl', 'rb') as f:
                self.data = pickle.load(f)
            with open(self.datapath + self.rates + 'dict_data.pkl', 'rb') as f:
                domain_neighs_Dict = pickle.load(f)
                self.all_neighs_Dict = pickle.load(f)
            self.users = self.data['reviewerID']
            self.pos_items = self.data['asin']
            self.neg_items = self.data['neg_asin']
            self.domains = self.data['domain']
        elif self.mode == 'train':
            with open(self.datapath + self.rates + 'train_data_file.pkl', 'rb') as f:
                self.data = pickle.load(f)
            with open(self.datapath + 'itemlist_data.pkl', 'rb') as f:
                self.domain_itemList = pickle.load(f)
                # self.all_itemList = pickle.load(f)
            with open(self.datapath + self.rates + 'dict_data.pkl', 'rb') as f:
                domain_neighs_Dict = pickle.load(f)
                self.all_neighs_Dict = pickle.load(f)
            # domain_userDict = self.domain_neighs_Dict[0]
        else:
            print('Error!')

        for d in range(self.n_domain):
            self.item_domain_index_tables[d][0] = 0
            self.user_domain_index_tables[d][0] = 0
        self.domain_neighs_Dict = [[{self.user_domain_index_tables[d][key]: [self.item_domain_index_tables[d][v] for v in val]
                                    for key, val in domain_neighs_Dict[0][d].items()} for d in range(self.n_domain)],
                                   [{self.item_domain_index_tables[d][key]: [self.user_domain_index_tables[d][v] for v in val]
                                    for key, val in domain_neighs_Dict[1][d].items()} for d in range(self.n_domain)]]
        self.domain_userDict = domain_neighs_Dict[0]

    def sample_neg_item(self, user, domain):
        neg_items = []
        while True:
            if len(neg_items) == self.n_neg:
                break
            neg_item = np.random.choice(self.domain_itemList[domain], size=1)[0]
            if (neg_item not in self.domain_userDict[domain][user]) and neg_item not in neg_items:
                neg_items.append(neg_item)
        return neg_items

    def get_neigh(self, u_i, neighs_dict, layer):
        neigh = []
        if u_i in neighs_dict:
            neigh_Len = len(neighs_dict[u_i])
            if neigh_Len < self.n_neigh[layer]:
                neigh = np.random.choice(neighs_dict[u_i], size=neigh_Len, replace=False).tolist()
                neigh.extend([0] * (self.n_neigh[layer] - neigh_Len))
            else:
                neigh = np.random.choice(neighs_dict[u_i], size=self.n_neigh[layer], replace=False).tolist()
        else:
            neigh.extend([0] * self.n_neigh[layer])
        return neigh

    # get neigh from cross domains for user
    def get_domain_neigh_user(self, users, domain, has_neigh):
        neighset = users
        domain_neigh = []
        for layer in range(self.n_layer):
            layer_neigh = []
            for u in neighset:
                if has_neigh:
                    temp_neigh = self.get_neigh(u, self.domain_neighs_Dict[layer%2][domain], layer)
                else:
                    temp_neigh = [0]*self.n_neigh[layer]
                layer_neigh.extend(temp_neigh)
            neighset = layer_neigh
            domain_neigh.append(torch.LongTensor(layer_neigh))
        return domain_neigh

    # get neigh from all domain for user
    def get_all_neigh_user(self, user):
        allneigh = []
        neighset = [user]
        for layer in range(self.n_layer):
            layer_neigh = []
            for u in neighset:
                temp_neigh = self.get_neigh(u, self.all_neighs_Dict[layer%2], layer)
                layer_neigh.extend(temp_neigh)
            neighset = layer_neigh
            allneigh.append(torch.LongTensor(layer_neigh))
        return allneigh

    def get_domain_neigh_item(self, items, domain, has_neigh):
        neighset = items
        domain_neigh = []
        for layer in range(self.n_layer):
            layer_neigh = []
            for i in neighset:
                if has_neigh:
                    temp_neigh = self.get_neigh(i, self.domain_neighs_Dict[(layer+1)%2][domain], layer) #domain neighs for item
                else:
                    temp_neigh = [0]*self.n_neigh[layer]
                layer_neigh.extend(temp_neigh)
            neighset = layer_neigh
            domain_neigh.append(torch.LongTensor(layer_neigh).view(len(items), -1))
        return domain_neigh

    # def get_domain_neigh_item(self, items, domain):
    #     neigh = []
    #     neighset = items
    #     for layer in range(self.n_layer):
    #         layer_neigh = []
    #         for i in neighset:
    #             temp_neigh = self.get_neigh(i, self.domain_neighs_Dict[(layer+1)%2][domain], layer) #domain neighs for item
    #             layer_neigh.extend(temp_neigh)
    #         neighset = layer_neigh
    #         neigh.append(torch.LongTensor(layer_neigh).view(len(items), -1))
    #     return neigh

    def get_all_neigh_item(self, items):
        neigh = []
        neighset = items
        for layer in range(self.n_layer):
            layer_neigh = []
            for i in neighset:
                temp_neigh = self.get_neigh(i, self.all_neighs_Dict[(layer+1)%2], layer) #domain neighs for item
                layer_neigh.extend(temp_neigh)
            neighset = layer_neigh
            neigh.append(torch.LongTensor(layer_neigh).view(len(items), -1))
        return neigh

    def getitem(self, index):
        # 'asin', 'reviewerID', 'ratings', 'timestamp','domain'
        user = self.data.iloc[index]['reviewerID']
        pos_item = self.data.iloc[index]['asin']
        domain = self.data.iloc[index]['domain']
        neg_items = self.sample_neg_item(user, domain)
        items = [pos_item] + neg_items

        user_common = torch.LongTensor([user])
        items_common = torch.LongTensor(items)

        user_lists = []
        item_lists = []
        user_neigh_lists = []
        item_neigh_lists = []
        for d in range(self.n_domain):
            if user in self.user_domain_index_tables[d]:
                user_domain = self.user_domain_index_tables[d][user]
                user_lists.append(torch.LongTensor([user_domain]))
                user_neigh_lists.append(self.get_domain_neigh_user([user_domain], d, True))
            else:
                user_lists.append(torch.LongTensor([0]))
                user_neigh_lists.append(self.get_domain_neigh_user([user], d, False))
            if d == domain:
                items_domain = [self.item_domain_index_tables[d][i] for i in items]
                item_lists.append(torch.LongTensor(items_domain))
                item_neigh_lists.append(self.get_domain_neigh_item(items_domain, d, True))
            else:
                item_lists.append(torch.LongTensor([0]*len(items)))
                item_neigh_lists.append(self.get_domain_neigh_item(items, d, False))

        user_lists.append(user_common)
        item_lists.append(items_common)

        # user_neighs = self.get_domain_neigh_user(user_domain_lists)
        user_neigh_lists.append(self.get_all_neigh_user(user))
        # item_neighs = self.get_domain_neigh_item(item_domain_lists)
        item_neigh_lists.append(self.get_all_neigh_item(items))
        label = np.zeros([len(items)], dtype=np.float32)
        label[0] = 1.0

        # user = torch.LongTensor([user])
        # items = torch.LongTensor(items)
        label = torch.tensor(label)
        return user_lists, domain, item_lists, user_neigh_lists, item_neigh_lists, label

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.getitem(index)
        elif self.mode == 'valid':
            user = self.users[index]
            items = self.neg_items[index] + [self.pos_items[index]]
            domain = self.domains[index]

            user_common = torch.LongTensor([user])
            items_common = torch.LongTensor(items)

            user_lists = []
            item_lists = []
            user_neigh_lists = []
            item_neigh_lists = []
            for d in range(self.n_domain):
                if user in self.user_domain_index_tables[d]:
                    user_domain = self.user_domain_index_tables[d][user]
                    user_lists.append(torch.LongTensor([user_domain]))
                    user_neigh_lists.append(self.get_domain_neigh_user([user_domain], d, True))
                else:
                    user_lists.append(torch.LongTensor([0]))
                    user_neigh_lists.append(self.get_domain_neigh_user([user], d, False))
                if d == domain:
                    items_domain = [self.item_domain_index_tables[d][i] for i in items]
                    item_lists.append(torch.LongTensor(items_domain))
                    item_neigh_lists.append(self.get_domain_neigh_item(items_domain, d, True))
                else:
                    item_lists.append(torch.LongTensor([0] * len(items)))
                    item_neigh_lists.append(self.get_domain_neigh_item(items, d, False))

            user_lists.append(user_common)
            item_lists.append(items_common)

            user_neigh_lists.append(self.get_all_neigh_user(user))
            item_neigh_lists.append(self.get_all_neigh_item(items))

            # user_neighs = self.get_domain_neigh_user(user)
            # user_neighs.append(self.get_all_neigh_user(user))
            # item_neighs = self.get_domain_neigh_item(items)
            # item_neighs.append(self.get_all_neigh_item(items))
            #
            # user = torch.LongTensor([user])
            # items = torch.LongTensor(items)
            return user_lists, domain, item_lists, user_neigh_lists, item_neigh_lists
        else:
            user = self.users[index]
            items = self.neg_items[index] + [self.pos_items[index]]
            domain = self.domains[index]

            user_common = torch.LongTensor([user])
            items_common = torch.LongTensor(items)

            user_lists = []
            item_lists = []
            user_neigh_lists = []
            item_neigh_lists = []
            for d in range(self.n_domain):
                if user in self.user_domain_index_tables[d]:
                    user_domain = self.user_domain_index_tables[d][user]
                    user_lists.append(torch.LongTensor([user_domain]))
                    user_neigh_lists.append(self.get_domain_neigh_user([user_domain], d, True))
                else:
                    user_lists.append(torch.LongTensor([0]))
                    user_neigh_lists.append(self.get_domain_neigh_user([user], d, False))
                if d == domain:
                    items_domain = [self.item_domain_index_tables[d][i] for i in items]
                    item_lists.append(torch.LongTensor(items_domain))
                    item_neigh_lists.append(self.get_domain_neigh_item(items_domain, d, True))
                else:
                    item_lists.append(torch.LongTensor([0] * len(items)))
                    item_neigh_lists.append(self.get_domain_neigh_item(items, d, False))

            user_lists.append(user_common)
            item_lists.append(items_common)

            user_neigh_lists.append(self.get_all_neigh_user(user))
            item_neigh_lists.append(self.get_all_neigh_item(items))

            # user_neighs = self.get_domain_neigh_user(user)
            # user_neighs.append(self.get_all_neigh_user(user))
            # item_neighs = self.get_domain_neigh_item(items, domain)
            # item_neighs.append(self.get_all_neigh_item(items))

            # user = torch.LongTensor([user])
            # items = torch.LongTensor(items)
            # return user, domain, items, user_neighs, item_neighs
            return user_lists, domain, item_lists, user_neigh_lists, item_neigh_lists

    def __len__(self):
        return len(self.data)
        # if self.mode == 'valid' or self.mode == 'test':
        #     return len(self.data)
        # else:
        #     return int(0.2 * len(self.data))