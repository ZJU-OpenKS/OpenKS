import torch, pickle
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected, coalesce
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.utils.subgraph import subgraph
from numpy.random import choice
import os


class Haw_data(Dataset):
    def __init__(self, mode='train', datatype='Electronics'):
        self.mode = mode
        self.type = datatype
        self.file_path = './Time_data/' + self.type + '_data.pkl'
        with open(self.file_path, 'rb') as f:
            self.train_set = pickle.load(f)
            self.test_set = pickle.load(f)
            self.valid_set = pickle.load(f)
            self.entire_graph = pickle.load(f)
            self.click_time = pickle.load(f)
            self.n_user, self.n_item = pickle.load(f)

        with open('./Time_data/' + self.type + '_dict.pkl', 'rb') as f:
            self.nei = pickle.load(f)

        self.entire_graph = to_undirected(torch.LongTensor(self.entire_graph))
        self.max_len = max([len(seq) for _, seq, _, _, _ in self.test_set])
        self.min_time = min(self.click_time)
        self.max_time = max(self.click_time)
        
        if self.mode == 'train':
            self.dataset = self.train_set
        elif self.mode == 'test':
            self.dataset = self.test_set
        else:
            self.dataset = self.valid_set

        self.len = len(self.dataset)
        self.n_nodes = self.n_user + self.n_item
        
        if self.mode == 'train':
            del self.test_set
            del self.valid_set
        elif self.mode == 'test':
            del self.train_set
            del self.valid_set
        else:
            del self.train_set
            del self.test_set
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        uid, seq, sid, click_time, y = self.dataset[index]
        sub_nodes, batch_edges = [], []
        # if len(seq) > 30 and self.type == 'ML':
        #     slices= choice(len(seq), 30).tolist()
        #     seq = [seq[i] for i in slices]
        if len(seq) > 20 and (self.type == 'ML'):
            seq = seq[-20:]
        for root in seq:
            root_id = root[0] + self.n_user
            neighbours, nei_1_hop, sub_edges = [], [], [[root_id], [root_id]]
            for node in self.nei[root_id]:
                if node[1] >= root[1]:
                    break
                nei_1_hop.append(node)

            thre = 10
            if len(nei_1_hop) > thre:
                slices= choice(len(nei_1_hop), thre).tolist()
                nei_1_hop = [nei_1_hop[i] for i in slices]

            sub_edges[0] += [node[0] for node in nei_1_hop]
            sub_edges[1] += [root_id for _ in nei_1_hop]

            neighbours += nei_1_hop
            
            for nei in nei_1_hop:
                nei_2_hop, sub_edges_2_hop = [], [[], []]
                for node in self.nei[nei[0]]:
                    if node[1] >= nei[1]:
                        break
                    nei_2_hop.append(node)

                if len(nei_2_hop) > 5:
                    slices = choice(len(nei_2_hop), 5).tolist()
                    nei_2_hop = [nei_2_hop[i] for i in slices]

                sub_edges_2_hop[0] = [node[0] for node in nei_2_hop]
                sub_edges_2_hop[1] = [nei[0] for _ in nei_2_hop]

                neighbours += nei_2_hop
                sub_edges[0] += sub_edges_2_hop[0]
                sub_edges[1] += sub_edges_2_hop[1]

            neighbours = list(set(map(lambda x: x[0], neighbours)))
            if root_id in neighbours:
                neighbours.remove(root_id)
            sub_nodes += neighbours

            sub_edges = torch.LongTensor(sub_edges)
            batch_edges.append(sub_edges)

        batch_edges = coalesce(torch.cat(batch_edges, -1))
        sub_nodes = list(map(lambda x: x[0] + self.n_user, seq)) + list(set(sub_nodes))
        sub_nodes = torch.LongTensor(sub_nodes)
        batch_edges, _ = subgraph(sub_nodes, batch_edges, relabel_nodes=True)
        batch_edges, _ = add_remaining_self_loops(batch_edges)
        tar_edges, tar_nodes = self.get_tar_subg(sid, click_time)

        return uid, sid, sub_nodes, batch_edges, y, torch.LongTensor(seq), click_time, tar_edges, tar_nodes

    def get_tar_subg(self, sid, click_time):
        sid += self.n_user
        neighbours, nei_1_hop, sub_edges = [], [], [[sid], [sid]]

        for node in self.nei[sid]:
            if node[1] >= click_time:
                break
            nei_1_hop.append(node)

        thre = 10
        if len(nei_1_hop) > thre:
            slices= choice(len(nei_1_hop), thre).tolist()
            nei_1_hop = [nei_1_hop[i] for i in slices]

        sub_edges[0] += [node[0] for node in nei_1_hop]
        sub_edges[1] += [sid for _ in nei_1_hop]

        neighbours += nei_1_hop
        
        for nei in nei_1_hop:
            nei_2_hop, sub_edges_2_hop = [], [[], []]
            for node in self.nei[nei[0]]:
                if node[1] >= nei[1]:
                    break
                nei_2_hop.append(node)

            if len(nei_2_hop) > 5:
                slices = choice(len(nei_2_hop), 5).tolist()
                nei_2_hop = [nei_2_hop[i] for i in slices]

            sub_edges_2_hop[0] = [node[0] for node in nei_2_hop]
            sub_edges_2_hop[1] = [nei[0] for _ in nei_2_hop]

            neighbours += nei_2_hop
            sub_edges[0] += sub_edges_2_hop[0]
            sub_edges[1] += sub_edges_2_hop[1]

        neighbours = list(set(map(lambda x: x[0], neighbours)))
        if sid in neighbours:
            neighbours.remove(sid)

        sub_edges = torch.LongTensor(sub_edges)
        tar_nodes = [sid] + neighbours
        batch_edges, _ = subgraph(tar_nodes, sub_edges, relabel_nodes=True)
        batch_edges, _ = add_remaining_self_loops(batch_edges)
        return batch_edges, tar_nodes 


def collate_HAW(batch):
    # return batch
    batch = list(zip(*batch))
    uids = torch.LongTensor(batch[0])
    sids = torch.LongTensor(batch[1])
    return uids, sids, batch[2], batch[3], batch[4], batch[5], torch.Tensor(batch[6]), batch[7], batch[8]
