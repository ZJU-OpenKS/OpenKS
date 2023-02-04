import pickle
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import k_hop_subgraph, to_undirected, subgraph


class MultiSessionsGraph(InMemoryDataset):
    def __init__(self, root='../processed/nyc', phrase='train', transform=None, pre_transform=None):
        assert phrase in ['train', 'test', 'val', '0.2', '0.4', '0.6', '0.8', 'vis']
        self.phrase = phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return [self.phrase + '.pkl', '../lt_dist_graph.pkl']
    
    @property
    def processed_file_names(self):
        return [self.phrase + '_session_graph_' + '.pt']
    
    def download(self):
        pass
    
    def process(self):
        with open(self.raw_dir + '/' + self.raw_file_names[0], 'rb') as f:
            data = pickle.load(f)
            n_user, n_poi = pickle.load(f)

        data_list = []
        for uid, poi, sequences, location, y in tqdm(data):
            i, x, senders, nodes = 0, [], [], {}
            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])

            edge_index = torch.LongTensor([senders[: -1], senders[1: ]])
            x = torch.LongTensor(x)
            y = torch.LongTensor([y])
            uid = torch.LongTensor([uid])
            poi = torch.LongTensor([poi])

            data_list.append(Data(x=x, edge_index=edge_index, num_nodes=len(nodes), \
                                y=y, uid=uid, poi=poi, location=location))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

