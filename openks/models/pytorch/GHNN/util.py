import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.data import Data

def nomalize_data(dataset):
    if dataset.data.x is None or dataset.data.x.shape[1] == 0: # torch.Size([num, 0])
        tmp = []
        for i in range(len(dataset)):
            x = torch.ones((dataset[i].num_nodes, 1))
            if dataset[i].edge_attr == None:
                tmp.append(Data(x=x, y=dataset[i].y, edge_index=dataset[i].edge_index))
            else:
                tmp.append(Data(x=x, y=dataset[i].y, edge_index=dataset[i].edge_index, edge_attr=dataset[i].edge_attr))
        dataset = tmp
    else:
        dataset = [data for data in dataset]
    return dataset