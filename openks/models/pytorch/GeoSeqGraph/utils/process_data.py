import json, random
import pandas as pd
import numpy as np
import pickle as pkl
from math import cos, asin, sqrt, pi
from tqdm import tqdm


def distance(lat1, lon1, lat2, lon2):
    r = 6371
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 2 * r * asin(sqrt(a))

def remap(df: pd.DataFrame, n_user, n_poi):
    uid_dict = dict(zip(pd.unique(df['uid']), range(n_user)))
    poi_dict = dict(zip(pd.unique(df['poi']), range(n_poi)))
    df['uid'] = df['uid'].map(uid_dict)
    df['poi'] = df['poi'].map(poi_dict)
    return df

def gen_nei_graph(df: pd.DataFrame, n_user, n_poi):
    nei_dict = {idx: [] for idx in range(n_user + n_poi)}
    edges = [[], []]
    for uid, item in df.groupby('uid'):
        poi_list = [poi + n_user for poi in item['poi'].tolist()][:-1]
        nei_dict[uid] += poi_list
        edges[0] += [uid for _ in poi_list]
        edges[1] += poi_list

    for poi, item in df.groupby('poi'):
        uid_list = item['uid'].tolist()
        nei_dict[poi + n_user] += uid_list

    return nei_dict, edges

def gen_loc_graph(df: pd.DataFrame, n_user, n_poi, thre):
    poi_loc = {}
    for poi, item in df.groupby('poi'):
        poi_loc[poi] = (item['lat'].iloc[0], item['lon'].iloc[0])

    dist_mat = []
    for poi in tqdm(range(n_poi)):
        lat, lon = poi_loc[poi]
        dist_mat.append(np.array([distance(lat, lon, p[0], p[1]) for p in poi_loc.values()]))
    dist_mat = np.stack(dist_mat, axis=0)

    adj_mat = np.triu(dist_mat <= thre, k=1)
    num_edges = adj_mat.sum()
    print(f'Edges on dist_graph: {num_edges}, avg degree: {num_edges / n_poi}')

    idx_mat = np.arange(n_poi).reshape(-1, 1).repeat(n_poi, -1)
    row_idx = idx_mat[adj_mat]
    col_idx = idx_mat.T[adj_mat]
    edges = np.stack((row_idx, col_idx))

    nei_dict = {poi: [] for poi in range(n_poi)}
    for i in range(edges.shape[1]):
        src, dst = edges[:, i]
        nei_dict[src].append(dst)
        nei_dict[dst].append(src)
    return dist_mat, edges, nei_dict

random.seed(1234)
dist_threshold = 0.5

source_pth = '../raw_data/dataset_tsmc2014/dataset_TSMC2014_TKY.txt'
# source_pth = '../raw_data/dataset_tsmc2014/dataset_TSMC2014_NYC.txt'

dist_pth = '../processed_data/tky/'
# dist_pth = '../processed_data/nyc/'
col_names = ['uid', 'poi', 'cat_id', 'cat_name', 'lat', 'lon', 'offset', 'time']

review_df = pd.read_csv(source_pth, sep='\t', header=None, names=col_names, encoding='unicode_escape').loc[:, ['uid', 'poi', 'lat', 'lon']]
n_user, n_poi = pd.unique(review_df['uid']).shape[0], pd.unique(review_df['poi']).shape[0]
review_df = remap(review_df, n_user, n_poi)

loc_dict = {poi: None for poi in range(n_poi)}
for poi, item in review_df.groupby('poi'):
    lat, lon = item['lat'].iloc[0], item['lon'].iloc[0]
    loc_dict[poi] = (lat, lon)

print(f'\nData loaded from {source_pth}')
print(f'User: {n_user}\tPOI: {n_poi}\n')

print('Start loading review data')
train_set, val_test = [], []
for uid, item in review_df.groupby('uid'):
    pos_list = item['poi'].tolist()

    def gen_neg(pos_list):
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, n_poi - 1)
        return neg

    neg_list = [gen_neg(pos_list) for _ in pos_list]
    for i in range(1, len(pos_list)):
        location = (item['lat'].iloc[i], item['lon'].iloc[i])  
        if i != len(pos_list) - 1:
            train_set.append((uid, pos_list[i], pos_list[: i], loc_dict[pos_list[i]], 1))
            train_set.append((uid, neg_list[i], pos_list[: i], loc_dict[neg_list[i]], 0))
        else:
            val_test.append((uid, pos_list[i], pos_list[: i], loc_dict[pos_list[i]], 1))
            val_test.append((uid, neg_list[i], pos_list[: i], loc_dict[neg_list[i]], 0))
    
random.shuffle(train_set)
random.shuffle(val_test)

piv = len(val_test) // 2
val_set, test_set = val_test[: piv], val_test[piv: ]
print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')

with open(dist_pth+'raw/train.pkl', 'wb') as f:
    pkl.dump(train_set, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump((n_user, n_poi), f, pkl.HIGHEST_PROTOCOL)
with open(dist_pth+'raw/test.pkl', 'wb') as f:
    pkl.dump(test_set, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump((n_user, n_poi), f, pkl.HIGHEST_PROTOCOL)
with open(dist_pth+'raw/val.pkl', 'wb') as f:
    pkl.dump(val_set, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump((n_user, n_poi), f, pkl.HIGHEST_PROTOCOL)
print('CTR data dumped, generating graph neighbour dict...\n')

ui_nei_dict, ui_edges = gen_nei_graph(review_df, n_user, n_poi)
with open(dist_pth+'ui_graph.pkl', 'wb') as f:
    pkl.dump(ui_nei_dict, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(ui_edges, f, pkl.HIGHEST_PROTOCOL)
print('UI graph dumped, generating location graph...\n')

dist_mat, dist_edges, dist_dict = gen_loc_graph(review_df, n_user, n_poi, dist_threshold)
with open(dist_pth+'/processed/dist_graph.pkl', 'wb') as f:
    pkl.dump(dist_edges, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(dist_dict, f, pkl.HIGHEST_PROTOCOL)
# np.save(dist_pth+'dist_mat.npy', dist_mat)

dist_on_graph = dist_mat[dist_edges[0], dist_edges[1]]
np.save(dist_pth + 'dist_on_graph.npy', dist_on_graph)
print('Distance graph dumped, process done.')

