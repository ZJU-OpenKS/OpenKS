import pickle as pkl
import pandas as pd
import random
import numpy as np

random.seed(1234)

col_names = ["reviewerID","asin","score","unixReviewTime"]


def remap(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])

def gen_graph(user_count):
    edges = [[], []]
    his_time = []
    for uid, hist in reviews_df.groupby('reviewerID'):
        item_list = hist['asin'].tolist()[: -3]
        time_list = hist['unixReviewTime'].tolist()[:-3]
        for itemID in item_list:
            edges[0].append(uid)
            edges[1].append(itemID + user_count)
        his_time += time_list
    return np.array(edges), np.array(his_time)

def gen_nei_dict(user_count, item_count, entire_graph, time_list):
    node_nei = {node: [] for node in range(user_count + item_count)}
    for i in range(entire_graph.shape[1]):
        node_nei[entire_graph[0, i]].append((entire_graph[1, i], time_list[i]))
        node_nei[entire_graph[1, i]].append((entire_graph[0, i], time_list[i]))

    for node in node_nei:
        node_nei[node] = sorted(node_nei[node], key=lambda x: x[1])
    with open(f'./Time_data/{dataset_type}_dict.pkl', 'wb') as f:
        pkl.dump(node_nei, f, pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    dataset_type = 'ML'
    reviews_df = pd.read_csv('./raw_data/ml-1m/ratings.dat', sep='::', header=None, names=col_names)
    remap(reviews_df, 'reviewerID')
    remap(reviews_df, 'asin')
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])

    user_count = len(pd.unique(reviews_df.loc[:, ['reviewerID']].squeeze()))
    item_count = len(pd.unique(reviews_df.loc[:, ['asin']].squeeze()))
    print(user_count, item_count)

    train_set, test_set = [], []
    poses, negs = 0, 0
    for reviewerID, hist in reviews_df.groupby('reviewerID'):
        his_pos = hist[(hist['score'] >= 3)]
        pos_list = his_pos[['asin', 'unixReviewTime']].values
        def gen_neg():
            neg = pos_list[0, 0]
            while neg in pos_list[:, 0]:
                neg = random.randint(0, item_count - 1)
            return neg
        neg_list = [gen_neg() for _ in range(pos_list.shape[0])]

        for i in range(1, pos_list.shape[0]):
            seq = pos_list[: i]
            if i < pos_list.shape[0] - 3:
                train_set.append((reviewerID, seq, pos_list[i, 0], pos_list[i, 1], 1))
                train_set.append((reviewerID, seq, neg_list[i], pos_list[i, 1], 0))
            else:
                test_set.append((reviewerID, seq, pos_list[i, 0], pos_list[i, 1], 1))
                test_set.append((reviewerID, seq, neg_list[i], pos_list[i, 1], 0))

    random.shuffle(train_set)
    random.shuffle(test_set)
    piv = len(test_set) // 2
    test_set, valid_set = test_set[:piv], test_set[piv:]
    print(len(train_set), len(test_set), len(valid_set))

    u_i_graph, time_list = gen_graph(user_count)
    gen_nei_dict(user_count, item_count, u_i_graph, time_list)

    with open(f'./Time_data/{dataset_type}_data.pkl', 'wb') as f:
        pkl.dump(train_set, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(test_set, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(valid_set, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(u_i_graph, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(time_list, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump((user_count, item_count), f, pkl.HIGHEST_PROTOCOL)