import time

import pandas as pd
import numpy as np

def load_data(path, col_names):
    all_data = pd.read_csv(path, sep='\t', header=None, names=col_names)
    all_data = all_data.loc[:, ['uid', 'sid', 'time']]
    # print(all_data['uid'])
    all_data['uid'] -= 1
    all_data['sid'] -= 1
    # all_data.sort_values(by='time', ascending=True)
    return all_data

def data_split(all_data, train_ratio, valid_ratio, test_ratio):
    data_group = all_data.groupby('uid')
    train_list, valid_list, test_list = [], [], []

    num_zero_train, num_zero_valid, num_zero_test = 0, 0, 0

    for _, group in data_group:
        user = pd.unique(group.uid)[0]
        num_items_user = len(group)
        num_train = int(train_ratio * num_items_user)
        num_valid = int(valid_ratio * num_items_user)
        group = group.sort_values(by='time', ascending=True)
        train_idx = np.zeros(num_items_user, dtype='bool')
        valid_idx = np.zeros(num_items_user, dtype='bool')
        test_idx = np.zeros(num_items_user, dtype='bool')

        train_idx[:num_train] = True
        valid_idx[num_train:num_train+num_valid] = True
        test_idx[num_train+num_valid:] = True

        if len(group[train_idx]) == 0:
            num_zero_train += 1
        else:
            train_list.append(group[train_idx])

        if len(group[valid_idx]) == 0:
            num_zero_valid += 1
        else:
            valid_list.append(group[valid_idx])

        if len(group[test_idx]) == 0:
            num_zero_test += 1
        else:
            test_list.append(group[test_idx])

    train_df = pd.concat(train_list)
    valid_df = pd.concat(valid_list)
    test_df = pd.concat(test_list)

    print('# zero train, valid, test: %d, %d, %d' % (num_zero_train, num_zero_valid, num_zero_test))

    return train_df.loc[:, ['uid', 'sid']], valid_df.loc[:, ['uid', 'sid']], test_df.loc[:, ['uid', 'sid']]


if __name__ == '__main__':
    # file_path = './ml-1m/ratings.dat'
    root_path = ''
    file_path = root_path + 'ml-100k_1/ml-100k/u.data'
    col_names = ['uid', 'sid', 'score', 'time']
    all_data = load_data(file_path, col_names)
    # print(len(pd.unique(all_data['uid'])))
    # print(len(pd.unique(all_data['sid'])))
    # print(len(all_data))
    # time.sleep(1000)

    train_data, valid_data, test_data = data_split(all_data, 0.8, 0.1, 0.1)

    train_data.to_csv(root_path + 'ml-100k_1/train.csv', index=False)
    valid_data.to_csv(root_path + 'ml-100k_1/valid.csv', index=False)
    test_data.to_csv(root_path + 'ml-100k_1/test.csv', index=False)

    unique_uid = pd.unique(all_data['uid'])
    with open(root_path + 'ml-100k_1/unique_uid.txt', 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    unique_sid = pd.unique(all_data['sid'])
    with open(root_path + 'ml-100k_1/unique_sid.txt', 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    print(len(unique_uid))
    print(len(unique_sid))
