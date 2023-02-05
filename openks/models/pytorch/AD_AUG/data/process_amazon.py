import copy

import pandas as pd
import numpy as np
import random

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

        va_te = [idx for idx in range(num_train, num_items_user)]
        random.shuffle(va_te)
        train_idx[:num_train] = True
        valid_idx[va_te[:num_valid]] = True
        test_idx[va_te[num_valid:]] = True

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


def remap_user_item(data, mp_user, mp_item):
    dd = data.copy()
    for i in range(len(data)):
        dd.loc[i, 'uid'] = mp_user[dd.loc[i, 'uid']]
        dd.loc[i, 'sid'] = mp_item[dd.loc[i, 'sid']]
    return dd


def to_df(file_path):
    df = pd.read_json(file_path, lines=True)
    df_ = df.rename(columns={'reviewerID': 'uid', 'asin': 'sid', 'unixReviewTime': 'time'})
    return df_

def data_filter(oridata, userThreshold, itemThreshold):
    data = copy.deepcopy(oridata)
    flag = True
    shape = data.shape[0]
    i = 0
    while(flag):
        flag = False
        print(i)
        i += 1
        indexItem = data[['time', 'sid']].groupby('sid').count() >= itemThreshold
        item = set(indexItem[indexItem['time'] == True].index)
        data = data[data['sid'].isin(item)]

        indexUser = data[['time', 'uid']].groupby('uid').count() >= userThreshold
        user = set(indexUser[indexUser['time'] == True].index)
        data = data[data['uid'].isin(user)]
        if data.shape[0] != shape:
            shape = data.shape[0]
            flag = True
    return data

if __name__ == '__main__':
    root_path = ''
    # file_path = root_path + 'amazon/reviews_Digital_Music_5.json'
    file_path = root_path + 'amazon_beauty/reviews_Beauty_5.json'
    # file_path = root_path + 'amazon/reviews_Clothing_Shoes_and_Jewelry_5.json'
    # file_path = root_path + 'amazon/reviews_Electronics_5.json'
    reviews_df = to_df(file_path)

    col_names = ['uid', 'sid', 'time']
    all_data = reviews_df.loc[:, col_names]

    mp_user = {uid: i for i, uid in enumerate(pd.unique(all_data.uid))}
    mp_item = {sid: i for i, sid in enumerate(pd.unique(all_data.sid))}
    all_data['uid'] = all_data['uid'].map(mp_user)
    all_data['sid'] = all_data['sid'].map(mp_item)

    print(all_data)

    train_data, valid_data, test_data = data_split(all_data, 0.8, 0.1, 0.1)

    train_data.to_csv(root_path + 'amazon_beauty/train.csv', index=False)
    valid_data.to_csv(root_path + 'amazon_beauty/valid.csv', index=False)
    test_data.to_csv(root_path + 'amazon_beauty/test.csv', index=False)

    unique_uid = pd.unique(pd.concat([train_data, valid_data, test_data])['uid'])
    with open(root_path + 'amazon_beauty/unique_uid.txt', 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    unique_sid = pd.unique(pd.concat([train_data, valid_data, test_data])['sid'])
    with open(root_path + 'amazon_beauty/unique_sid.txt', 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    print(len(unique_uid))
    print(len(unique_sid))
