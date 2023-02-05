import pandas as pd
import copy
import numpy as np
import torch
import pickle



def userIndexing(filtered_datas):
    data = filtered_datas
    user_index_table = dict()
    data = list(data.groupby("reviewerID"))
    i = 1
    for turple in data:
        # unzip
        reviewerID, IDdataframe = turple
        user_index_table[reviewerID] = i
        i += 1
    return user_index_table

def itemIndexing(domainData):
    data = list(domainData.groupby("asin"))
    item_index_table = dict()
    i = 1
    for turple in data:
        # unzip
        asin, IDdataframe = turple
        item_index_table[asin] = i
        i += 1
    return item_index_table

def sample_neg_item(user, itemList, userDict, n_neg, domain): #
    neg_items = []
    while True:
        if len(neg_items) == n_neg:
            break
        neg_item = np.random.choice(itemList[domain], size=1)[0]
        if (neg_item not in userDict[domain][user]) and neg_item not in neg_items:
            neg_items.append(neg_item)
    return neg_items

def sample_neg_item_without_domain(user, itemList, userDict, n_neg): #
    neg_items = []
    while True:
        if len(neg_items) == n_neg:
            break
        neg_item = np.random.choice(itemList, size=1)[0]
        if (neg_item not in userDict[user]) and neg_item not in neg_items:
            neg_items.append(neg_item)
    return neg_items

def dataFilter(oridata, userThreshold, itemThreshold):
    data = copy.deepcopy(oridata)
    flag = True
    shape = data.shape[0]
    i = 1
    while (flag):
        flag = False
        print(i)
        i += 1
        indexItem = data[["ratings", "asin"]].groupby('asin').count() >= itemThreshold
        item = set(indexItem[indexItem['ratings'] == True].index)
        data = data[data['asin'].isin(item)]
        if data.shape[0] != shape:
            shape = data.shape[0]
            flag = True

        indexUser = data[["ratings", "reviewerID"]].groupby('reviewerID').count() >= userThreshold
        user = set(indexUser[indexUser['ratings'] == True].index)
        data = data[data['reviewerID'].isin(user)]
        if data.shape[0] != shape:
            shape = data.shape[0]
            flag = True
    return data

def process(filtered_datas, nNeg, dataset):
    #所有数据集
    if dataset != [-1]:
        filtered_datas = filtered_datas[filtered_datas['domain'].isin(dataset)]
    print(filtered_datas.shape)
    users = []
    post_items = []
    neg_items = []
    domains = []
    itemList = []
    userlist = []
    userdict = []
    for i in range(3):
        itemData = list(filtered_datas[filtered_datas['domain'].isin([i])].groupby("asin"))
        asinlist = []
        for turple in itemData:
            # unzip
            itemID, IDdataframe = turple
            asinlist.append(itemID)
            # print(asinlist)
        itemList.append(asinlist)
        print("domian item num:",i,len(asinlist))
    for i in range(3):
        userData = list(filtered_datas[filtered_datas['domain'].isin([i])].groupby("reviewerID"))
        reviewerlist = []
        asinlist = dict()
        for turple in userData:
            # unzip
            reviewerID, IDdataframe = turple
            temp = list(IDdataframe['asin'])
            asinlist[reviewerID] = temp
            reviewerlist.append(reviewerID)
            # print(asinlist)
        # print("asin",len(asinlist))
        # print(len(reviewerlist))
        print("domian user num:", i, len(reviewerlist))
        userdict.append(asinlist)
        userlist.append(reviewerlist)
    for i in range(3):
        for user in userlist[i]:
            neg_item = sample_neg_item(user, itemList, userdict, nNeg, i)
            users.append(user)
            post_items.append(userdict[i][user])
            neg_items.append(neg_item)
            domains.append(i)
    sample_data = {
    "reviewerID":users,
    "asin":post_items,
    "neg_asin":neg_items,
    "domain":domains  
}
    data = pd.DataFrame(sample_data)
    return data

def process_hr(filtered_datas, nNeg, dataset):
    #所有数据集
    if dataset != [-1]:
        filtered_datas = filtered_datas[filtered_datas['domain'].isin(dataset)]
    users = []
    post_items = []
    neg_items = []
    domains = []
    itemList = []
    userlist = []
    userdict = []
    for i in range(3):
        itemData = list(filtered_datas[filtered_datas['domain'].isin([i])].groupby("asin"))
        asinlist = []
        for turple in itemData:
            # unzip
            itemID, IDdataframe = turple
            asinlist.append(itemID)
            # print(asinlist)
        itemList.append(asinlist)
        print("domian item num:",i,len(asinlist))
    for i in range(3):
        userData = list(filtered_datas[filtered_datas['domain'].isin([i])].groupby("reviewerID"))
        reviewerlist = []
        asinlist = dict()
        for turple in userData:
            # unzip
            reviewerID, IDdataframe = turple
            temp = list(IDdataframe['asin'])
            asinlist[reviewerID] = temp
            reviewerlist.append(reviewerID)
            # print(asinlist)
        # print("asin",len(asinlist))
        # print(len(reviewerlist))
        print("domian user num:", i, len(reviewerlist))
        userdict.append(asinlist)
        userlist.append(reviewerlist)
    for i in range(3):
        for user in userlist[i]:
            for item in userdict[i][user]:
                neg_item = sample_neg_item(user, itemList, userdict, nNeg, i)
                users.append(user)
                post_items.append(item)
                neg_items.append(neg_item)
                domains.append(i)
    sample_data = {
    "reviewerID":users,
    "asin":post_items,
    "neg_asin":neg_items,
    "domain":domains
}
    data = pd.DataFrame(sample_data)
    return data

def full_data_process(filtered_datas, nNeg):
    #不分领域 数据集构造
    users = []
    post_items = []
    neg_items = []
    domains = []
    itemList = []
    userlist = []
    userdict = []

    itemData = list(filtered_datas.groupby("asin"))
    for turple in itemData:
        # unzip
        itemID, IDdataframe = turple
        itemList.append(itemID)

    userData = list(filtered_datas.groupby("reviewerID"))
    userlist = []
    userdict = dict()
    for turple in userData:
        # unzip
        reviewerID, IDdataframe = turple
        temp = list(IDdataframe['asin'])
        userdict[reviewerID] = temp
        userlist.append(reviewerID)

    for user in userlist:
        neg_item = sample_neg_item_without_domain(user, itemList, userdict, nNeg)
        users.append(user)
        post_items.append(userdict[user])
        neg_items.append(neg_item)
    sample_data = {
    "reviewerID":users,
    "asin":post_items,
    "neg_asin":neg_items
}
    data = pd.DataFrame(sample_data)
    return data

def datareader(paths, nFO, nSO):
    Datas = pd.DataFrame(columns=('asin', 'reviewerID', 'ratings', 'timestamp','domain'))
    i = 0
    for path in paths:
        df = pd.read_csv(path, names=['asin', 'reviewerID', 'ratings', 'timestamp'], nrows=400000)
        df['domain'] = i
        Datas = pd.concat([Datas,df],axis=0)
        i += 1
    filteredDatas = dataFilter(Datas, nFO, nSO)
    print('** Ratings number: **',len(filteredDatas))
    return filteredDatas


def savepkl(filtered_datas, mode):
    with open(mode + 'datafile.pkl', 'wb') as file:
        pickle.dump(filtered_datas, file)
    return

def readpkl(path):
    with open(path, 'rb') as file:
        filtered_datas = pickle.load(file)
    return filtered_datas


if __name__ == '__main__':
    paths = ['Books.csv', 'CDs_and_Vinyl.csv', 'Movies_and_TV.csv']
    n_domain = 3
    train_ratio = 0.7
    valid_ratio = 0.15
    n_layer = 2
    #n_neigh =
    domain=3
    nFO=5
    nSO=10
    nNeg = 100
    dataset = [0, 1, 2]
    test_ratio = 1 - train_ratio - valid_ratio

    skip = False
    if skip:
        print('*** data reading ***')
        with open('reduced_step1data.pkl', 'rb') as file:
            filtered_datas = pickle.load(file)
    else:
        print('*** data filtering ***')
        filtered_datas = datareader(paths, nFO, nSO)
        with open('reduced_step1data.pkl', 'wb') as file:
            pickle.dump(filtered_datas, file)

    sorted_datas = filtered_datas.sort_values(by = 'timestamp', axis = 0, ascending = True)
    print('** Ratings number: **', len(sorted_datas))
    for i in range(domain):
        temp = filtered_datas[filtered_datas['domain'].isin([i])]
        print('**domain Ratings number: **',i, len(temp))
    #print(sorted_datas)
    user_index_table = userIndexing(sorted_datas)
    item_index_table = itemIndexing(sorted_datas)
    sorted_datas['reviewerID'] = sorted_datas['reviewerID'].map(user_index_table)
    sorted_datas['asin'] = sorted_datas['asin'].map(item_index_table)
    print('** user number: **', len(user_index_table))
    print('** item number: **', len(item_index_table))

    # process(sorted_datas,nNeg, dataset)
    print('*** data spliting ***')
    trainlength = round(train_ratio * len(sorted_datas))
    validlength = round(valid_ratio * len(sorted_datas))
    testlength = len(sorted_datas) - trainlength - validlength
    train_set = sorted_datas.iloc[:trainlength]
    valid_set = sorted_datas.iloc[trainlength:trainlength+validlength]
    test_set = sorted_datas.iloc[trainlength+validlength:]

    # 全图数据
    print('** one graph valid data processing **')
    all_valid_data = full_data_process(valid_set, nNeg)
    print('** one graph test data processing **')
    all_test_data = full_data_process(test_set, nNeg)
    print('** valid data processing **')
    valid_data = process(valid_set, nNeg, dataset)
    print('** test data processing **')
    test_data = process(test_set, nNeg, dataset)
    print('*** data saving ***')
    savepkl(train_set, 'train')
    savepkl(valid_data, 'valid')
    savepkl(test_data, 'test')
    savepkl(valid_set, 'valid_')
    savepkl(test_set, 'test_')
    savepkl(all_valid_data, 'all_valid_')
    savepkl(all_test_data, 'all_test_')
    print('*** task complete ***')
