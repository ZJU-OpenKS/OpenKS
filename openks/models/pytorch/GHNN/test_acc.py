import numpy as np


file_name = "result_file_maxstep_proteins_collab.txt"
dataset_name = []
with open(file_name, mode="r") as loss_file:
    list_acc = loss_file.readlines()
    data = {}
    new_name = ''
    for row in list_acc:
        if row[-2] == '#':
            continue
        row = row.split('\t')
        data_name = row[0]
        if data_name != new_name:
            dataset_name.append(data_name)
            acc = []
            new_name = data_name
            acc.append(row[1])
            data[row[0]] = acc
        else:
            acc.append(row[1])
            data[row[0]] = acc
    
    dataset_acc = []
    for key, value in data.items():
        new_value = '/'.join(value)
        acc_list = new_value.split("/")
        acc_list = [float(i) for i in acc_list]
        arr_mean = format(np.mean(acc_list), '.4f')
        arr_std = format(np.std(acc_list), '.4f')
        dataset_acc.append((arr_mean, arr_std))

    # dataset_name = ["PROTEINS", "DD", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K", "COLLAB"]
    result_file = open(file_name, mode="a", encoding="utf-8")
    for idx, data in enumerate(dataset_name):
        result_file.write("%s"%data + "\t" + str(dataset_acc[idx]) + "\n")
    result_file.close()