import os
import os.path as osp
import sys
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from arguments import arg_parse
from model import train_student, GNN, RW_NN, get_entropy
from aug import TUDataset_aug
from util import nomalize_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args, teacher, student):
    teacher.train()
    student.train()

    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    criterion = nn.CrossEntropyLoss().to(device)
    
    for data, data_unlabel in zip(train_loader, unsup_train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        
        # supervised loss
        _, p_t = teacher(data.x, data.edge_index, data.batch)
        _, p_s = student(data)
        sup_loss_t = criterion(p_t, data.y)
        sup_loss_s = criterion(p_s, data.y)

        entropy_teacher = get_entropy(p_t)
        entropy_student = get_entropy(p_s)
        weight_t = args.tau * torch.exp(-entropy_teacher)
        weight_s = args.tau * torch.exp(-entropy_student)
        sup_loss_harmony = 0.5 * ((1 + weight_s) * criterion(p_t, data.y) + (1 + weight_t) * criterion(p_s, data.y))

        sup_loss = sup_loss_t + sup_loss_s + 0.01 * sup_loss_harmony

        # unsupervised loss
        unsup_loss = train_student(args.tau, data_unlabel, teacher, student)
        loss = sup_loss + unsup_loss * args.lamda

        loss.backward()

        sup_loss_all += sup_loss.item()
        unsup_loss_all += unsup_loss.item()
        loss_all += loss.item() * data.num_graphs

        optimizer.step()

    print(sup_loss_all, unsup_loss_all)
    return loss_all / len(train_loader.dataset)


def test(loader, teacher, student):
    teacher.eval()
    student.eval()

    correct_t = 0
    correct_s = 0
    for data in loader:
        data = data.to(device)

        _, p = teacher(data.x, data.edge_index, data.batch)
        _, pred = p.max(dim=1)
        correct_t += pred.eq(data.y).sum().item()

        _, p = student(data)
        _, pred = p.max(dim=1)
        correct_s += pred.eq(data.y).sum().item()
    return correct_t / len(loader.dataset), correct_s / len(loader.dataset)


def seed_everything(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # seed_everything()
    args = arg_parse()
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.DS)
    dataset = TUDataset(path, name=args.DS, cleaned=False)
    args.num_features = max(dataset.num_features, 1)
    args.num_classes = dataset.num_classes
    dataset = nomalize_data(dataset)
    dataset_origin = dataset

    dataset = TUDataset_aug(path, name=args.DS, cleaned=False, aug=args.aug)
    dataset = nomalize_data(dataset)

    # label:0.1, unlabel:0.5, val:0.1, test:0.2
    label_end = math.ceil(len(dataset)*0.1)
    unlabel_start = math.ceil(len(dataset)*0.2)
    unlabel_ratio_end = math.ceil(len(dataset)*0.7)
    unlabel_end = math.ceil(len(dataset)*0.7)
    val_end = math.ceil(len(dataset)*0.8)

    np.random.seed(seed=21)
    idx = list(np.random.permutation(len(dataset)))

    train_label_idx = idx[:label_end]
    train_unlabel_idx = idx[:unlabel_ratio_end] # unlabel_start
    valset_idx = idx[unlabel_end:val_end]
    testset_idx = idx[val_end:]

    train_labelset = [dataset[i] for i in train_label_idx]
    train_unlabelset = [dataset[i] for i in train_unlabel_idx]
    valset = [dataset_origin[i] for i in valset_idx]
    testset = [dataset_origin[i] for i in testset_idx]

    train_loader = DataLoader(train_labelset, batch_size=args.batch_size//4, shuffle=True)
    unsup_train_loader = DataLoader(train_unlabelset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    print(len(train_labelset), len(train_unlabelset), len(valset), len(testset))

    teacher = GNN(args).to(device)
    student = RW_NN(args).to(device)
    # optimizer_t = torch.optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer_s = torch.optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam([{'params': teacher.parameters()},
                                  {'params': student.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    best_val_acc = None
    for epoch in range(args.num_epoch):
        loss = train(args, teacher, student)
        val_acc_t, val_acc_s = test(val_loader, teacher, student)

        if best_val_acc is None or val_acc_t >= best_val_acc:
            test_acc_t, test_acc_s = test(test_loader, teacher, student)
            best_val_acc = val_acc_t

        # loss = train(args, teacher, student, compress)
        # val_acc_t, val_acc_s = test(val_loader, teacher, student)
        # if best_val_acc is None or val_acc_s >= best_val_acc:
        #     test_acc_t, test_acc_s = test(test_loader, teacher, student)
        #     best_val_acc = val_acc_s

        print('Epoch: {:03d}, Loss: {:.4f}, Validation ACC: {:.4f}, '
              'Test ACC_t: {:.4f}, Test ACC_s: {:.4f},'.format(epoch, loss, val_acc_t, test_acc_t, test_acc_s))

    result_file = open("./result_file_ghnn.txt", mode="a", encoding="utf-8")
    result_file.write("%s"%args.DS+str(args.tau) + "\t" + str(test_acc_t) + "\t" + str(test_acc_s) + "\t" + "%s"%args.batch_size + "\n")
    result_file.close()
