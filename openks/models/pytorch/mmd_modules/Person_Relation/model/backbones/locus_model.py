
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch._six import container_abcs

class LocusModule(nn.Module):
    """ TLocusModule
    """
    def __init__(self):
        super().__init__()
        self.T = 0.03
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn_cid = nn.BatchNorm1d(20)
        self.bn_time = nn.BatchNorm1d(20)

    def forward(self, locus_pid, locus_cid, locus_time):
        locus_time = locus_time.float()
        batch_size, locus_num = locus_cid.shape
        #print("locus_cid", locus_cid.shape, locus_cid[0])
        #print("locus_time", locus_time.shape, locus_time[0])

        locus_cid1, locus_time1 = self.locus_aug(locus_cid, locus_time)
        locus_cid2, locus_time2 = self.locus_aug(locus_cid, locus_time)
        #print("locus_cid1", locus_cid1.shape,locus_cid1[0])
        #print("locus_time1", locus_time1.shape,locus_time1[0])
        #print("locus_cid2", locus_cid2.shape, locus_cid2[0])
        #print("locus_time2", locus_time2.shape, locus_time2[0])
        q = self.base_encoder(locus_cid1, locus_time1)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            k = self.base_encoder(locus_cid2, locus_time2)
            k = F.normalize(k, dim=1)

        #positive : N*1
        #l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        #negative : N*K
        #l_neg = torch.einsum('nc,ck->nk', [q, k.transpose(1, 0)])
        #all : N*K
        l_all = torch.einsum('nc,ck->nk', [q, k.transpose(1, 0).detach()])

        logist = l_all
        logist /= self.T

        label = torch.linspace(0, batch_size-1, steps=batch_size).cuda().long()

        return logist, label

    def locus_test(self, locus_pid, locus_cid, locus_time, locus_g_pids=None):
        locus_time = locus_time.float()
        batch_size, locus_num = locus_cid.shape
        #print("locus_cid", locus_cid.shape, locus_cid[0])
        #print("locus_time", locus_time.shape, locus_time[0])

        #print("locus_cid1", locus_cid1.shape,locus_cid1[0])
        #print("locus_time1", locus_time1.shape,locus_time1[0])
        #print("locus_cid2", locus_cid2.shape, locus_cid2[0])
        #print("locus_time2", locus_time2.shape, locus_time2[0])
        q = self.base_encoder(locus_cid, locus_time)
        q = F.normalize(q, dim=1)
        k = q.clone()
        q = q[0,:].unsqueeze(0)

        l_all = torch.einsum('nc,ck->nk', [q, k.transpose(1, 0).detach()])

        logist = l_all
        logist /= self.T

        label = torch.linspace(0, batch_size-1, steps=batch_size).cuda().long()

        sorted_logist, sorted_indices = torch.sort(logist, descending=True, dim=-1)
        #print("sorted_indices", sorted_indices.shape)
        _, ind2 = torch.sort(locus_time[0], descending=False, dim=-1)
        print("base_pid", locus_pid[0])
        print("base_g_pid", locus_g_pids[0][ind2])
        print("base_cid", locus_cid[0][ind2]+1)
        print("base_time", locus_time[0][ind2])
        for i in range(3):
            print("match:", i+1)
            ind = sorted_indices[0][i+1]
            _, ind2 = torch.sort(locus_time[ind], descending=False, dim=-1)
            print("locus_pid", locus_pid[ind])
            print("locus_g_pid", locus_g_pids[ind][ind2])
            print("locus_cid", locus_cid[ind][ind2]+1)
            print("locus_time", locus_time[ind][ind2])

        return logist, label

    def base_encoder(self, locus_cid, locus_time):
        # locus_cid = self.bn_cid(locus_cid.float())
        locus_time = self.bn_time(locus_time)
        x = torch.cat((locus_cid.unsqueeze(-1), locus_time.unsqueeze(-1)), dim=2)

        x = x.transpose(2, 1)
        x = self.conv1(x)

        x = F.relu(self.bn1(x))
        x_skip = self.conv2(x)

        x = F.relu(self.bn2(x_skip))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # print("x", x.shape, x[0][:32])
        return x

    def locus_aug(self, locus_cid, locus_time):
        # 需要注意locus_time的范围，目前以分钟为单位，2022-01-01:00:00:00为开始时间
        new_locus_cid = locus_cid.clone()
        new_locus_time = locus_time.clone()
        batch_size, locus_num = new_locus_cid.shape
        for i in range(batch_size):
            for j in range(locus_num):
                random_num = random.random()
                if random_num < 0.2:
                    random_cid = random.randint(0, 50)
                    random_time = (random.random() - 0.5) * 100
                    new_locus_cid[i, j] = random_cid
                    new_locus_time[i, j] += random_time
                elif random_num < 0.4:
                    random_cid = random.randint(0, 50)
                    new_locus_cid[i, j] = random_cid
                elif random_num < 0.6:
                    random_time = (random.random() - 0.5) * 100
                    new_locus_time[i, j] += random_time
                else:
                    random_time = (random.random() - 0.5) * 10
                    new_locus_time[i, j] += random_time

        return new_locus_cid, new_locus_time