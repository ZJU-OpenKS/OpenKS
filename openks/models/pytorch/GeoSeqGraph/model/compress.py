import torch
from torch import nn
import math
import torch.nn.functional as F


class SampleSimilarities(nn.Module):
    def __init__(self, feats_dim, queueSize, T, device):
        super(SampleSimilarities, self).__init__()
        self.inputSize = feats_dim
        self.queueSize = queueSize
        self.T = T
        self.index = 0
        stdv = 1. / math.sqrt(feats_dim / 3)
        self.device = device
        self.register_buffer('memory', torch.rand(self.queueSize, feats_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, q, update=True):
        batchSize = q.shape[0]
        queue = self.memory.clone()
        out = torch.mm(queue.detach(), q.transpose(1, 0))
        out = out.transpose(0, 1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        if update:
            # update memory bank
            with torch.no_grad():
                out_ids = torch.arange(batchSize).to(self.device)
                out_ids += self.index
                out_ids = torch.fmod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, q)
                self.index = (self.index + batchSize) % self.queueSize
        return out


class CompReSS(nn.Module):
    def __init__(self , teacher_feats_dim, student_feats_dim, queue_size=128000, T=0.04):
        super(CompReSS, self).__init__()

        self.l2norm = Normalize(2).cuda()
        self.criterion = KLD().cuda()
        self.student_sample_similarities = SampleSimilarities(student_feats_dim , queue_size , T).cuda()
        self.teacher_sample_similarities = SampleSimilarities(teacher_feats_dim , queue_size , T).cuda()

    def forward(self, teacher_feats, student_feats):

        teacher_feats = self.l2norm(teacher_feats)
        student_feats = self.l2norm(student_feats)

        similarities_student = self.student_sample_similarities(student_feats)
        similarities_teacher = self.teacher_sample_similarities(teacher_feats)

        loss = self.criterion(similarities_teacher , similarities_student)
        return loss


class CompReSSA(nn.Module):
    def __init__(self, teacher_feats_dim, queue_size=128000, T=0.04):
        super(CompReSSA, self).__init__()

        self.l2norm = Normalize(2).cuda()
        self.criterion = KLD().cuda()
        self.teacher_sample_similarities = SampleSimilarities(teacher_feats_dim, queue_size, T).cuda()

    def forward(self, teacher_feats, student_feats):

        teacher_feats = self.l2norm(teacher_feats)
        student_feats = self.l2norm(student_feats)

        similarities_student = self.teacher_sample_similarities(student_feats, update=False)
        similarities_teacher = self.teacher_sample_similarities(teacher_feats)

        loss = self.criterion(similarities_teacher, similarities_student)
        return loss


class SampleSimilaritiesMomentum(nn.Module):
    def __init__(self, feats_dim, queueSize, T):
        super(SampleSimilaritiesMomentum, self).__init__()
        self.inputSize = feats_dim
        self.queueSize = queueSize
        self.T = T
        self.index = 0
        stdv = 1. / math.sqrt(feats_dim / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, feats_dim).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, feats_dim))

    def forward(self, q , q_key):
        batchSize = q.shape[0]
        queue = self.memory.clone()
        out = torch.mm(queue.detach(), q.transpose(1, 0))
        out = out.transpose(0, 1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        # update memory bank
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, q_key)
            self.index = (self.index + batchSize) % self.queueSize
        return out


class CompReSSMomentum(nn.Module):
    def __init__(self, hidden_dim, compress_memory_size, compress_t, device):
        super(CompReSSMomentum, self).__init__()
        teacher_feats_dim = hidden_dim
        student_feats_dim = hidden_dim
        queue_size = compress_memory_size
        T = compress_t
        self.l2norm = Normalize(2).to(device)
        self.criterion = KLD().to(device)
        self.student_sample_similarities = SampleSimilarities(student_feats_dim, queue_size, T, device).to(device)
        self.teacher_sample_similarities = SampleSimilarities(teacher_feats_dim, queue_size, T, device).to(device)

    def forward(self, teacher_feats, student_feats):
        teacher_feats = self.l2norm(teacher_feats)
        student_feats = self.l2norm(student_feats)

        similarities_student = self.student_sample_similarities(student_feats)
        similarities_teacher = self.teacher_sample_similarities(teacher_feats)

        loss = self.criterion(similarities_teacher, similarities_student)
        return loss


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class KLD(nn.Module):
    def forward(self, targets, inputs):
        targets = F.softmax(targets, dim=1)
        inputs = F.log_softmax(inputs, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')

