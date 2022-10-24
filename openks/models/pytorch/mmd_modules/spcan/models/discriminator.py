import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import copy
from collections import OrderedDict
from torch.autograd import Function
import numpy as np

BATCH_SIZE = 16
INI_DISC_SIG_SCALE = 0.1
INI_DISC_A = 1
LAST_WEIGHT_LIMIT = -2
INTEGRAL_SIGMA_VAR = 0.1

class Discriminator_Weights_Adjust(nn.Module):

    def __init__(self, form_weight, last_weight):

        super(Discriminator_Weights_Adjust, self).__init__()

        self.main_var = torch.FloatTensor([0]).cuda()
        self.l1_var = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.l2_var = nn.Parameter(torch.zeros(1),requires_grad=True)
        default_const = last_weight - form_weight

        self.k_var = torch.FloatTensor([default_const]).cuda()

        self.f_weight = form_weight
        self.l_weight = last_weight

    def forward(self, main_weight, l1_weight, l2_weight, l3_weight):

        w_main = main_weight + self.main_var

        w_l1 = l1_weight + self.l1_var
        w_l2 = l2_weight + self.l2_var

        if abs(w_l1.data[0]) > self.f_weight:
            w_l1 = w_l1 - np.sign(w_l1.data[0]) * (abs(w_l1.data[0]) - self.f_weight)        
        if abs(w_l2.data[0]) > self.f_weight:
            w_l2 = w_l2 - np.sign(w_l2.data[0]) * (abs(w_l2.data[0]) - self.f_weight)  

        w_l3 = (w_main - self.k_var) - w_l1 - w_l2

        l1_rev = np.sign(w_l1.data[0])
        l2_rev = np.sign(w_l2.data[0])
        l3_rev = np.sign(w_l3.data[0])

        return w_main, w_l1, w_l2, w_l3, l1_rev, l2_rev, l3_rev


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class BottleDiscriminator(nn.Module):
    def __init__(self):
        super(BottleDiscriminator, self).__init__()
        self.domain_pred = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))

    def forward(self, bottle, cond, l=None):
        if (cond == 'reverse'):
            bottle_reverse = grad_reverse(bottle, l*-1)
            dom_pred = self.domain_pred(bottle_reverse)
        
        else:
            dom_pred = self.domain_pred(bottle)

        return dom_pred


class LayersDiscriminator(nn.Module):
    def __init__(self, form_w, last_w):
        super(LayersDiscriminator, self).__init__()
        self.form_weight = form_w
        self.last_weight = last_w

        self.disc_weight = Discriminator_Weights_Adjust(self.form_weight, self.last_weight)

        self.domain_pred_l1 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))
        self.domain_pred_l2 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))
        self.domain_pred_l3 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))

        self.process_l1 = nn.AvgPool2d(kernel_size=56)
        self.process_l2 = nn.AvgPool2d(kernel_size=28)
        self.process_l3 = nn.AvgPool2d(kernel_size=14)

        self.l1_bottleneck = nn.Linear(256, 256)
        self.l2_bottleneck = nn.Linear(512, 256)
        self.l3_bottleneck = nn.Linear(1024, 256)

    def forward(self, l1, l2, l3, l=None,
                init_w_main=None, init_w_l1=None,
                init_w_l2=None, init_w_l3=None):

        process_l1 = self.process_l1(l1).view(l1.size(0), -1)
        bottle_l1 = self.l1_bottleneck(process_l1)

        process_l2 = self.process_l2(l2).view(l2.size(0), -1)
        bottle_l2 = self.l2_bottleneck(process_l2)

        process_l3 = self.process_l3(l3).view(l3.size(0), -1)
        bottle_l3 = self.l3_bottleneck(process_l3)

        disc_main, disc_l1, disc_l2, disc_l3, l1_rev, l2_rev, l3_rev = self.disc_weight(init_w_main, init_w_l1,
                                                                init_w_l2, init_w_l3)

        bottle_l1 = grad_reverse(bottle_l1, l*l1_rev)
        bottle_l2 = grad_reverse(bottle_l2, l*l2_rev)
        bottle_l3 = grad_reverse(bottle_l3, l*l3_rev)

        dom_pred_l1 = self.domain_pred_l1(bottle_l1)
        dom_pred_l2 = self.domain_pred_l2(bottle_l2)
        dom_pred_l3 = self.domain_pred_l3(bottle_l3)

        return dom_pred_l1.squeeze(), dom_pred_l2.squeeze(), dom_pred_l3.squeeze(), \
                            disc_main, disc_l1, disc_l2, disc_l3, l1_rev, l2_rev, l3_rev
