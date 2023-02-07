import torch, torch.nn as nn, torch.nn.functional as F
import math

class PriorDiscriminator(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        self.l0 = nn.Linear(arg.dfac, arg.dis_hid)
        self.l1 = nn.Linear(arg.dis_hid, arg.dis_hid)
        self.l2 = nn.Linear(arg.dis_hid, 1)

    def forward(self, x):
        prior = torch.rand_like(x)
        term_a = torch.log(self.discriminate(prior)).mean()
        term_b = torch.log(1.0 - self.discriminate(x)).mean()
        return - (term_a + term_b)


    def discriminate(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))