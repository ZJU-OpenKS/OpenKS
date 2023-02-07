import torch, torch.nn as nn
from cdae import CDAE

class ADV(nn.Module):
    def __init__(self, n_user, n_item, arg, device):
        super(ADV, self).__init__()
        self.n_item = n_item
        self.arg = arg
        self.device = device

        self.G = CDAE(arg, n_user, n_item, device, aug=True).to(device)
        self.D = CDAE(arg, n_user, n_item, device).to(device)

    def forward(self, u, x, anneal, opt, opt_aug, is_train=False):
        # Optimize augmenter
        self.G.train()
        self.G.zero_grad()
        self.D.eval()
        mi_aug = torch.zeros(1).to(self.device)
        aug = self.G(u, x, x, is_train=is_train)
        gumbel = self.GumbelMax(aug)

        aug_graph = torch.zeros_like(x).to(self.device)
        aug_graph[x == 0] = (1 - (1 - x) * gumbel)[x == 0]
        aug_graph[x == 1] = (x * gumbel)[x == 1]

        _, recon, z_aug = self.D(u, aug_graph, x, is_train=is_train)

        add = torch.norm(aug_graph[x == 0], p=1) / (x == 0).sum()
        drop = torch.norm(1 - aug_graph[x == 1], p=1) / (x == 1).sum()
        reg_aug = add + drop
        aug_loss = -recon + anneal * reg_aug

        aug_loss.backward()
        opt_aug.step()

        self.D.train()
        self.D.zero_grad()
        self.G.eval()
        mi_rec = torch.zeros(1).to(self.device)
        aug = self.G(u, x, x, is_train=is_train)
        gumbel = self.GumbelMax(aug)

        aug_graph = torch.zeros_like(x).to(self.device)
        aug_graph[x == 0] = (1 - (1 - x) * gumbel)[x == 0]
        aug_graph[x == 1] = (x * gumbel)[x == 1]

        _, recon, z_aug = self.D(u, aug_graph, x, is_train=is_train)
        rec_loss = recon
        kl = torch.zeros(1).to(self.device)

        rec_loss.backward()
        opt.step()

        return rec_loss, aug_loss, mi_rec, mi_aug, recon, kl, reg_aug, drop, add# torch.Tensor([0.]), reg_aug

    def GumbelMax(self, prob):
        bias = 0.0001
        delta = ((bias - (1 - bias)) * torch.rand(prob.size()) + (1 - bias)).to(self.device)
        p_k = torch.log(delta) - torch.log(1 - delta) + prob
        p_k = torch.sigmoid(p_k / self.arg.tau_aug)
        return p_k