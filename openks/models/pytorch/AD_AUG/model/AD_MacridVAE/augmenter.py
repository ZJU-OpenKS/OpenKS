import torch, torch.nn as nn, torch.nn.functional as F
from torch._C import device
from torch.nn import Parameter
from torch.nn.modules.linear import Linear

class Augmenter_inner(nn.Module):
    def __init__(self, n_items, arg, device):
        super(Augmenter_inner, self).__init__()
        self.W = Parameter(torch.zeros((arg.dfac, arg.dfac)))
        nn.init.xavier_uniform_(self.W)
        self.arg = arg
        self.device = device

    def forward(self, cates, h, z, x):
        probs = []
        for k in range(self.arg.kfac):
            w_k = z[k] @ self.W @ h.T
            w_k = w_k * cates[:, k].view(1, -1)

            bias = 0.0001
            delta = ((bias - (1 - bias)) * torch.rand(w_k.size()) + (1 - bias)).to(self.device)
            p_k = torch.log(delta) - torch.log(1 - delta) + w_k
            p_k = torch.sigmoid(p_k / self.arg.tau_aug)
            probs.append(p_k)
        
        all_item_prob = torch.stack(probs).mean(dim=0)
        reg_loss_aug = torch.norm(all_item_prob - x, p=1) / x.numel()

        return all_item_prob, reg_loss_aug

class Augmenter_separate(nn.Module):
    def __init__(self, n_items, arg, device):
        super(Augmenter_separate, self).__init__()
        self.mlp = nn.Sequential(
            Linear(2 * arg.dfac, arg.h_dim),
            nn.ReLU(),
            Linear(arg.h_dim, arg.h_dim),
            nn.ReLU(),
            Linear(arg.h_dim, 1)
        )
        self.items_embed = Parameter(torch.zeros((n_items, arg.dfac)))
        
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        self.items = Parameter(torch.zeros((n_items, arg.dfac)))

        self.n_items = n_items
        self.arg = arg
        self.device = device

    def forward(self, cates, h, z, x):
        probs, regs = [], 0.
        for k in range(self.arg.kfac):
            input_z = z[k].unsqueeze(1).repeat(1, self.n_items, 1)
            input_h = self.items.unsqueeze(0).repeat(input_z.size(0), 1, 1)
            w_k = self.mlp(torch.cat((input_z, input_h), dim=-1)).squeeze()
            w_k = w_k * cates[:, k].view(1, -1)

            bias = 0.0001
            delta = ((bias - (1 - bias)) * torch.rand(w_k.size()) + (1 - bias)).to(self.device)
            p_k = torch.log(delta) - torch.log(1 - delta) + w_k
            p_k = torch.sigmoid(p_k / self.arg.tau_aug)

            regs = regs + (torch.norm(p_k, p=1) / p_k.numel())
            prob_k = torch.zeros_like(p_k).to(self.device)

            prob_k[x == 1] = 1 - p_k[x == 1]
            prob_k[x == 0] = p_k[x == 0]
            probs.append(prob_k)

        all_item_prob = torch.stack(probs).mean(dim=0)
        # reg_loss_aug = torch.norm(all_item_prob - x, p=1) / x.numel()
        reg_loss_aug = regs / k

        return all_item_prob, reg_loss_aug

class Augmenter(nn.Module):
    def __init__(self, n_items, arg, device):
        super(Augmenter, self).__init__()
        self.mlp = nn.Sequential(
            Linear(2 * arg.dfac, arg.h_dim),
            nn.ReLU(),
            Linear(arg.h_dim, arg.h_dim),
            nn.ReLU(),
            Linear(arg.h_dim, 1)
        )
        
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        

        self.n_items = n_items
        self.arg = arg
        self.device = device

    def forward(self, cates, h, z, x):
        probs, regs = [], 0.
        for k in range(self.arg.kfac):
            input_z = z[k].unsqueeze(1).repeat(1, self.n_items, 1)
            input_h = h.unsqueeze(0).repeat(input_z.size(0), 1, 1)
            w_k = self.mlp(torch.cat((input_z, input_h), dim=-1)).squeeze()
            w_k = w_k * cates[:, k].view(1, -1)

            bias = 0.0001
            delta = ((bias - (1 - bias)) * torch.rand(w_k.size()) + (1 - bias)).to(self.device)
            p_k = torch.log(delta) - torch.log(1 - delta) + w_k
            p_k = torch.sigmoid(p_k / self.arg.tau_aug)

            regs = regs + (torch.norm(p_k, p=1) / p_k.numel())
            prob_k = torch.zeros_like(p_k).to(self.device)

            prob_k[x == 1] = 1 - p_k[x == 1]
            prob_k[x == 0] = p_k[x == 0]
            probs.append(prob_k)

        all_item_prob = torch.stack(probs).mean(dim=0)
        # reg_loss_aug = torch.norm(all_item_prob - x, p=1) / x.numel()
        reg_loss_aug = regs / k

        return all_item_prob, reg_loss_aug
