import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
# from prior_discriminator import PriorDiscriminator


class MacridVAE(nn.Module):
    def __init__(self, arg, num_items, device, aug=False):
        super(MacridVAE, self).__init__()

        kfac, dfac = arg.kfac, arg.dfac
        self.lam = arg.rg
        self.lr = arg.lr
        self.random_seed = arg.seed
        self.n_items = num_items
        self.arg = arg
        self.device = device
        self.aug = aug

        self.init_weight(kfac, dfac)

    def init_weight(self, kfac, dfac):
        self.enc_dims = [self.n_items, dfac, dfac]
        self.encoder = []
        proj_hid = self.arg.proj_hid

        if not self.aug:
            self.proj_head = nn.Sequential(
                nn.Linear(dfac, proj_hid),
                nn.ReLU(),
                # nn.Tanh(),
                nn.Linear(proj_hid, proj_hid)
                # nn.BatchNorm1d(kfac)
            )

            for m in self.proj_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)

        # self.prior_d = PriorDiscriminator(self.arg)

        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2  # mu & var
            self.encoder.append(nn.Linear(d_in, d_out))
            nn.init.xavier_uniform_(self.encoder[-1].weight)
            nn.init.trunc_normal_(self.encoder[-1].bias, std=0.001)
            self.encoder.append(nn.Tanh())
        self.encoder = nn.Sequential(*self.encoder[:-1])

        # item representation h_i, maintained by the clustering & decoding part
        self.items = Parameter(torch.zeros((self.n_items, dfac)))

        # concept(facet) prototype m_i
        self.cores = Parameter(torch.zeros((kfac, dfac)))

        nn.init.xavier_uniform_(self.items)
        nn.init.xavier_uniform_(self.cores)

    def encode(self, input, is_train):
        cores = F.normalize(self.cores, dim=1)
        items = F.normalize(self.items, dim=1)
        cates = self.generate_cate(is_train, cores, items)
        z_list = []

        for k in range(self.arg.kfac):
            cates_k = cates[:, k].reshape(1, -1)

            x_k = input * cates_k
            h_k = F.normalize(x_k, dim=1)
            h_k = F.dropout(h_k, p=1 - self.arg.keep, training=is_train)
            h_k = self.encoder(h_k)

            mu_k = F.normalize(h_k[:, :self.arg.dfac], dim=1)
            lnvarq_sub_lnvar0 = -h_k[:, self.arg.dfac:]
            std_k = torch.exp(0.5 * lnvarq_sub_lnvar0) * self.arg.std
            epsilon = torch.randn(mu_k.size()).to(self.device)
            z_k = mu_k + epsilon * std_k

            z_k = F.normalize(z_k, dim=1)
            z_list.append(z_k)

        z = torch.stack(z_list, dim=0)
        z_proj = self.proj_head(z.transpose(0, 1))

        if is_train:
            return z_proj.transpose(0, 1)
        else:
            return z
        # return z

    def forward(self, input, output, is_train=False):
        cores = F.normalize(self.cores, dim=1)
        items = F.normalize(self.items, dim=1)
        cates = self.generate_cate(is_train, cores, items)
        z_list = []
        probs, kl = 0., 0.

        for k in range(self.arg.kfac):
            cates_k = cates[:, k].reshape(1, -1)
            # Encoder part
            x_k = input * cates_k

            h_k = F.normalize(x_k, dim=1)
            h_k = F.dropout(h_k, p=1 - self.arg.keep, training=is_train)
            h_k = self.encoder(h_k)

            mu_k = F.normalize(h_k[:, :self.arg.dfac], dim=1)
            lnvarq_sub_lnvar0 = -h_k[:, self.arg.dfac:]
            std_k = torch.exp(0.5 * lnvarq_sub_lnvar0) * self.arg.std
            kl_k = 0.5 * (-lnvarq_sub_lnvar0 + torch.exp(lnvarq_sub_lnvar0) - 1.).sum(dim=1).mean()

            epsilon = torch.randn(mu_k.size()).to(self.device)
            z_k = mu_k + (epsilon * std_k if is_train else 0)

            if not self.aug:
                # kl += self.prior_d(z_k)
                kl += kl_k

            # Decoder part
            z_k = F.normalize(z_k, dim=1)
            z_list.append(z_k)
            logits_k = torch.mm(z_k, items.T) / self.arg.tau
            probs_k = torch.exp(logits_k) * cates_k
            probs += probs_k

        logits = torch.log(probs)
        logits = F.log_softmax(logits, dim=1)
        recon_loss = torch.sum(-logits * output, dim=-1).mean()

        z = torch.stack(z_list, dim=0)

        if self.aug:
            return probs

        z_proj = self.proj_head(z.transpose(0, 1))

        return probs, recon_loss, kl, z_proj.transpose(0, 1)

    def generate_reg(self):
        return torch.sum(torch.Tensor([torch.norm(param) for param in self.parameters()]))

    def generate_cate(self, is_train, cores, items):
        # clustering
        cates_logits = torch.mm(items, cores.T) / self.arg.tau

        # Gumbel-Softmax sampling
        cates_sample = RelaxedOneHotCategorical(1, logits=cates_logits).sample()
        cates_mode = torch.softmax(cates_logits, dim=1)

        # categorial matrix C
        cates = cates_sample if is_train else cates_mode
        return cates