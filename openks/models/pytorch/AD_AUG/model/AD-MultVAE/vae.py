import torch
import torch.nn as nn
import torch.nn.functional as F

class MultVAE(nn.Module):
    def __init__(self, arg, num_items, device, aug=False):
        super(MultVAE, self).__init__()
        self.n_items = num_items
        self.arg = arg
        self.device = device
        self.aug = aug

        self.init_weight(arg.dfac)

    def init_weight(self, dfac):
        self.enc_dims = [self.n_items, dfac, dfac]
        self.dec_dims = [dfac, dfac, self.n_items]
        self.encoder, self.decoder = [], []

        proj_hid = self.arg.proj_hid

        if not self.aug:
            self.proj_head = nn.Sequential(
                nn.Linear(dfac, proj_hid),
                nn.ReLU(),
                # nn.Tanh(),
                nn.Linear(proj_hid, proj_hid)
                # nn.BatchNorm1d(kfac)
                # nn.Linear(proj_hid, proj_hid),
                # nn.BatchNorm1d(proj_hid)
            )

            for m in self.proj_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)


        for i, (enc_in, enc_out, dec_in, dec_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:], self.dec_dims[:-1], self.dec_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                enc_out *= 2  # mu & var

            self.encoder.append(nn.Linear(enc_in, enc_out))
            self.decoder.append(nn.Linear(dec_in, dec_out))

            nn.init.xavier_uniform_(self.encoder[-1].weight)
            nn.init.xavier_uniform_(self.decoder[-1].weight)
            nn.init.trunc_normal_(self.encoder[-1].bias, std=0.001)
            nn.init.trunc_normal_(self.decoder[-1].bias, std=0.001)
    
            self.encoder.append(nn.Tanh())
            self.decoder.append(nn.Tanh())

        self.encoder = nn.Sequential(*self.encoder[:-1])
        self.decoder = nn.Sequential(*self.decoder[:-1])

    def encode(self, input, is_train):
        h = F.normalize(input, dim=1)
        h = F.dropout(h, p=1 - self.arg.keep, training=is_train)
        h = self.encoder(h)

        mu = F.normalize(h[:, :self.arg.dfac], dim=1)
        lnvarq_sub_lnvar0 = -h[:, self.arg.dfac:]
        std = torch.exp(0.5 * lnvarq_sub_lnvar0) * self.arg.std
        epsilon = torch.randn(mu.size()).to(self.device)
        z = mu + (epsilon * std if is_train else 0)

        # z = F.normalize(z, dim=1)
        z_proj = self.proj_head(z)
        # z_proj = self.proj_head(h)

        return z_proj


    def forward(self, input, ouput, is_train=False):
        h = F.normalize(input, dim=1)

        # if not self.aug:
        #     h = F.dropout(h, p=1 - self.arg.keep, training=is_train)

        h = F.dropout(h, p=1 - self.arg.keep, training=is_train)

        h = self.encoder(h)

        mu = F.normalize(h[:, :self.arg.dfac], dim=1)
        lnvarq_sub_lnvar0 = -h[:, self.arg.dfac:]
        std = torch.exp(0.5 * lnvarq_sub_lnvar0) * self.arg.std
        epsilon = torch.randn(mu.size()).to(self.device)
        kl = 0.5 * (-lnvarq_sub_lnvar0 \
            + torch.exp(lnvarq_sub_lnvar0) - 1.).sum(dim=1).mean()
        z = mu + (epsilon * std if is_train else 0)

        # z = F.normalize(z, dim=1)

        probs = self.decoder(z)

        if self.aug:
            return probs

        logits = F.log_softmax(probs, dim=1)
        recon_loss = torch.sum(-logits * ouput, dim=-1).mean()

        z_proj = self.proj_head(z)
        # z_proj = self.proj_head(h)
        return probs, recon_loss, kl, z_proj