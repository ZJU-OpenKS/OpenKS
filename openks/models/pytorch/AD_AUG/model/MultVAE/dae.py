import torch
import torch.nn as nn
import torch.nn.functional as F

class MultDAE(nn.Module):
    def __init__(self, arg, num_items, device, aug=False):
        super(MultDAE, self).__init__()
        self.n_items = num_items
        self.arg = arg
        self.device = device
        self.aug = aug

        self.init_weight(arg.dfac)

    def init_weight(self, dfac):
        self.enc_dims = [self.n_items, dfac, dfac]
        self.dec_dims = [dfac, dfac, self.n_items]
        self.encoder, self.decoder = [], []

        for i, (enc_in, enc_out, dec_in, dec_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:], self.dec_dims[:-1], self.dec_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                enc_out *= 1  # mu & var

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

    def forward(self, input, is_train=False):
        h = F.normalize(input, dim=1)
        h = F.dropout(h, p=1 - self.arg.keep, training=is_train)
        h = self.encoder(h)

        probs = self.decoder(h)

        logits = F.log_softmax(probs, dim=1)
        recon_loss = torch.sum(-logits * input, dim=-1).mean()

        return probs, recon_loss