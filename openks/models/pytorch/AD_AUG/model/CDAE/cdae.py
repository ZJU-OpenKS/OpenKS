import torch
import torch.nn as nn
import torch.nn.functional as F

class CDAE(nn.Module):
    def __init__(self, arg, num_users, num_items, device, aug=False):
        super(CDAE, self).__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.arg = arg
        self.device = device
        self.aug = aug

        self.init_weight(arg.dfac)

    def init_weight(self, dfac):
        self.user_embedding = nn.Embedding(self.n_users, dfac)
        self.encoder = nn.Linear(self.n_items, dfac)
        self.decoder = nn.Linear(dfac, self.n_items)

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.trunc_normal_(self.encoder.bias, std=0.001)
        nn.init.trunc_normal_(self.decoder.bias, std=0.001)

        # self.enc_dims = [self.n_items, dfac, dfac]
        # self.dec_dims = [dfac, dfac, self.n_items]
        # self.encoder, self.decoder = [], []

        # for i, (enc_in, enc_out, dec_in, dec_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:], self.dec_dims[:-1], self.dec_dims[1:])):
        #     if i == len(self.enc_dims[:-1]) - 1:
        #         enc_out *= 1  # mu & var

            # self.encoder.append(nn.Linear(enc_in, enc_out))
            # self.decoder.append(nn.Linear(dec_in, dec_out))

            # nn.init.xavier_uniform_(self.encoder[-1].weight)
            # nn.init.xavier_uniform_(self.decoder[-1].weight)
            # nn.init.trunc_normal_(self.encoder[-1].bias, std=0.001)
            # nn.init.trunc_normal_(self.decoder[-1].bias, std=0.001)
    
            # self.encoder.append(nn.Tanh())
            # self.decoder.append(nn.Tanh())

        # self.encoder = nn.Sequential(*self.encoder[:-1])
        # self.decoder = nn.Sequential(*self.decoder[:-1])

    def forward(self, user_id, input, is_train=False):
        # h = F.normalize(input, dim=1)
        h = F.dropout(input, p=1 - self.arg.keep, training=is_train)
        h = self.encoder(h) + self.user_embedding(user_id)
        probs = self.decoder(h)

        # logits = F.log_softmax(probs, dim=1)
        # recon_loss = torch.sum(-logits * input, dim=-1).mean()
        logits = torch.sigmoid(probs)
        recon_loss = F.binary_cross_entropy(logits, input, reduction='none').sum(dim=1).mean()

        # return probs, recon_loss
        return logits, recon_loss