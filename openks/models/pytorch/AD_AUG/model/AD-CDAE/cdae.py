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

        proj_hid = self.arg.proj_hid

        if not self.aug:
            self.proj_head = nn.Sequential(
                nn.Linear(dfac, proj_hid),
                nn.ReLU(),
                nn.Linear(proj_hid, proj_hid)
            )

            for m in self.proj_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.trunc_normal_(self.encoder.bias, std=0.001)
        nn.init.trunc_normal_(self.decoder.bias, std=0.001)


    def encode(self, user_id, input, is_train):
        h = F.dropout(input, p=1 - self.arg.keep, training=is_train)
        h = self.encoder(h) + self.user_embedding(user_id)

        z_proj = self.proj_head(h)
        return z_proj

    def forward(self, user_id, input, output, is_train=False):
        h = F.dropout(input, p=1 - self.arg.keep, training=is_train)
        h = self.encoder(h) + self.user_embedding(user_id)
        probs = self.decoder(h)

        logits = torch.sigmoid(probs)

        if self.aug:
            return logits

        recon_loss = F.binary_cross_entropy(logits, output, reduction='none').sum(dim=1).mean()

        z_proj = self.proj_head(h)
        return logits, recon_loss, z_proj