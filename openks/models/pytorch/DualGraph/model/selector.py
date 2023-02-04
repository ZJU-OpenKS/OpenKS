from torch import nn
from .encoder import GINEncoder, Classifier


class Selector(nn.Module):
    def __init__(self, opt):
        super(Selector, self).__init__()
        self.encoder = GINEncoder(opt)
        self.classifier = Classifier(opt)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        encoding = self.encoder(x, edge_index, batch)
        logits = self.classifier(encoding)
        return encoding, logits

