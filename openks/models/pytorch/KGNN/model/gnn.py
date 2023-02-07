import torch.nn as nn
from .encoder import GINEncoder, MemNNEncoder, GINClassifier, MemNNClassifier


class GNN(nn.Module):
    """ A sequence model for graph classification. """

    def __init__(self, opt):
        super(GNN, self).__init__()
        self.encoder = GINEncoder(opt)
        self.classifier = GINClassifier(opt)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        encoding = self.encoder(x, edge_index, batch)
        logits = self.classifier(encoding)
        return logits
