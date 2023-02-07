import torch.nn as nn
from .encoder import GINEncoder, MemNNEncoder, GINClassifier, MemNNClassifier


class MemNN(nn.Module):
    """ A sequence model for graph classification. """

    def __init__(self, opt):
        super(MemNN, self).__init__()
        self.encoder = MemNNEncoder(opt)
        self.classifier = MemNNClassifier(opt)

    def forward(self, x, q):
        encoding = self.encoder(x, q)
        logits = self.classifier(encoding)
        return logits