from torch import nn

from .encoder import MLPEncoder, Classifier


class GK(nn.Module):
    """ """

    def __init__(self, opt):
        super(GK, self).__init__()
        self.encoder = MLPEncoder(opt)
        self.classifier = Classifier(opt)

    def forward(self, x):
        encoding = self.encoder(x)
        logits = self.classifier(encoding)
        return logits, encoding