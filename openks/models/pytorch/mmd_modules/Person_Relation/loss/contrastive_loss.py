
import torch
from torch import nn

def compute_contrastive_loss(logist, label):
    pred = logist.argmax(dim=1)
    acc = (pred == label).sum().float() / pred.shape[0]
    print("locus_acc", acc)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logist, label)
    return loss