import torch.nn as nn
import torch
import torch.nn.functional as F


BATCH_SIZE = 16
INI_DISC_SIG_SCALE = 0.1
INI_DISC_A = 1
LAST_WEIGHT_LIMIT = -2
INTEGRAL_SIGMA_VAR = 0.1


class FeatureExtractor(nn.Module):
    def __init__(self, pre_trained):
        super(FeatureExtractor, self).__init__()

        self.conv1 = pre_trained.conv1
        self.bn1 = pre_trained.bn1
        self.relu = pre_trained.relu
        self.maxpool = pre_trained.maxpool

        self.layer1 = pre_trained.layer1
        self.layer2 = pre_trained.layer2
        self.layer3 = pre_trained.layer3
        self.layer4 = pre_trained.layer4
        self.process = pre_trained.avgpool

    def forward(self, x1):

        base1 = self.conv1(x1)
        base1 = self.bn1(base1)
        base1 = self.relu(base1)
        base1 = self.maxpool(base1)
        l1 = self.layer1(base1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        process = self.process(l4).view(l4.size(0), -1)
        
        return l1, l2, l3, process


class Bottleneck(nn.Module):
    def __init__(self, pre_trained):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Linear(pre_trained.fc.in_features, 256)

    def forward(self, process):
        bottle = self.bottleneck(process)
        return bottle


class Classifier(nn.Module):
    def __init__(self, classes):
        super(Classifier, self).__init__()

        self.num_class = classes
        self.classifier = nn.Linear(256, self.num_class)


    def forward(self, bottle):

        class_pred = self.classifier(bottle)
        
        return class_pred