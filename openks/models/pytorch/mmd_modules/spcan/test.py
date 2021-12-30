from model.classifier import FeatureExtractor, Classifier, Bottleneck
from model.discriminator import BottleDiscriminator, LayersDiscriminator
import torch

def train():
    x = torch.randn(5, 256)
    y = torch.tensor([1, 0, 2, 1, 0])
    pred = model(x)
    loss = cri(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()


model = Classifier(3)
print(model.classifier.weight[1][1])
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
cri = torch.nn.CrossEntropyLoss()
train()
print(model.classifier.weight[1][1])