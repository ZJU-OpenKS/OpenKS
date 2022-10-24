from model.ic_resnet import ic_resnet50
import torch

pattern_path = 'pattern_zoo/detection/ic_resnet50_k9.json'
load_path = 'ckpt/detection/r50_imagenet_retrain/ckpt.pth.tar'

net = ic_resnet50(pattern_path=pattern_path)
state = torch.load(load_path, 'cpu')
net.load_state_dict(state, strict=False)
state_keys = set(state.keys())
model_keys = set(net.state_dict().keys())
missing_keys = model_keys - state_keys
print(missing_keys)
inputs = torch.rand(1, 3, 224, 224)
outputs = net.forward(inputs)
print(outputs.shape)
