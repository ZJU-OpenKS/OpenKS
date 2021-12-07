# IC-Conv 

This repository is an implementation using paddle2paddle of the paper [Inception Convolution with Efficient Dilation Search](https://arxiv.org/pdf/2012.13587.pdf).

## Getting Started

**Download** ImageNet pre-trained [checkpoints](https://drive.google.com/file/d/11diajagP3BKekV4iztnkm_B2iN8opGUo/view?usp=sharing).

Extract the file to get the following directory tree

```
|-- README.md
|-- ckpt
|   |-- detection
|   |-- human_pose
|   |-- segmentation
|-- config
|-- model
|-- pattern_zoo
```

### Easy Use

Users can quickly use IC-Conv in the following simple ways. 

```python
from model.ic_resnet import ic_resnet50
import paddle

pattern_path = 'pattern_zoo/detection/ic_resnet50_k9.json'
load_path = 'ckpt/detection/r50_imagenet_retrain/ckpt.pth.tar'

net = ic_resnet50(pattern_path=pattern_path)
state = paddle.load(load_path)
net.set_dict(state)
state_keys = set(state.keys())
model_keys = set(net.state_dict().keys())
missing_keys = model_keys - state_keys
print(missing_keys)
inputs = paddle.rand(1, 3, 224, 224)
outputs = net.forward(inputs)
print(outputs.shape)
```