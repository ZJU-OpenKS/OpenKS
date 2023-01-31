# IC-Conv 

This repository is an official implementation of the paper [Inception Convolution with Efficient Dilation Search](https://arxiv.org/pdf/2012.13587.pdf).

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
```

### For 2d Human Pose Estimation using MMPose

[MMPose](https://github.com/open-mmlab/mmpose) users can use IC-Conv in the following ways. 

1. Copying the config files to the config path of mmpose, such as

```bash
cp human_pose/config/ic_res50_k13_coco_640x640.py your_mmpose_path/mmpose/configs/bottom_up/resnet/coco/ic_res50_k13_coco_640x640.py
```

2. Copying the inception conv files to the model path of mmpose,

```bash
cp human_pose/model/ic_conv2d.py your_mmpose_path/mmpose/mmpose/models/backbones/ic_conv2d.py
cp human_pose/model/ic_resnet.py your_mmpose_path/mmpose/mmpose/models/backbones/ic_resnet.py
```

3. Running it directly like [this](https://github.com/open-mmlab/mmpose/blob/master/docs/getting_started.md).

## Model Zoo

We provided the pre-trained weights of IC-ResNet-50, IC-ResNet-101and IC-ResNeXt-101 (32x4d) on ImageNet and the weights trained on specific tasks. 

For users with limited computing power, you can directly reuse our provided IC-Conv and ImageNet pre-training weights for detection, segmentation, and 2d human pose estimation tasks on other datasets. 

**Attentions**: The links in the tables below are relative paths. Therefore, you should clone the repository and download [checkpoints](https://drive.google.com/file/d/1Dx3q_4TjYsAuw7_egKIOG1WdqvMi-u2k/view?usp=sharing). 

#### Object Detection

|     Detector     | Backbone |  Lr  |  AP  |                   dilation_pattern                    |                          checkpoint                          |
| :--------------: | :------: | :--: | :--: | :---------------------------------------------------: | :----------------------------------------------------------: |
| Faster-RCNN-FPN  |  IC-R50  |  1x  | 38.9 | [pattern](pattern_zoo/detection/ic_resnet50_k9.json)  | [ckpt](ckpt/detection/faster-rcnn-ic-r50/ckpt_e14.pth)/[imagenet_retrain_ckpt](ckpt/detection/r50_imagenet_retrain/ckpt.pth.tar) |
| Faster-RCNN-FPN  | IC-R101  |  1x  | 41.9 | [pattern](pattern_zoo/detection/ic_resnet101_k9.json) | [ckpt](ckpt/detection/faster-rcnn-ic-r101/ckpt_e14.pth)/[imagenet_retrain_ckpt](ckpt/detection/r101_imagenet_retrain/ckpt.pth.tar) |
| Cascade-RCNN-FPN |  IC-R50  |  1x  | 42.4 | [pattern](pattern_zoo/detection/ic_resnet50_k9.json)  | [ckpt](ckpt/detection/cascade-rcnn-ic-r50/ckpt_e14.pth)/[imagenet_retrain_ckpt](ckpt/detection/r50_imagenet_retrain/ckpt.pth.tar) |
| Cascade-RCNN-FPN | IC-R101  |  1x  | 45.0 | [pattern](pattern_zoo/detection/ic_resnet101_k9json)  | [ckpt](ckpt/detection/cascade-rcnn-ic-r101/ckpt_e14.pth)/[imagenet_retrain_ckpt](ckpt/detection/r101_imagenet_retrain/ckpt.pth.tar) |

#### Instance Segmentation

|     Detector     | Backbone |  Lr  | box AP | mask AP |                     dilation_pattern                     |                          checkpoint                          |
| :--------------: | :------: | :--: | :----: | :-----: | :------------------------------------------------------: | :----------------------------------------------------------: |
|  Mask-RCNN-FPN   |  IC-R50  |  1x  |  40.0  |  35.9   | [pattern](pattern_zoo/segmentation/ic_resnet50_k9.json)  | [ckpt](ckpt/segmentation/faster-rcnn-ic-r50/ckpt_e14.pth)/[imagenet_retrain_ckpt](ckpt/segmentation/r50_imagenet_retrain/ckpt.pth.tar) |
|  Mask-RCNN-FPN   | IC-R101  |  1x  |  42.6  |  37.9   | [pattern](pattern_zoo/segmentation/ic_resnet101_k9.json) | [ckpt](ckpt/segmentation/faster-rcnn-ic-r101/ckpt_e14.pth)/[imagenet_retrain_ckpt](ckpt/segmentation/r101_imagenet_retrain/ckpt.pth.tar) |
| Cascade-RCNN-FPN |  IC-R50  |  1x  |  43.4  |  36.8   | [pattern](pattern_zoo/segmentation/ic_resnet50_k9.json)  | [ckpt](ckpt/segmentation/cascade-rcnn-ic-r50/ckpt_e14.pth)/[imagenet_retrain_ckpt](ckpt/segmentation/r50_imagenet_retrain/ckpt.pth.tar) |
| Cascade-RCNN-FPN | IC-R101  |  1x  |  45.7  |  38.7   | [pattern](pattern_zoo/segmentation/ic_resnet101_k9.json) | [ckpt](ckpt/segmentation/cascade-rcnn-ic-r101/ckpt_e14.pth)/[imagenet_retrain_ckpt](ckpt/segmentation/segmentation/r101_imagenet_retrain/ckpt.pth.tar) |

#### 2d Human Pose Estimation

We adjust the learning rate of resnet backbone in MMPose and get better baseline results. Please see the specific config files in `config/human_pose/`.

##### Results on COCO val2017 without multi-scale test

|                           Backbone                           | Input Size |    AP    |                    dilation_pattern                     |                          checkpoint                          |
| :----------------------------------------------------------: | :--------: | :------: | :-----------------------------------------------------: | :----------------------------------------------------------: |
| [R50(mmpose)](https://github.com/open-mmlab/mmpose/tree/master/configs/bottom_up/resnet) |  640x640   |   47.9   |                            ~                            |                              ~                               |
|        [R50](human_pose/config/res50_coco_640x640.py)        |  640x640   |   51.0   |                            ~                            |                              ~                               |
|   [IC-R50](human_pose/config/ic_res50_k13_coco_640x640.py)   |  640x640   | **62.2** | [pattern](pattern_zoo/human_pose/ic_resnet50_k13.json)  | [ckpt](ckpt/human_pose/ic_res50_k13_coco_640x640_lr0.001/ckpt.pth)/[imagenet_retrain_ckpt](ckpt/human_pose/ic_res50_k13_imagenet_retrain/ckpt.pth) |
|       [R101](human_pose/config/res101_coco_640x640.py)       |  640x640   |   55.5   |                            ~                            |                              ~                               |
|  [IC-R101](human_pose/config/ic_res101_k13_coco_640x640.py)  |  640x640   | **63.3** | [pattern](pattern_zoo/human_pose/ic_resnet101_k13.json) | [ckpt](ckpt/human_pose/ic_res101_k13_coco_640x640_lr0.0005/ckpt.pth)/[imagenet_retrain_ckpt](ckpt/human_pose/ic_res101_k13_imagenet_retrain/ckpt.pth) |

##### Results on COCO val2017 with multi-scale test. 3 default scales ([2, 1, 0.5]) are used

|                           Backbone                           | Input Size |    AP    |
| :----------------------------------------------------------: | :--------: | :------: |
| [R50(mmpose)](https://github.com/open-mmlab/mmpose/tree/master/configs/bottom_up/resnet) |  640x640   |   52.5   |
|        [R50](human_pose/config/res50_coco_640x640.py)        |  640x640   |   55.8   |
|   [IC-R50](human_pose/config/ic_res50_k13_coco_640x640.py)   |  640x640   | **65.8** |
|       [R101](human_pose/config/res101_coco_640x640.py)       |  640x640   |   60.2   |
|  [IC-R101](human_pose/config/ic_res101_k13_coco_640x640.py)  |  640x640   | **68.5** |

## Acknowledgement

The human pose estimation experiments are built upon [MMPose](https://github.com/open-mmlab/mmpose).

## Citation

If our paper helps your research, please cite it in your publications:

```
@article{liu2020inception,
 title={Inception Convolution with Efficient Dilation Search},
 author={Liu, Jie and Li, Chuming and Liang, Feng and Lin, Chen and Sun, Ming and Yan, Junjie and Ouyang, Wanli and Xu, Dong},
 journal={arXiv preprint arXiv:2012.13587},
 year={2020}
}
```