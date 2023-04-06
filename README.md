### ViT Pose

ViTPose is a 2D Human Pose Estimation model based on the Vision transformer architecture. The official repo is [1]. Goal here is to create a version of VIT Pose without the framework code(mmpose/mmcv) for easy understanding/hacking. Only inference is supported.

```pip install -r requirements.txt```

Download the model weights from [1] - VitPose-B - single task training - classic decoder.

image => preprocess => model => postprocess => keypoints

```python main.py```

Adapted from:
 1. [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) 
 2. [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch)
