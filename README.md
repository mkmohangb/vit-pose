### VIT Pose

Create a version of VIT Pose without the framework code which is easy to understand. 

```pip install -r requirements.txt```

Download the model weights from [1] - VitPose-B - single task training - classic decoder.

image => preprocess => model => postprocess => keypoints

```python main.py```

Adapted from:
 1. [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) 
 2. [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch)
