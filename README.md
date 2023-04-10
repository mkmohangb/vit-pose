## ViT Pose

ViTPose is a 2D Human Pose Estimation model based on the Vision transformer architecture. The official repo is [1]. Goal here is to create a version of VIT Pose without the framework code(mmpose/mmcv) for easy understanding/hacking. Only inference is supported.

### 1. Execution
Download the model weights from [1] - VitPose-B - single task training - classic decoder.

```pip install -r requirements.txt```

```python main.py```

### 2. Details

image => preprocess => model => postprocess => keypoints

#### a. Preprocess 
-  calculate center/scale, do affine_transform
   - center - x + w/2, y + h/2
   - adjust (w,h) based on the image aspect ratio. scale - ((w,h)/200) * padding (200 is used to normalize the scale)
   - Affine transform  
-  convert to tensor & /255
-  normalize the tensor

#### b. Model
  - Backbone - Patch Embedding + Encoder blocks
  - Decoder or Head - heatmaps(64 x 48) corresponding to the number of key points
  
<img width="576" alt="Screenshot 2023-04-10 at 12 23 44 PM" src="https://user-images.githubusercontent.com/2610866/230844903-fd0d0ccb-19ba-4cc3-b63b-a7cf9c22e97e.png">


#### c. Postprocess
   - Heatmaps to keypoints
     - For each heatmap, calculate the location of max value
     - add +/-0.25 shift to the locations for higher accuracy
     - scale = scale * 200. Transform back to the image dimensions -> location * scale + center - 0.5 * scale

#### d. Adapted from:
 1. [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) 
 2. [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch)
