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
   - (x, y, w, h) - bounding box of detected person in the image that is output by an object detector (e.g. YOLO or EfficientDet)
   - center - x + w/2, y + h/2
   - adjust (w,h) based on the image aspect ratio. scale - ((w,h)/200) * padding (200 is used to normalize the scale)
   - Affine transform  
-  convert to tensor & /255
-  normalize the tensor
-  tensor shape is [(1, 3, 256, 192)]

#### b. Model
  - Backbone - Patch Embedding + Pos. Embedding + Encoder blocks
    - patch embedding implemented using a Conv2D layer with the kernel size and stride equal to the patch size(16) and the out channels equal to the embedding dimension (768). Output shape is [(1, 768, 16, 12)]. Flattened & transposed to [(1, 192, 768)]
    - Position embedding is added to the output of patch emdedding.
    - this embedding output is fed to multiple layers of encoder blocks. Output shape [(1, 192, 768)] is same as input shape.
    - output is reshaped back to [(1, 768, 16, 12)]
  - Decoder or Head - heatmaps(64 x 48) corresponding to the number of key points
    - Encoder output is fed to a decoder which consists of 2 layers of ConvTranspose2D + BN + ReLU ([(1, 256, 64, 48)]) and a final conv1d layer with (1x1) kernel and 17 out channels([(1, 17, 64, 48)]).
  
<img width="576" alt="Screenshot 2023-04-10 at 12 23 44 PM" src="https://user-images.githubusercontent.com/2610866/230844903-fd0d0ccb-19ba-4cc3-b63b-a7cf9c22e97e.png">


#### c. Postprocess
   - Heatmaps to keypoints
     - For each heatmap, calculate the location of max value
     - add +/-0.25 shift to the locations for higher accuracy
     - scale = scale * 200. Transform back to the image dimensions -> location * scale + center - 0.5 * scale

### 3. Adapted from:
 1. [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) 
 2. [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch)
