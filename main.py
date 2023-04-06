import cv2
from importlib import import_module
from simple_head import TopdownHeatmapSimpleHead
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from vit import ViT
import numpy as np
import matplotlib.pyplot as plt
from preprocess import getBboxCenterScale, get_affine_transform

image_file = "imgs/000000000785.jpg"
bbox = np.array([280.79,  44.73, 218.7 , 346.68]) #output of detector
image_file = "imgs/000000196141.jpg"
bbox = np.array([247.76, 74.23, 169.67, 300.78]) #output of detector

img_width = 192
img_height = 256
image_size = np.array([img_width, img_height], dtype=np.float32)


class ViTPoseModel(nn.Module):
    def __init__(self, backbone, head):
        super(ViTPoseModel, self).__init__()
        self.backbone = backbone
        self.keypoint_head = head

    def forward(self, x, img_metas):
        x = self.backbone(x)
        x = self.keypoint_head.inference_model(x)
        res = self.keypoint_head.decode(img_metas, x, 
                                        img_size=[img_width, img_height])
        return res 

mod = import_module('ViTPose_base_coco_256x192')
model = getattr(mod, 'model')
weights = "vitpose-b.pth"

head = TopdownHeatmapSimpleHead(in_channels=model['keypoint_head']['in_channels'], 
                                out_channels=model['keypoint_head']['out_channels'],
                                num_deconv_filters=model['keypoint_head']['num_deconv_filters'],
                                num_deconv_kernels=model['keypoint_head']['num_deconv_kernels'],
                                num_deconv_layers=model['keypoint_head']['num_deconv_layers'],
                                extra=model['keypoint_head']['extra'])

backbone = ViT(img_size=model['backbone']['img_size'],
                patch_size=model['backbone']['patch_size']
                ,embed_dim=model['backbone']['embed_dim'],
                depth=model['backbone']['depth'],
                num_heads=model['backbone']['num_heads'],
                ratio = model['backbone']['ratio'],
                mlp_ratio=model['backbone']['mlp_ratio'],
                qkv_bias=model['backbone']['qkv_bias'],
                drop_path_rate=model['backbone']['drop_path_rate']
                )

pose_model = ViTPoseModel(backbone, head)
pose_model.load_state_dict(torch.load(weights)['state_dict'])
pose_model.eval()

img = cv2.imread(image_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
orig_img = img.copy()
img_metas = [{"image_file": image_file}]
center, scale = getBboxCenterScale(bbox, image_size)
img_metas[0]['center'] = center
img_metas[0]['scale'] = scale
trans = get_affine_transform(center, scale, 0, [img_width, img_height])
print(img.shape)
img = cv2.warpAffine(img, 
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
tensor = torch.from_numpy(img.astype('float32')).permute(2, 0, 1).div_(255.0)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
tensor = F.normalize(tensor, mean=mean, std=std, inplace=True)
out = pose_model(tensor.unsqueeze(0), img_metas)
kps = out["preds"][0]
for kp in kps:
    cv2.circle(orig_img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)

plt.imshow(orig_img)
plt.show()
