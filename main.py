import cv2
from importlib import import_module
from simple_head import TopdownHeatmapSimpleHead
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from vit import ViT
import numpy as np
import matplotlib.pyplot as plt

image_file = "000000000785.jpg"
bbox = np.array([280.79,  44.73, 218.7 , 346.68]) #output of detector
image_file = "000000196141.jpg"
bbox = np.array([247.76, 74.23, 169.67, 300.78]) #output of detector

img_width = 192
img_height = 256
image_size = np.array([img_width, img_height], dtype=np.float32)

def getBboxCenterScale(bbox, image_size):
    padding = 1.25
    pixel_std = 200.0
    aspect_ratio = image_size[0] / image_size[1]

    x, y, w, h = bbox[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w, h], dtype=np.float32) / pixel_std
    scale = scale * padding

    return center, scale


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.
    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian
    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.
    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.
    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)
    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.
    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

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
