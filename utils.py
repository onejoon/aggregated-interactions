import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import Resize, InterpolationMode


def vis_top_attr(img, attr, patch_size, th=0.8, base=np.zeros(3)):
    img = np.array(img)
    img_size = img.shape[:2]
    resize = Resize(img_size, interpolation=InterpolationMode.NEAREST)

    th = np.quantile(attr.flatten(), th)
    heatmap = torch.Tensor(attr.reshape(patch_size[0],patch_size[1]))
    heatmap = resize(heatmap.unsqueeze(0)).squeeze()
    tmp_img = img.copy()
    tmp_img[heatmap<th] = base*255
    
    plt.figure(figsize=(4,4))
    plt.imshow(tmp_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def vis_bottom_attr(img, attr, patch_size, th=0.5, base=np.zeros(3)):
    img = np.array(img)
    img_size = img.shape[:2]
    resize = Resize(img_size, interpolation=InterpolationMode.NEAREST)

    th = np.quantile(attr.flatten(), th)
    heatmap = torch.Tensor(attr.reshape(patch_size[0],patch_size[1]))
    heatmap = resize(heatmap.unsqueeze(0)).squeeze()
    tmp_img = img.copy()
    tmp_img[heatmap>th] = base*255
    
    plt.figure(figsize=(4,4))
    plt.imshow(tmp_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
