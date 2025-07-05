import os
import sys
import cv2
import torch
import argparse
import subprocess
import numpy as np
from glob import glob
from PIL import Image
import os.path as osp
import scipy.io as io
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt

def sobel_kernel(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    k = np.zeros(shape)
    p = [
        (j, i)
        for j in range(shape[0])
        for i in range(shape[1])
        if not (i == (shape[1] - 1) / 2.0 and j == (shape[0] - 1) / 2.0)
    ]

    for j, i in p:
        j_ = int(j - (shape[0] - 1) / 2.0)
        i_ = int(i - (shape[1] - 1) / 2.0)
        k[j, i] = (i_ if axis == 0 else j_) / float(i_ * i_ + j_ * j_)
    return torch.from_numpy(k).unsqueeze(0)

def get_depthmap(labelmap, metric = 'euc',ksize = 5):
    label_list = labelmap.max()
    
    if len(labelmap.shape)>2: labelmap =  labelmap[:,:,0]
    sobel_x, sobel_y = (sobel_kernel((ksize, ksize), i) for i in (0, 1))
    sobel_ker = torch.cat([sobel_y, sobel_x], dim=0).view(2, 1, ksize, ksize).float()
    
    depth_map = np.zeros(labelmap.shape, dtype=np.float32)
    dir_map = np.zeros((*labelmap.shape, 2), dtype=np.float32)

    for id in range(1, label_list + 1):
        labelmap_i = labelmap.copy()
        labelmap_i[labelmap_i != id] = 0
        labelmap_i[labelmap_i == id] = 1
        
        if metric == 'euc':
            depth_i = distance_transform_edt(labelmap_i)
        elif metric == 'taxicab':
            depth_i = distance_transform_cdt(labelmap_i, metric='taxicab')

        depth_map += depth_i

        dir_i_before = dir_i = np.zeros_like(dir_map)
        dir_i = torch.nn.functional.conv2d(torch.from_numpy(depth_i).float().view(1, 1, *depth_i.shape), sobel_ker,
                                           padding=ksize // 2).squeeze().permute(1, 2, 0).numpy()

        # The following line is necessary
        dir_i[(labelmap_i == 0), :] = 0

        dir_map += dir_i
    
    return depth_map