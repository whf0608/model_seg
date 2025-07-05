import sys
# sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/valite_dataset_sys_v2/lib/utils")
# sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/valite_dataset_sys_v2/lib/lib")
from glob import glob 
import cv2
import matplotlib.pyplot as plt
import  numpy as  np
import os.path as osp
from pathlib import Path
from torch.utils.data import Dataset
import torch
import json
from pathlib import Path
import xml.etree.ElementTree as ET
# from labeling_save import svg2mask
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
import sklearn

def get_data(data_root):
    data_fs = glob(data_root+'/*')
    data_set = []
    for data_f in data_fs:
        with open(data_f,'r') as f:
            lines = f.readlines()
            for line in lines:
                data_set.append(line.replace('\n',''))
    data_set = list(set(data_set))
    return data_set

def get_image_label(svg_file,use_label=[],color_map=[]):
    json_file = svg_file.replace('.svg','.json')
    img_f = svg_file.replace('.svg','.png')
    # print(img_f)
    
    img = cv2.imread(img_f,-1)
    mask = np.zeros(img.shape,np.uint8)
    mask_b = np.zeros(img.shape,np.uint8)
    # print("json file:", svg_file)
    masks_dir = svg2mask(svg_file)
    with open(json_file) as f:
        labels = json.load(f)
        class_c =  list(set(labels.values()))
    if len(use_label)==0 or use_label is None:
        use_label = class_c
            
    for k in masks_dir.keys():
        # if k in labels.keys(): print(labels[k])
        # print(k)
        if k in labels.keys() and labels[k] in use_label : 
            cur_color = color_map[use_label.index(labels[k])] 
            cur_color = tuple([int(x) for x in cur_color])
            mask = cv2.drawContours(mask,[masks_dir[k]],-1,cur_color, thickness=-1)
            c_v = use_label.index(labels[k])+1
            mask_b = cv2.drawContours(mask_b,[masks_dir[k]],-1,(int(c_v),int(c_v),int(c_v)), thickness=-1)
    return img, mask_b,mask 
def svg2mask(svg_f):
    tree = ET.parse(svg_f)
    root = tree.getroot()
    # 遍历SVG元素
    pointss0 = {}
    for child in root:
        # 获取元素标签名和属性
        tag = child.tag
        attrs = child.attrib
        # 打印元素信息
        # print(f"Element: {tag}")
        if 'class' in attrs.keys():
            cls = attrs['class']
            value= attrs['points']
            # for key, value in attrs.items():
                # print(f"Attribute: {key}={value}")
                # if key =='points':
            points0=[]
            points  = value.split(' ')
            for point in points:
                x,y = point.split(',')
                points0.append([int(x),int(y)])
            points0=[]
            points  = value.split(' ')
            for point in points:
                x,y = point.split(',')
                points0.append([int(x),int(y)])
            points0 = np.array(points0)
            pointss0[cls]=points0
    return pointss0
    
class MapDataset(Dataset):
    def __init__(self,
                 img_dirs=['train'],
                 augmentations=None,
                 data_root=None,
                 img_size=(640, 640),
                 split=None,
                 use_cls=[''],
                 index='',
                 limt_num=None,
                 use_sub=[],
                 **arg
                 ):
       
        self.transform = augmentations
        self.is_transform = not not augmentations

        self.data_root = data_root
        self.img_size = img_size
        self.use_cls = use_cls
        self.use_sub=use_sub
        self.n_classes = len(self.use_cls)
        self.limt_num = limt_num
        
        if Path(self.data_root).exists():
            self.imgs = get_data(self.data_root)
            
        print('loading data num:', len(self.imgs))
        print('use cls:', self.use_cls)
        
        if self.limt_num is None:
            print('dataset size:', self.__len__())
        else:
            print('dataset_limt_num:',self.limt_num)
        
        self.color_map = np.array([[255, 255, 255],
                               [0,  0,  254], [12,  255,   7], [255, 89,    1], [5,   255, 133],
                               [255,   2, 251], [89,  1,  255],[255, 254, 137] , [3,   100, 220],
                               [160, 78, 158], [172, 175,  84],  [101, 173, 255], [60,   91, 112],
                               [104, 192,  63], [139, 69,  46], [119, 255, 172], [254, 255,   3],
                               [0,  0,  254], [12,  255,   7], [255, 89,    1], [5,   255, 133],
                               [255,   2, 251], [89,  1,  255],[255, 254, 137] , [3,   100, 220],
                               [172, 175,  84], [160, 78, 158], [101, 173, 255], [60,   91, 112],
                               [104, 192,  63], [139, 69,  46], [119, 255, 172], [254, 255,   3]],np.uint8)
            
    def prepare_img(self, idx):
        img_p = self.imgs[idx]
        img,mask,mask_rgb  = get_image_label(img_p,use_label=self.use_cls,color_map=self.color_map)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, self.img_size)
        mask= cv2.resize(mask, self.img_size)
        mask_rgb= cv2.resize(mask_rgb, self.img_size)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.transpose(2, 0, 1))
        mask_rgb = torch.from_numpy(mask_rgb.transpose(2, 0, 1))
        return img,mask,mask_rgb,img_p

    def __getitem__(self, idx):
        img,mask,mask_rgb,img_p = self.prepare_img(idx)
        result = {'image':img,'mask':mask,'mask_rgb':mask_rgb,'path':img_p}
        
        for sub_f in self.use_sub:
            img = cv2.imread(img_p.replace('sum.svg',sub_f+'.png'))
            img = cv2.resize(img, self.img_size)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            result[sub_f] = img
        
        return result

    def __len__(self):
        
        if self.limt_num:
            return self.limt_num
        else:
            return len(self.imgs)
