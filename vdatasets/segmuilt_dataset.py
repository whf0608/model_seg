import cv2
import json
from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
import os
from  glob import glob
from osgeo import gdal
import albumentations as A  
from  .transforms.hotspot_encode import get_depthmap

class SegMuiltDataset(Dataset):
    def __init__(self,data_root='', use_sub=['images','labels'],hotspotsubf=['labels'], catsubf=[],img_size=[1024,1024],img_dirs=['train'],
                 limt_num=None,data_transform=False,**arg):
        print('init SegMuiltDataset')
        self.data_root =data_root
        self.size = img_size
        self.use_sub=  use_sub
        self.catsubf = catsubf
        self.hotspotsubf = hotspotsubf
        self.limt_num = limt_num
        self.img_data = glob(os.path.join(data_root,img_dirs[0],self.use_sub[0],'*.tif'))
        if self.limt_num is None or self.limt_num<0:
            self.limt_num = len(self.img_data)
        self.data_transform = data_transform
        
        self.transform = A.Compose([
            A.RandomCrop(width=self.size[0], height=self.size[1]),
            A.HorizontalFlip(p=0.5),  # 水平翻转  
            A.RandomRotate90(p=0.5),  # 随机旋转90度倍数  
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),  # 平移、缩放、旋转  
        ], p=1.0)  # 应用增强的概率  
        
        print('loading data num: ',len(self.img_data), os.path.join(data_root,img_dirs[0],self.use_sub[0],'*.tif'))
        print('use data transform ', data_transform)
        
    def get_image(self, img_p):
        imgs = {}
        if Path(img_p).exists():
            data  = gdal.Open(img_p)
            tmp_img = data.ReadAsArray()
            img = tmp_img.transpose(1, 2, 0)
            # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # img = cv2.flip(img,0)
            # img = cv2.resize(img,self.size)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)
            imgs[self.use_sub[0]] = img
            
        for sub_f in  self.use_sub[1:]:
            sub_p = img_p.replace(self.use_sub[0],sub_f).replace('.tif','.png')
            if Path(sub_p).exists():
                sub_img = cv2.imread(sub_p,-1)
            # else:
            #     sub_img = np.zeros([self.size[0],self.size[1],3],np.uint8)
                
            if len(sub_img.shape)==2:
                sub_img = np.expand_dims(sub_img,2)
                sub_img = sub_img.repeat(3,2)
            # sub_img = cv2.resize(sub_img,self.size)
            
            if sub_f in self.hotspotsubf:
                imgs['hotspot'+sub_f] =  get_depthmap(sub_img)
            
            imgs[sub_f] = torch.from_numpy(sub_img.transpose(2,0,1))
            if sub_f in self.catsubf:
                imgs[self.use_sub[0]] = torch.cat([imgs[self.use_sub[0]],imgs[sub_f]],0)
           
        imgs['paths'] = img_p
        return imgs

    def __getitem__(self, idx):
        img_p =  self.img_data[idx%self.limt_num%len(self.img_data)]
        imgs = self.get_image(img_p)
        # imgs['img_info'] =img_info
        if self.data_transform:
            image = imgs[self.use_sub[0]].numpy()
            mask = imgs[self.use_sub[-1]].numpy()
            augmented = self.transform(image=image.transpose(1,2,0), mask=mask.transpose(1,2,0))
            imgs[self.use_sub[0]] = torch.from_numpy(augmented['image'].transpose(2,0,1))
            imgs[self.use_sub[-1]] = torch.from_numpy(augmented['mask'].transpose(2,0,1))
        return imgs

    def __len__(self):
        if self.limt_num>len(self.img_data):
            return self.limt_num
        return len(self.img_data)


# if __name__ == "__main__":
#     disasterdataset = DisasterDataset("../model_config/dataset_config_xBDearthquake.json")

#     for img_meta  in disasterdataset:
#         print(img_meta)