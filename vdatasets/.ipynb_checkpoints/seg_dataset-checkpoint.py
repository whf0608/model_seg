from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import numpy as np
import os.path as osp
from pathlib import Path
import json
import torch
try:
    from .transforms import imutils
except: from transforms import imutils
    
class SegDataset(Dataset):
    """Custom datasets for change detection. An example of file structure
        │   ├── dataset
        │   │   │   ├── sub_dir
        │   │   │   ├── sub_dir
        │   │   │   ├── sub_dir
        │   │   │   ├── sub_dir
    """
    def __init__(self,
                 is_transform=False,
                 dataset_path=None,
                 crop_size=640,
                 sub_dir=[''],
                 limt_num=None,
                 data_type='train',
                 **arg
                 ):
        self.transform = None
        self.is_transform = is_transform

        self.data_root = dataset_path
        self.img_size = (crop_size,crop_size)
        self.use_sub = sub_dir
        self.limt_num = limt_num

        with open(f'{dataset_path}/{data_type}.txt', "r") as f:
            self.data_list = [data_name.strip() for data_name in f]
        if self.limt_num is not None:
            self.data_list = self.data_list[:self.limt_num]
            
        print("----------------init dataset---------------")
        print('loading dataset:', self.data_root,'use subdir:', self.use_sub, 'num: ',len(self.data_list))

    def __getitem__(self, idx):

        img_name = self.data_list[idx]
        imgs = {}
        for sub_dir in self.use_sub:
           path =f'{self.data_root}/{sub_dir}/{img_name}.png'
           img = np.array(Image.open(path), np.float32)
           # if len(img.shape)==2: img = np.stack((img ,)*3, axis=-1)
           # if self.is_transform:
           #      img = self.transform(img)
           if sub_dir not in ['mask','label','masks','labels']:
               imgs[sub_dir] = imutils.normalize_img(img).transpose(2, 0, 1)
           else: imgs[sub_dir] = img
            
        return imgs

    def __len__(self):
         return len(self.data_list )
        

    

