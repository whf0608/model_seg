import argparse
import os
import numpy as np
from torch.utils.data import Dataset
from .transforms import imutils
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image


def img_loader(path):
    img = np.array(Image.open(path), np.float32)
    return img

class BDsegDataset(Dataset):
    def __init__(self, dataset_path=None, data_list=None, crop_size=512, max_iters=None, type='train', data_loader=img_loader, suffix='.png',**arg):
        self.dataset_path = dataset_path
        with open( data_list, "r") as f:
            self.data_list = [data_name.strip() for data_name in f]

        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label
        
    def __transforms_(self, aug, pre_img, post_img, label):
        pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label
        
    def __getitem__(self, index):
        # try:
            pre_path = os.path.join(self.dataset_path,  self.data_list[index]  +self.suffix)
            label_path = pre_path.replace('image','label') 
            # print(pre_path)
            pre_img = self.loader(pre_path)
            if len(pre_img.shape)==2:
                pre_img = np.stack((pre_img,)*3, axis=-1)
            if pre_img.shape[-1]==4:
                pre_img =pre_img[:,:,:3]
            clf_label = self.loader(label_path)
            
            if len(clf_label.shape)==3:
                clf_label = clf_label[:,:,0]
            
            if 'train' in self.data_pro_type:
                pre_img, post_img, clf_label = self.__transforms(True, pre_img, pre_img, clf_label)
            elif 'train_test' in self.data_pro_type:
                pre_img, post_img, clf_label = self.__transforms_(False, pre_img, pre_img, clf_label)
                clf_label = np.asarray(clf_label)
            else:
                pre_img, post_img, clf_label = self.__transforms(False, pre_img, pre_img, clf_label)
                clf_label = np.asarray(clf_label)
                
            loc_label = clf_label.copy()
            loc_label[loc_label >1] = 1
            # loc_label[loc_label == 3] = 1
            clf_label[clf_label>3]=3
            data_idx = self.data_list[index]
            # except:
            #     pre_img, pre_img, loc_label, clf_label, data_idx = self.__getitem__(index+1)
            return pre_img, pre_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)
