import argparse
import os

import imageio
import numpy as np
from torch.utils.data import Dataset
# import cv2 
from .transforms import imutils as imutils
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image


def img_loader(path):
    img = np.array(Image.open(path), np.float32)
    return img

class MultimodalDamageAssessmentDatsetxbd(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, suffix='.png'):
        self.dataset_path = dataset_path
        self.data_list = data_list
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

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path,  self.data_list[index] +   self.suffix)
        post_path = pre_path.replace('/t1/','/t2/')
        label_path = pre_path.replace('t1','mask1_2')
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)[:,:,0:3]   
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        # post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        # print("===================",post_img.shape)

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)


class DamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, suffix='.png'):
        self.dataset_path = dataset_path
        self.data_list = data_list
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
        
class MultimodalDamageAssessmentDatsetWuhan(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, suffix='.png'):
        self.dataset_path = dataset_path
        self.data_list = data_list
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

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path,  self.data_list[index]  +self.suffix)
        post_path = pre_path
        label_path = pre_path.replace('image','mask') 
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)[:,:,0:3] 
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        # post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)
        
class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
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

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster'  + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', self.data_list[index] + '_building_damage'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)

class MultimodalDamageAssessmentDatset_Inference(Dataset):
    def __init__(self, dataset_path, data_list, data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.suffix = suffix

    def __transforms(self, pre_img, post_img):
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,)*3, axis=-1) 
        
        pre_img, post_img = self.__transforms(pre_img, post_img)
    
        data_idx = self.data_list[index]
        return pre_img, post_img, data_idx
    
    def __len__(self):
        return len(self.data_list)



class MultimodalDamageAssessmentDatsetxbd_Inference(Dataset):
    def __init__(self, dataset_path, data_list, data_loader=img_loader, suffix='.png'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.suffix = suffix

    def __transforms(self, pre_img, post_img):
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path,  self.data_list[index] +   self.suffix)
        post_path = os.path.join(self.dataset_path, self.data_list[index] +   self.suffix)
        # label_path = os.path.join(self.dataset_path,self.data_list[index].replace('t1','mask1_2') +   self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)[:,:,0:3] 
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        # post_img = np.stack((post_img,)*3, axis=-1) 
        
        pre_img, post_img = self.__transforms(pre_img, post_img)
    
        data_idx = self.data_list[index]
        return pre_img, post_img, data_idx
    
    def __len__(self):
        return len(self.data_list)