import glob
import os
import os.path as osp
from collections import OrderedDict
from functools import reduce
from glob import glob
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pathlib import Path
import torch


class DisasterDataset(Dataset):
    """Custom datasets for change detection. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── my_dataset
        │   │   ├── train
        │   │   │   ├── t1_dir
        │   │   │   ├── t2_dir
        │   │   │   ├── t1_b_dir
        │   │   │   ├── t2_b_dir
        │   │   │   ├── mask1_dir
        │   │   │   ├── mask2_dir
        │   │   │   ├── mask1_b_dir
        │   │   │   ├── mask2_b_dir

    """

    def __init__(self,
                 img_dirs=['train'],
                 sub_dirs=['t1', 't2', 't1_b', 't2_b', 'mask1', 'mask2', 'mask1_b', 'mask2_b','cd_mask_1','mask2_1'],
                 ann_dir=None,
                 img_suffixs=['.png', '.png', '.png', '.png', '.png', '.png', '.png', '.png', '.png', '.png',],
                 transform=None,
                 is_transform=True,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 img_size=(640,640),
                 augmentations=None,
                 debug=False):
        self.transform = transform
        self.is_transform = is_transform
        # self.img_dir = img_dir
        self.augmentations = augmentations
        self.ann_dir = ann_dir
        self.img_suffixs = img_suffixs
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.sub_dirs = sub_dirs
        self.size = img_size
        
        if len(self.size)>1:
            self.size = self.size[0]
        self.debug = debug
        self.n_classes=4
        self.img_infos = {}

        # join paths if data_root is specified
        self.img_dirs=[]
        if self.data_root is not None:
            for img_dir in img_dirs:
                img_dir = osp.join(self.data_root, img_dir)
                self.img_dirs.append(img_dir)
                self.img_dir = img_dir

                self.get_img_infos()
        # transform/augment data
        if self.transform is None:
            self.transform = self.get_default_transform() if not self.test_mode \
                else self.get_test_transform()
            self.labels_transform = self.get_labels_transform()

        # debug, visualize augmentations
        if self.debug:
            self.transform = A.Compose([t for t in self.transform if not isinstance(t, (A.Normalize, ToTensorV2,
                                                                                        ToTensorTest))])

    def get_img_infos(self):

        img_info = {}
        for sub_dir, img_suffix in zip(self.sub_dirs, self.img_suffixs):
            path = osp.join(self.img_dir, sub_dir)
            # print(path)
            if Path(path).exists():
                path0 = osp.join(path, '*' + img_suffix)
                # print('---------',path0)
                imgs_fs = glob(path0)
                img_info[sub_dir] = [path, len(imgs_fs)]
            else:
                img_info[sub_dir] = ''

        self.img_infos['sub_imgs'] = img_info
        self.img_infos['imgs'] = []

        sub_dir = self.sub_dirs[0]
        path = osp.join(self.img_dir, sub_dir, '*' + self.img_suffixs[0])

        imgs_fs = glob(path)

        for img_f in imgs_fs:
            img_info = {}
            img_info[sub_dir] = img_f
            for sd, img_suffix in zip(self.sub_dirs[1:], self.img_suffixs[1:]):
                img_path = img_f.replace(sub_dir, sd).replace(self.img_suffixs[0], img_suffix)
                if Path(img_path).exists():
                    img_info[sd] = img_path
             

            self.img_infos['imgs'].append(img_info)

    def get_default_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.Resize(self.size, self.size),
            # A.Normalize(),
            # ToTensorV2()
        ])
        return default_transform
    
    def get_labels_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.Resize(self.size, self.size),
            # ToTensorV2()
        ])
        return default_transform

    def get_test_transform(self):
        """Set the test transformation."""
        pass

    def get_image(self, img_info):
        imgs = []
        for sub_dir in self.sub_dirs:
            if sub_dir not in  img_info.keys():
                img=None
            else:
                img = cv2.imread(img_info[sub_dir])
            if img is not None:
                img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                imgs.append(np.zeros(imgs[0].shape,np.uint8))
        imgs = np.array(imgs)
        return imgs

    def prepare_img(self, idx):
        img_info = self.img_infos['imgs'][idx]
        
        imgs = self.get_image(img_info)
        return imgs

    def format_results(self, results, **kwargs):
        """Place holder to format result to datasets specific output."""
        pass

    def __getitem__(self, idx):
        imgs= self.prepare_img(idx)
        
        if self.augmentations is not None:
            imgs = self.augmentations(imgs)

        if self.is_transform:
            # print(img[0].shape,lbl[0].shape)
            images={}
            for img, sub_dir in  zip(imgs[:2],self.sub_dirs[:2]):
                
                img = self.transform(image = img)['image']
     
                img = torch.from_numpy(img.transpose(2, 0, 1))

                images[sub_dir] = img/255

            for img, sub_dir in  zip(imgs[2:],self.sub_dirs[2:]):
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = self.labels_transform(image=img[:,:,0])['image']
                
                # img[img<10]=0
                img=img/255.0
                # img[img > 0] = 1.0
                img = torch.from_numpy(img).unsqueeze(0)
                images[sub_dir] = img
        
        return images

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos['imgs'])