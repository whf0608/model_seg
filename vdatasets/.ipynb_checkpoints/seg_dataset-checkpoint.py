from torch.utils.data import Dataset
from glob import glob
import cv2
import numpy as np
import os.path as osp
from pathlib import Path
import json
import torch


class SegDataset(Dataset):
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
                 augmentations=None,
                 is_transform=False,
                 data_root=None,
                 img_size=(640, 640),
                 n_classes=3,
                 split=None,
                 use_sub=[''],
                 index='',
                 limt_num=None,
                 used_datainfo=True,
                 use_ori_size=True,
                 **arg
                 ):
        
        self.transform = augmentations
        self.is_transform = is_transform

        self.data_root = data_root
        self.img_size = img_size
        self.n_classes = n_classes

        self.use_sub = use_sub
        self.limt_num = limt_num
        self.use_ori_size = use_ori_size
        
        if self.data_root is not None:
            if type(self.data_root)==list or ('*' in self.data_root):
                self.img_infos = self.multi_datasets_deal(self.data_root,img_dirs,index,used_datainfo=used_datainfo)
            else:
                for img_dir in img_dirs:
                    self.img_dir = osp.join(self.data_root, img_dir)
                    self.img_infos = Dataset_infos(img_dir=self.img_dir, index=index,used_datainfo=used_datainfo).img_infos
        else:
            print('erro data_root',data_root)
            exit(0)
        print("----------------init dataset---------------")
        print('loading dataset:', self.data_root)
        print('use subdir:', self.use_sub)
        if self.limt_num is None:
            print('dataset size:', self.__len__())
        else:
            print('dataset_limt_num:',self.limt_num)
        
        print('data_info:')
        for k in self.img_infos['sub_imgs'].keys():
            print(k+' num:',self.img_infos['sub_imgs'][k][1])
        print("------------------------------------------")
        
    def get_image(self, img_info):
        imgs = {}
        w0,h0 = None,None
        for sub_dir in self.use_sub:
            if len(img_info[sub_dir]) > 1:
                if Path(img_info[sub_dir]).exists():
                    # print(img_info[sub_dir])
                    img = cv2.imread(img_info[sub_dir],-1)
                    if len(img.shape)==2:
                        img = np.expand_dims(img,2)
                        img = img.repeat(3,2)
                    # print(img_info[sub_dir],img.shape)
                    if self.use_ori_size:
                        # img = img[:self.img_size[0],:self.img_size[1]]
                        w,h,c = img.shape
                        
                        if w0 is None:
                            w0 = np.random.randint(0,w-320)
                            h0 = np.random.randint(0,h-320)
                        
                        img = img[w0:w0+320,h0:h0+320]
                    try:
                        img = cv2.resize(img, self.img_size)
                    except:
                        print('exist but empty',img_info[sub_dir])
                        img = np.zeros((self.img_size[0],self.img_size[1], 3), np.uint8)
                else:
                    # print('not path',img_info[sub_dir])
                    img = np.zeros((self.img_size[0],self.img_size[1], 3), np.uint8)
                if img is not None:
                    imgs[sub_dir] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # print('not exists',sub_dir)
                img = np.zeros((self.img_size[0],self.img_size[1], 3), np.uint8)
                imgs[sub_dir] =img
                
        return imgs

    def prepare_img(self, idx):
        img_info = self.img_infos['imgs'][idx]

        imgs = self.get_image(img_info)
        return imgs

    def __getitem__(self, idx):
        imgs = self.prepare_img(idx)

        if self.is_transform:
            imgs = self.transform(imgs)

        images = {}
        for key in imgs.keys():
            img = imgs[key]
            img = torch.from_numpy(img.transpose(2, 0, 1))
            images[key] = img

        return images

    def __len__(self):
        
        if self.limt_num:
            return self.limt_num
        else:
            return len(self.img_infos['imgs'])
    
    def multi_datasets_deal(self,data_roots='',img_dirs=['train'],index='t2',used_datainfo=True):
        img_infos_sum={}
        img_infos_sum['sub_imgs'] = {}
        img_infos_sum['imgs'] = []
        img_infos_sum['index'] = index
        if type(data_roots)==str:
            data_roots = [data_roots]
        data_roots_=[]
        for data_root in data_roots:
                data_root_=glob(data_root)
                data_roots_.extend(data_root_)
        data_roots =data_roots_
        
        for data_root in  data_roots:
            for img_dir in img_dirs:
                img_dir = osp.join(data_root, img_dir)
                img_infos = Dataset_infos(img_dir=img_dir, index=index,used_datainfo=used_datainfo).img_infos
            for k in img_infos['sub_imgs'].keys():
                if k not in img_infos_sum['sub_imgs'].keys():
                    img_infos_sum['sub_imgs'][k]=['',img_infos['sub_imgs'][k][1]]
                else:
                    img_infos_sum['sub_imgs'][k][1]+=img_infos['sub_imgs'][k][1]
            print('loading dataset '+ img_dir+' num:',len(img_infos['imgs']))
            img_infos_sum['imgs'].extend(img_infos['imgs'])

        return img_infos_sum


class Dataset_infos:
    def __init__(self, img_dir='', save=True, index='t2',used_datainfo=True):
        self.sub_dirs = ['t2', 't1', 't1_b', 't2_b', 'mask1', 'mask1_1',
                         'mask1_2','mask2', 'mask1_b', 'mask2_b','mask1_b1',
                         'mask2_b1', 'cd_mask_1','mask2_1', 'mask2_2',
                         'cd_cd_1', 'cd_cd', 'mask2_b2', 'images', 'labels', 'image', 'label', 
                         'label_b','mask_deg_1','mask_depth_1','mask_deg_2','mask_depth_2','mask1_3','mask2_3','cd_mask_3','cd_mask']
        self.img_suffixs = ['.jpg', '.png','.mat']
  
        self.img_dir = img_dir
        self.img_infos = {}
        self.save = save

        self.datainfo_p = osp.join(self.img_dir, index+'_dataset_info.json')
        if Path(self.datainfo_p).exists() and used_datainfo:
            with open(self.datainfo_p, 'r') as f:
                self.img_infos = json.load(f)
        else:
            print('select index: '+index+'  index_n:',self.sub_dirs.index(index))
            self.get_img_infos(index=self.sub_dirs.index(index))

        if self.sub_dirs.index(index) != self.img_infos['index']:
            # self.save=False
            self.get_img_infos(index=self.sub_dirs.index(index))

    def get_img_infos(self, index=0):
        img_info = {}
        for sub_dir in self.sub_dirs:
            path = osp.join(self.img_dir, sub_dir)
            if Path(path).exists():
                imgs_fs=[]
                for img_suffix in  self.img_suffixs:
                    path0 = osp.join(path, '*' + img_suffix)
                    imgs_fs += glob(path0)
                img_info[sub_dir] = [path, len(imgs_fs)]
            else:
                img_info[sub_dir] = [path, 0]

        self.img_infos['sub_imgs'] = img_info
        self.img_infos['imgs'] = []
        self.img_infos['index'] = index

        sub_dir = self.sub_dirs[index]
        # path = osp.join(self.img_dir, sub_dir, '*' + self.img_suffixs[index])
        imgs_fs=[]
        for img_suffix in  self.img_suffixs:
            path0 = osp.join(self.img_dir,sub_dir, '*' + img_suffix)
            imgs_fs += glob(path0)
        # print('sacanning images:',  len(imgs_fs),self.img_dir)
        for img_f in imgs_fs:
            img_info = {}
            for sd in self.sub_dirs[:]:
                img_path = img_f.replace(sub_dir, sd).replace('.'+img_f.split('.')[-1], self.img_suffixs[1])
                if Path(img_path).exists():
                    img_info[sd] = img_path
                else:
                    img_info[sd] = ''
            self.img_infos['imgs'].append(img_info)

        if self.save:
            with open(osp.join(self.img_dir, sub_dir+'_dataset_info.json'), 'w') as f:
                f.write(json.dumps(self.img_infos))
