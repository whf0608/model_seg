from os.path import splitext
from os import listdir
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms.functional as transF
from imgaug import augmenters as iaa
import scipy.io as io
import cv2
from pathlib import Path
from .transforms.offset_helper import DTOffsetHelper

mean_std_dict = {
    'WHU_BUILDING': [0.3, [0.43782742, 0.44557303, 0.41160695], [0.19686149, 0.18481555, 0.19296625], '.png'], \
    'Inriaall': [0.2, [0.31815762, 0.32456695, 0.29096074], [0.18410079, 0.17732723, 0.18069517], '.png'],
    'Wuhan1': [0.3, [0.43782742, 0.44557303, 0.41160695], [0.19686149, 0.18481555, 0.19296625], '.png'],
    'Wuhan': [0.3, [0.43782742, 0.44557303, 0.41160695], [0.19686149, 0.18481555, 0.19296625], '.png'],
    'imgs8': [0.3, [0.43782742, 0.44557303, 0.41160695], [0.19686149, 0.18481555, 0.19296625], '.png'],
    'Mass': [0.9, [0.31815762, 0.32456695, 0.29096074], [0.18410079, 0.17732723, 0.18069517], '.png'],
    'xBDearthquake': [0.3, [0.43782742, 0.44557303, 0.41160695], [0.19686149, 0.18481555, 0.19296625], '.png'],
    'harvey': [0.3, [0.43782742, 0.44557303, 0.41160695], [0.19686149, 0.18481555, 0.19296625], '.png']
    }


class SegfixDataset(Dataset):
    def __init__(self,data_root='', imgs_dir='', masks_dir='', img_size=[],training=False, get_edge=False,index='t1',use_sub=[],**arg):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = 1
        self.training = training
        self.get_edge = get_edge
        self.img_size=img_size
        self.index_file = index
        self.use_sub = use_sub
        data_name = imgs_dir.split('/')[-4]
        print('SegfixDataset loading',data_name)
        if data_name not in  mean_std_dict.keys():
            data_name = 'WHU_BUILDING'
        self.res, self.mean, self.std, self.shuffix = mean_std_dict[data_name]
        self.ids = [splitext(file)[0] for file in listdir(masks_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        img = Image.open(self.imgs_dir + self.ids[0] + self.shuffix)
        self.transform = iaa.Sequential([
            iaa.Rot90([0, 1, 2, 3]),
            iaa.VerticalFlip(p=0.5),
            iaa.HorizontalFlip(p=0.5),

        ])

    def __len__(self):
        return len(self.ids)

    def _load_mat(self, filename):
        return io.loadmat(filename)

    def _load_maps(self, filename, ):
        dct = self._load_mat(filename)
        distance_map = dct['depth'].astype(np.int32)
        dir_deg = dct['dir_deg'].astype(np.float)  # in [0, 360 / deg_reduce]
        deg_reduce = dct['deg_reduce'][0][0]

        dir_deg = deg_reduce * dir_deg - 180  # in [-180, 180]
        return distance_map, dir_deg

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir + idx + self.shuffix
        img_file = self.imgs_dir + idx + self.shuffix
        # mask = Image.open(mask_file)
        # img = Image.open(img_file)

        img = cv2.imread(img_file)
        img = cv2.resize(img, (512, 512))
        if Path(mask_file).exists():
            mask = cv2.imread(mask_file)
            # print(mask.shape)
            # print(img.shape)
            mask = cv2.resize(mask, (512, 512))

            distance_map, angle_map = self._load_maps(self.imgs_dir.replace(self.index_file, self.use_sub[-1]) + idx)

            distance_map = np.array(distance_map, np.float)
            angle_map = np.array(angle_map, np.float)
            # print(distance_map.shape)
            distance_map = cv2.resize(distance_map, (512, 512))
            angle_map = cv2.resize(angle_map, (512, 512))

        else:
            print('no', img_file)
            # mask = np.zeros((512,512,3))
            # angle_map= np.zeros((512,512))
            # distance_map= np.zeros((512,512))

        _, direction_map = DTOffsetHelper.align_angle(torch.tensor(angle_map), num_classes=8,
                                                      return_tensor=True)
        if self.training:
            # print(img.shape,mask.shape,direction_map.shape,distance_map.shape)
            img, mask = self.transform(image=img, segmentation_maps=np.stack(
                (mask[np.newaxis, :, :, 0], direction_map[np.newaxis, :, :], distance_map[np.newaxis, :, :]),
                axis=-1).astype(np.int32))
            mask, direction_map, distance_map = mask[0, :, :, 0], mask[0, :, :, 1], mask[0, :, :, 2]
        img, mask = transF.to_tensor(img.copy()), (transF.to_tensor(mask.copy()) > 0).int()
        img = transF.normalize(img, self.mean, self.std)
        # print("|||||||||==========",direction_map.shape)
        return {
            'image': img.float(),
            'mask': mask.float(),
            'direction_map': direction_map,
            'distance_map': distance_map,
            'name': self.imgs_dir + idx + self.shuffix
        }

