import cv2
import json
from torch.utils.data import Dataset
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

    def __init__(self,dataset_root='',img_size=[1024,1024],**arg):
        with open(dataset_root,'r') as f:
            dataset_config = json.load(f)
        print('init')
        self.data_root = dataset_config['data_info']['data_root']
        self.sub_dirs = dataset_config['data_info']['use_sub']
        self.size = dataset_config['data_info']['img_size']
        self.size = img_size
        self.img_data = dataset_config['data']
        self.merge = dataset_config['data_info']['merge']
        self.dataset_num = dataset_config['data_info']['dataset_num']

    def get_image(self, img_info):
        imgs = {}
        imgs_path = {}
        for k in img_info:
            if "image" not in k and  "mask" not in k: continue
            v = img_info[k]
            if not self.merge:
                v = [v]
            imgs_path[k]=[]
            for p in v:
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_path[k].append(p)
                # img = merage_img(imgs_)
            # else:
            #     img = cv2.imread(v)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(int(self.size[0]),int(self.size[1])))
            img = torch.from_numpy(img.transpose(2, 0, 1))    
            imgs[k] = img
        imgs['paths']=imgs_path
        return imgs

    def __getitem__(self, idx):
        img_info =  self.img_data[idx%self.dataset_num]
        imgs = self.get_image(img_info)
        # imgs['img_info'] =img_info
        return imgs

    def __len__(self):
        return len(self.img_data)


# if __name__ == "__main__":
#     disasterdataset = DisasterDataset("../model_config/dataset_config_xBDearthquake.json")

#     for img_meta  in disasterdataset:
#         print(img_meta)