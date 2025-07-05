from glob import glob
import torch
import cv2
import numpy as np
from pathlib import Path
from models import get_model
from utiles.tensor2imgmask import show_result_hard_orig
import argparse
import json
from metrics.m_metrics.metrics_np  import Metrics
from utiles.mask_utles import  show_image_mask_
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def test(imgs_path,cfg,device,load_weights='',save_path='',num_limt=1,img_size=(640,640),n_classes=3,show_mask=False,show_pre_mask=True,show_img=False,model_data_papra=False):
    m = Metrics(texformt_path=save_path+'tex.txt',single=True)

    model = get_model(cfg['model'])(3, n_classes+1)
    model = model.to(device).eval()

    if len(load_weights) > 0:
        model = torch.nn.parallel.DataParallel(model)
        print('loading weights', load_weights)
        model.load_state_dict(torch.load(load_weights, map_location='cpu'))
    
    for ni,img_p in enumerate(imgs_path[0:num_limt]):
        img = cv2.imread(img_p)
        if img is None:  continue
        img_orgin = cv2.resize(img,img_size)
        
        img = img_orgin.copy()
        if show_img: 
            cv2.imwrite(save_path+img_p.split('/')[-1].replace('.png','_img.png'),img)
        
        mask2_p = img_p.replace('t1','mask1_1')
        if Path(mask2_p).exists():
            mask = cv2.imread(mask2_p)
            mask = cv2.resize(mask, img_size)
        else:
            print("not exists mask==================")
            mask = np.ones(img.shpe)
        if show_mask:
            cv2.imwrite(save_path+img_p.split('/')[-1].replace('.png','_label.png'),mask)

        
        img = img.transpose(2,0,1)
        img = np.array([img])
        img = torch.from_numpy(img).to(device).float()
        img =torch.nn.functional.normalize(img)

        # mask = torch.from_numpy(mask).to(device).float()
        # mask = torch.nn.functional.normalize(mask)
        with torch.no_grad():
            rs = model(img)
        
        for i,r in enumerate(rs):
            if i==0:
                img_pre_mask = show_result_hard_orig(r)
                if show_pre_mask:
                    if img_pre_mask.max() > 0:
                        # print(str(ni)+'___',img_p.replace('t2','cd_point').replace('_img','_img'+str(i+4)))
                        # cv2.imwrite(save_path+img_p.split('/')[-1].replace('.png','_img'+str(i)+'_.png'),img)
                        print(save_path + img_p.split('/')[-1])
                        cv2.imwrite(save_path + img_p.split('/')[-1], img_pre_mask)
                        img_orgin = cv2.cvtColor(img_orgin,cv2.COLOR_RGB2BGR)
                        show_image_mask_(img_pre_mask,img_orgin,n=0.1,save_path=save_path + img_p.split('/')[-1].replace('.png','_imglabel.png'))
                        
                        # print(img_p.replace('t1','result5').replace('.png',str(i+0)+'.png'))
                        # cv2.imwrite(img_p.replace('t1','result5').replace('.png',str(i+0)+'.png'),img)

                # img_pre_mask=torch.argmax(F.softmax(r, dim=1).float())/n_classes
                # mask = torch.argmax(mask.to(device).float(), dim=1)/n_classes
                # print(img_pre_mask.max(),img_pre_mask.min(),mask.max(),mask.min())
                m.running_online_single(img=img_pre_mask/255.0,mask_img=mask/255.0)
    m.show_metrics(name='xDBeq')


# imgs_path = glob(r'/home/wanghaifeng/project_work/datasets/changedetection_dataset/xBD*/train/t1/*.png')

# imgs_path = glob(r'/home/wanghaifeng/project_work/datasets/changedetection_dataset/Wuhan/train/t1/*.png')

# save_path='results/'

# model_name = 'model_multitask_2022_09_29_15_43_35'

# save_path='results/'+model_name+'/'
#
# weights_path = 'weights/'+model_name+'/model.pt'

def parse_args():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--img-file', help='images path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--weights-path', help='the checkpoint file')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    save_path = args.work_dir
    weights_path = args.weights_path
    img_file= args.img_file

    if not Path(save_path).exists():
        Path(save_path).mkdir()

    cfg = json.load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if '.json' in img_file:
        with open(img_file) as f:
            imgs_path = json.load(f)

    else: imgs_path = glob(img_file+'/*.png')

    test(imgs_path,cfg,device, load_weights=weights_path, save_path=save_path, img_size=(1024,1024), num_limt=2000000)


