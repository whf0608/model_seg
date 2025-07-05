from glob import glob
import os
from os.path import join
import sys
import json
from pathlib import Path
import torch
import cv2
import numpy as np
from models import get_model

sys.path.append(os.getcwd()+'/../satellite_data/valite_dataset_sys')
from lib.utils.mask2svg import mask2pointss,draw_svg_label_color
from skimage import data, exposure
from vdatasets.tools import Datasets_Deal
from utiles.mask_utles import show_image_mask_
from metrics.m_metrics.metrics_np import Metrics


def evel_models(save_root, modes_config,dataset_roots,device, models_predection=True, models_metrics=True, img_size=(1024, 1024),
                cross_dataset_test=False, num_limt0=5000, min_num_limt=100, 
                show_img=False, show_mask=False, show_gtmaskimg=False, show_premaskimg=False,
                                  save_mask=True, save_fixedmask=False,show_fixedmaskimg=False):
    os.makedirs(save_root, exist_ok=True)
    metrics = Metrics(texformt_path=save_root + '/tex1.txt')
    results = []
    for config in modes_config:
        model, model_name, dataset_name_train = get_model_dataset(config)
        print('init model:', model_name)

        for dataset_name, dataset_path, img_ps in get_test_data(dataset_roots, num_limt0, min_num_limt):
            if not cross_dataset_test and dataset_name_train != dataset_name:
                continue

            print('dataset_name: ', dataset_name, 'imgs num:', len(img_ps), 'dataset path:', dataset_path)
            result_save_path = join(save_root, model_name,dataset_name,config.split('/')[-2])
            if models_predection:
                print('result save path: ', result_save_path)
                os.makedirs(result_save_path, exist_ok=True)
                model_result_test(model=model, device=device,imgs_paths=img_ps[:], img_size=img_size,
                                  show_img=show_img, show_mask=show_mask, show_gtmaskimg=show_gtmaskimg, show_premaskimg=show_premaskimg,
                                  save_mask=save_mask, save_fixedmask=save_fixedmask,
                                  img_mask_flg=['t2', 'mask2_2'], save_root=result_save_path)
            if models_metrics:
                pre_mask_paths = glob(result_save_path + '/*.png')
                print('metrics mask num:', len(pre_mask_paths),'path:',result_save_path + '/*.png')
                metrics.running(name=dataset_name+'_'+ config.split('/')[-2], paths=pre_mask_paths,
                                num_limt=num_limt0, result_f=result_save_path, mask_f=join(dataset_path, 'train', 'mask2_2'),
                                subfix=['_prelabel.png', '.png'], deal_img_func=None)
                results.append([metrics.get_metrics(),[model_name,dataset_name, config.split('/')[-2]]])
    return results


def get_model_dataset(config):
    cfg = json.load(open(config))
    cfg['training']['batch_size']=1
    print('config:',cfg)
    n_classes=cfg['data']['n_classes']
    load_weights = config.replace('model_config.json','model.pt')
    data_name = cfg['data']['path'].split('/')[-1]
    
    load_weights0 = load_weights.replace('model.pt','model.pth')

    model = get_model(cfg['model'])(3, n_classes)
    if Path(load_weights).exists():
        print('loading weights',load_weights)
        model = torch.nn.parallel.DataParallel(model)
        model.load_state_dict(torch.load(load_weights, map_location='cpu'),strict=True)
    elif Path(load_weights0).exists():
        print('loading weights',load_weights)
        model.load_state_dict(torch.load(load_weights0, map_location='cpu'),strict=True)
        model = torch.nn.parallel.DataParallel(model)
    else:
        print('no weights: ',config.replace('model_config.json','model.pt'))
    return model,cfg['model'],data_name

    
def get_test_data(dataset_roots,num_limt0=5000,min_num_limt=100):
    dd=Datasets_Deal(dataset_roots,tt_pre=['t2'])
    # dd.clearn_subfile(ttv=[save_metric])
    testloader = dd.get_data(img_flg='/train/t2/*.png',num_limt0=num_limt0,min_num_limt=min_num_limt)
    return testloader
    
def model_result_test(model, device,imgs_paths=[],save_root='/home/wanghaifeng/whf_work/model_works/model_result',img_size=(1024,1024),img_mask_flg=['t2','mask2_2'],
                     show_img=False,show_mask=False,show_gtmaskimg=False,show_premaskimg=False,show_fixedmaskimg=False,save_mask=False,save_fixedmask=False):
    
    for ni,img_p in enumerate(imgs_paths):
        img = cv2.imread(img_p)
        if img is None:  
            print('not img:',img_p)
            continue
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img_orgin = cv2.resize(img,img_size)
        img = img_orgin.copy()
        
        save_path = join(save_root,img_p.split('/')[-1])
        
        mask2_p = img_p.replace(img_mask_flg[0],img_mask_flg[1])
        if Path(mask2_p).exists():
            mask = cv2.imread(mask2_p)
            mask = cv2.resize(mask, img_size)
        else:
            print("not exists mask==================")
            mask = np.ones(img.shape)
            
        if show_img: 
            cv2.imwrite(save_path.replace('.png','_img.png'),img)
        if show_mask:
            cv2.imwrite(save_path.replace('.png','_label.png'),mask)
        
        img = img.transpose(2,0,1)
        img = np.array([img])
        img = torch.from_numpy(img).to(device).float()
        img =torch.nn.functional.normalize(img)
        model = model.to(device)
        with torch.no_grad():
            rs = model(img)

        img0 = img_orgin.copy()
        mask_label0 = np.zeros(img0.shape,np.uint8)

        if show_gtmaskimg:
            mask = mask[:,:,0]
            mask_label0[mask==1]=[0,255,0]
            mask_label0[mask>1]=[255,0,255]
            mask_label0= cv2.resize(mask_label0,dsize = img_orgin.shape[:2])
            show_image_mask_(mask_label0,img0,n=0.5,save_path=save_path.replace('.png' ,'_gt.png'))
        
        if type(rs)== tuple:     
            buliding_mask0 = torch.argmax(rs[2],1).detach().cpu().numpy()[0]
            damged_mask0= torch.argmax(rs[4],1).detach().cpu().numpy()[0]    
        else:
            r_mask = torch.argmax(rs,1).detach().cpu().numpy()[0]
            buliding_mask0 = np.zeros(r_mask.shape,np.uint8)
            damged_mask0  =  np.zeros(r_mask.shape,np.uint8)
            buliding_mask0[r_mask==1] =1
            damged_mask0[r_mask==2] = 1
        
        buliding_mask0 = cv2.resize(buliding_mask0,dsize = img_orgin.shape[:2])
        damged_mask0 = cv2.resize(damged_mask0,dsize = img_orgin.shape[:2])
            
        if save_mask:
            pre_mask = np.zeros(img0.shape[:2],np.uint8)
            pre_mask[buliding_mask0>0]=1
            pre_mask[damged_mask0>0]=2
            cv2.imwrite(save_path.replace('.png','_prelabel.png'),pre_mask)
            
        if show_premaskimg:
            r_mask0 = np.zeros(img0.shape,np.uint8)
            r_mask0[buliding_mask0>0]=[0,255,0]
            r_mask0[damged_mask0>0]=[255,0,255]
            r_mask0 = cv2.resize(r_mask0,dsize = img_orgin.shape[:2])
            show_image_mask_(r_mask0,img_orgin,n=0.5,save_path=save_path.replace('.png' ,'_pre.png'))
            
        if save_fixedmask or show_fixedmaskimg:
            buliding_mask,contours0,contours = mask_pre_2_normal(buliding_mask0,full=True,hull=False,minpoly=True)
            damged_mask,contours0,contours = mask_pre_2_normal(damged_mask0,full=True,hull=False,minpoly=True)
            
            if save_fixedmask:
                pre_mask = np.zeros(img0.shape[:2],np.uint8)
                pre_mask[buliding_mask0>0]=1
                pre_mask[damged_mask>0]=2
                cv2.imwrite(save_path.replace('.png','_fixedlabel.png'),mask)
            
            if show_fixedmaskimg:
                r_mask1 = np.zeros(img0.shape,np.uint8)
                r_mask1[buliding_mask>0]=[0,255,0]
                r_mask1[damged_mask>0]=[255,0,255]

                show_image_mask_(r_mask1,img0,n=0.5,save_path=save_path.replace('.png' ,'_fix.png'))
                
def mask_pre_2_normal(mask, hull=False,full=False,minpoly=False):
    mask = np.array(mask*255,np.uint8)
    # mask = cv2.dilate(mask, None)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask= cv2.dilate(mask, kernel,2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask= cv2.erode(mask, kernel,2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours0=[]
    for contour in contours:
        if cv2.contourArea(contour)>100:
            if hull:contour = cv2.convexHull(contour)
            if minpoly:contour = cv2.approxPolyDP(contour,3,True)
            
            points=[]
            for c in contour[0:]:
                points.append([int(c[0][0]), int(c[0][1])])
            points.append([int(contour[0][0][0]), int(contour[0][0][1])])
            contour = points
            contours0.append(np.array(contour))
       
    if full:
        mask = np.zeros(mask.shape)
        mask = cv2.fillPoly(mask, contours0, 255)
    mask = mask.astype(np.uint8)
    return mask,contours0,contours

# img_size = (1024,1024)
# imgs_path = glob(r'/home/wanghaifeng/project_work/datasets/disaster_dataset/imgs8/train/t2/*.png')
# # imgs_path = glob('/home/wanghaifeng/project_work/datasets/disaster_dataset/harvey/train/t1/*.png')
# num_limt = len(imgs_path)
# num_limt = 1
# imgs_paths = imgs_path[0:num_limt]
# model_result_test(model=None,imgs_paths=img_ps,img_size=img_size,
#                   show_img=False,show_mask=False,show_gtmaskimg=False,show_premaskimg=False,save_mask=False,save_fixedmask=False
#                   img_mask_flg=['t2','mask2_2'],save_root='/home/wanghaifeng/whf_work/model_works/')