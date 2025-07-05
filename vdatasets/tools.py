from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# from init import print_log
import random
            

class Datasets_Deal:
    def __init__(self,dataset_roots,ttv_pre=['train'],
                  tt_pre=['t2'],
                 create=False,
                 is_not_used_tt_pre=False,
                 anlysis_tt_name=False
              ):
        self.dataset_roots=dataset_roots
        self.ttv_pre=ttv_pre
        self.tt_pre=tt_pre
        
        
        self.dataset_deal_running=True
        self.ttv_deal_running=True
        self.tt_deal_running=True
        self.imgs_deal_running=True
        
        
        self.datasets_infos={}
        self.anlysis_dataset_name_path=True
        self.anlysis_ttv_name_path=True
        
        self.anlysis_tt_name_path=True
        self.anlysis_tt_imgs_num=True
        self.show_all_tt_imgs_num=False
        self.show_tt_imgs_N=False
        self.tt_imgs_N=10
        
        self.is_not_used_tt_pre=is_not_used_tt_pre
        self.datasets_infos['tt_names']=set()
        self.anlysis_tt_name=anlysis_tt_name
        self.show_tt_names=False
    
  
        self.create_ttv = create and (not  self.is_not_used_tt_pre)
        self.create_tt =  create and (not  self.is_not_used_tt_pre)
     
        
        if self.is_not_used_tt_pre:
            assert  self.is_not_used_tt_pre==True & self.anlysis_tt_name==True
      
    
        self.running()
       
    
    def running(self):
        dataset_roots=self.dataset_roots
        
        ### 遍历数据集
        for dataset_root in dataset_roots:
            datasets = glob(dataset_root)
            if self.dataset_deal_running:self.dataset_deal(datasets)
         
        print('datasets num: ',len(self.datasets_infos))
        if self.show_tt_names:  print('sub_img_files: ',self.datasets_infos['tt_names']) 
    
    def get_data(self,img_flg='/train/t2/*.png',num_limt0=1000,min_num_limt=50):    
        for k in list(self.datasets_infos.keys())[1:]:
            dataset_path = self.datasets_infos[k]['dataset_path']
            imgs_path=glob(dataset_path+img_flg)
            if len(imgs_path)<min_num_limt:  continue
            if len(imgs_path)<num_limt0: 
                num_limt=len(imgs_path)
            else:
                num_limt = num_limt0
            index = [random.randint(0,len(imgs_path)-1)  for _  in range(num_limt)]
            imgs_path=np.array(imgs_path)
            
            yield k,dataset_path,imgs_path[index]

            
    def dataset_deal(self,datasets):
        
        for dataset in datasets:
            dataset_name=dataset.split('/')[-1]
            if self.anlysis_dataset_name_path: self.datasets_infos[dataset_name]={'dataset_path':dataset}
            if self.ttv_deal_running:  self.ttv_deal(dataset,dataset_name)
            if self.is_not_used_tt_pre: 
                self.tt_pre = self.datasets_infos['tt_names']
                self.ttv_deal(dataset,dataset_name)
    
    def ttv_deal(self,dataset_path,dataset_name):
        
        for ttv in  self.ttv_pre:
            ttv_p=dataset_path +'/'+ ttv
            if self.create_ttv and (not Path(ttv_p).exists()): 
                Path(ttv_p).mkdir()
                print('mkdir', ttv)
                
            if self.anlysis_ttv_name_path: self.datasets_infos[dataset_name][ttv]={'ttv_path':ttv_p}
                
            if self.anlysis_tt_name: self.anlysis_tt_name_func(ttv_p,dataset_name,ttv)
            
            if self.tt_deal_running: self.tt_deal(ttv_p,dataset_name,ttv)
            
            
                
    def tt_deal(self,ttv_path,dataset_name,ttv):
       
        for tt in  self.tt_pre:
            tt_p=ttv_path +'/'+ tt
            if self.create_tt and (not Path(tt_p).exists()): 
                Path(tt_p).mkdir()
                print('mkdir', tt_p)
            
            if self.anlysis_tt_name_path: self.datasets_infos[dataset_name][ttv][tt]={'tt_path':tt_p}
            
            if self.imgs_deal_running: self.imgs_deal(tt_p,dataset_name,ttv,tt)
            

    def imgs_deal(self,tt_p,dataset_name,ttv,tt):
        imgs_fs = glob(tt_p+'/*.png')
        imgs_fs = np.array(imgs_fs)
        if self.anlysis_tt_imgs_num: self.datasets_infos[dataset_name][ttv][tt]['imgs_num']=len(imgs_fs)
        
        if self.show_all_tt_imgs_num: print(len(imgs_fs),tt_p)
        if self.show_tt_imgs_N and len(imgs_fs)<self.tt_imgs_N: print(len(imgs_fs),tt_p)
                 
            
    def anlysis_tt_name_func(self,ttv_p,dataset_name,ttv):
        fs = glob(ttv_p+'/*')
        for f_p  in fs:
            if Path(f_p).is_dir():
                self.datasets_infos['tt_names'].add(f_p.split('/')[-1])
    
    def get_sub_files_all(self,datasets=None,ttvs=['train'],tts=['t1'],subfix=['/*.png']):
        imgs_fs =[]
        for k in datasets:
            for ttv_ in ttvs:
                for tt_,subfix_ in zip(tts,subfix):
                    imgs_p =self.datasets_infos[k][ttv_]['ttv_path']+'/'+tt_
                    
                    imgs_fs+=glob(imgs_p+subfix_)
        return imgs_fs
    
    def get_sub_files_all_by_paths(self,datasets_paths=[],ttvs=['train'],tts=['t1'],subfix=['/*.png']):
        imgs_fs =[]
        for k in datasets:
            for ttv_ in ttvs:
                for tt_,subfix_ in zip(tts,subfix):
                    imgs_p = k+'/'+ttv_+'/'+tt_
                    
                    imgs_fs+=glob(imgs_p+subfix_)
        return imgs_fs
    
    
    def clearn_subfile(self,ttv=['result4_2'],del_file=False):
        for k in list(self.datasets_infos.keys())[1:]:
            for ttv_ in ttv:
                path = self.datasets_infos[k]['train'][ttv_]['tt_path']
                img_fs = glob(path+'/*.png')
                if len(img_fs)>0:
                    for img_f in img_fs:
                        Path(img_f).unlink()
                    print('clearn '+path)
            if del_file: 
                Path(path).rmdir()
                print('clearn  file '+path)
    
class Image_deal:
    def __init__(self,deal_subfile,save_subfile):
        self.deal_subfile=deal_subfile
        self.save_subfile=save_subfile
        
    def running(self,datasets_infos):
        for k in list(datasets_infos.keys())[1:]:
            for dsubf,ssubf in zip(self.deal_subfile,self.save_subfile):
                mask1_dic = datasets_infos[k]['train'][dsubf]
                mask1_1_dic =  datasets_infos[k]['train'][ssubf]
                if mask1_dic['imgs_num']==0:
                    continue
                if mask1_dic['imgs_num']> mask1_1_dic['imgs_num']:
                    imgs_fs = glob(mask1_dic['tt_path']+'/*.png')
                    for img_path in imgs_fs[0:]:
                        assert dsubf in img_path, dsubf+'not in '+img_path
                        save_path = img_path.replace(dsubf,ssubf)
                        self.rgb2gray(img_path,save_path)
                        
    def show_num(self,datasets_infos):
        for k in list(datasets_infos.keys())[1:]:
            for dsubf,ssubf in zip(self.deal_subfile,self.save_subfile):
                mask1_dic = datasets_infos[k]['train'][dsubf]
                mask1_1_dic =  datasets_infos[k]['train'][ssubf]
                if mask1_dic['imgs_num']==0:
                    continue
                if mask1_dic['imgs_num']> mask1_1_dic['imgs_num']:
                    print(str(mask1_dic['imgs_num'])+'--'+str(mask1_1_dic['imgs_num']),mask1_dic['tt_path'])
                    
    def show_tt_num_diff(self,datasets_infos,tts=['t2','mask2','mask2_1'],used_first=True):
        for k in list(datasets_infos.keys())[1:]:
            imgs_num=datasets_infos[k]['train'][tts[0]]['imgs_num']
            s=k+' '+tts[0]+' num: '+str(imgs_num)
            if imgs_num==0: continue
            diff=[]
            for tt in tts[1:]:
                imgs_num0=datasets_infos[k]['train'][tt]['imgs_num']
                diff.append(imgs_num0-imgs_num)
                s+=' '+tt+' num: '+str(imgs_num0)
            print(s,diff)
            
    def rgb2gray(self,img_path,save_path):
        img = cv2.imread(img_path)
        assert len(img.shape)>1, 'image shape:'+str(len(img.shape))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img[img>0]=255
        print('Image deal save path: ',save_path)
        cv2.imwrite(save_path,img)