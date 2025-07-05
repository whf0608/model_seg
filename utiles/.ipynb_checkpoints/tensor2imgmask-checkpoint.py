from utiles.mask_utles import label2rgbmask2
import torch.nn.functional as F
import numpy as np
import torch

def show_img(t1_b,orgin=False):
    img = t1_b[0]
    img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    if not orgin: 
        img = (img-img.min())
        img=img/(img.max()+0.00001)
        img *= 255
        img = img.astype(np.uint8)
    return img

def show_result(r1):
    img1 = F.softmax(r1, dim=1)
    # img1 = torch.argmax(probs,1)
    img1 = img1.detach().cpu().numpy()
    img1 = img1[0].transpose(1,2,0)
    img1 = np.array(img1*255,np.uint8)
    return img1

def show_result_(r1):
    # img1 = F.softmax(r1, dim=1)
    # img1 = torch.argmax(probs,1)
    img1 = r1[0][0].detach().cpu().numpy()
    img1 = img1>0    
    img1 = np.array(img1*255,np.uint8)
    return img1

def show_result_hard(r1):
    img1 = F.softmax(r1, dim=1)
    img1 = torch.argmax(img1,1)
    img1 =F.one_hot(img1, 3)
    img1=img1[0].detach().cpu().numpy()
    img1 = np.array(img1*255,np.uint8)
    return img1

def show_result_hard_orig(r1):
    if r1.shape[1]==1:
        return show_result_(r1)
    # print(r1.shape)
    img1 = torch.argmax(r1,1)

    img1 = img1[0].detach().cpu().numpy()
    
    colormap={(0, 0, 0): 0, (0, 254, 0): 1, (254, 0, 0): 2, (0, 68, 254): 3}
    mask1_ = label2rgbmask2(img1, colormap=list(colormap.keys()))
    # print(mask1_.max(),mask1_.shape,'=================')
    return np.array(mask1_, np.uint8)

def show_mask(mask,img_mask=False):
    mask = mask.detach().cpu().float().numpy()
    if img_mask:
        mask/=255.0
    colormap = {(0, 0, 0): 0, (0, 254, 0): 1, (254, 0, 0): 2, (0, 68, 254): 3}
    mask1_ = label2rgbmask2(mask[0][0], colormap=list(colormap.keys()))
    return np.array(mask1_, np.uint8)