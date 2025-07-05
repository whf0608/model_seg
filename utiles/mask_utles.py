import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imgviz

def get_color_mask(mask,color='green',n=0.5):
    
    mask_contrary= cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
    if color == 'green':
        high_hsv = np.array([77,255,255])#这里填入三个max值
        low_hsv = np.array([35,43,46])
    elif color == 'red':
        high_hsv = np.array([10,255,255])#这里填入三个max值
        low_hsv = np.array([0,43,46])
    elif color == 'yellow':
        high_hsv = np.array([34,255,255])#这里填入三个max值
        low_hsv = np.array([26,43,46])
    elif color == 'blue':
        high_hsv = np.array([124,255,255])#这里填入三个max值
        low_hsv = np.array([100,43,46])
    elif color == 'white':
        high_hsv = np.array([180,30,255])#这里填入三个max值
        low_hsv = np.array([0,0,211])
    elif color == 'cyan':
        high_hsv = np.array([99,255,255])#这里填入三个max值
        low_hsv = np.array([78,43,46])
        
    mask_contrary  = cv2.inRange(mask_contrary,lowerb=low_hsv,upperb=high_hsv)#提取掩膜
    # mask_contrary[mask_contrary==0]=1
    # mask_contrary[mask_contrary==255]=0#把黑色背景转白色
    mask_bool = mask_contrary.astype(bool)

    mask_img=cv2.cvtColor(mask,cv2.COLOR_BGR2BGRA)
    mask_img[mask_bool==False]=[0,0,0,0]
    mask_img[:,:,-1] = int(255*n)
    return mask_img

def show_image_mask(image,mask,color='green',n=0.5,save_path=None):
    mask_img = get_color_mask(mask,color,n)
    # image_t2=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.imshow(mask_img)
    plt.axis('off')
    if save_path: 
        plt.savefig(save_path,bbox_inches='tight',pad_inches=0.0)

def show_image_mask_(mask,image,n=0.5,save_path=None):
    mask_img=cv2.cvtColor(mask,cv2.COLOR_RGB2RGBA)
    mask_img[:,:,-1]=int(255*n)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.imshow(mask_img)
    plt.axis('off')
    if save_path: 
        plt.savefig(save_path,bbox_inches='tight',pad_inches=0.0)
    
def mask2color(mask,color =  np.array([30/255, 144/255, 255/255, 0.6])):
    mask = cv2.cvtColor(mask,cv2.COLOR_RGBA2GRAY)
    h, w = mask.shape[-2:]
    mask = mask.astype(bool)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


def label_colormap(n_label=256):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0
    
    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b  
    # #我认为添加的一段，自己设置不同类别的rgb值
    # cmap[0,:] = [255,255,255] #背景的rgb值
    # cmap[1,:] = [255,0,0] #车的rgb
    # cmap[2,:] = [0,255,0] #人的rgb
    return cmap

### 返回字典{'0',[255,0,0]}
### list mask
def rgb2cls_masks(lbl,leaf_dict=None):
        v = set(leaf_dict.values())
        masks=[ np.zeros(lbl.shape,np.uint8) for _ in v ]
        h, w = lbl.shape[:2]
        if leaf_dict is None: 
            leaf_dict = {(0,0,0):0}
        
        idx = max(leaf_dict.values())+1
        white_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
        for i in range(h):
            for j in range(w):
                if tuple(lbl[i][j]) in leaf_dict or tuple(lbl[i][j]) == (0, 0, 0):
                    pass
                else:
                    print('add',tuple(lbl[i][j]))
                    leaf_dict[tuple(lbl[i][j])] = idx
                    idx += 1
        for color in leaf_dict.keys():
                mask = (lbl == list(color)).all(-1)
                # leaf = lbl * mask[..., None]      # colorful leaf with black background
                # np.repeat(mask[...,None],3,axis=2)    # 3D mask
                leaf = np.where(mask[..., None], white_mask, 0)
                # mask_name = './'+part+'/annotations/' + lbl_id +'_'+subClass +'_'+ str(idx) + '.png'  # ImageNumber_SubClass_idx.png
                # cv2.imwrite(mask_name, leaf)
                masks[leaf_dict[color]]+=leaf
    
        return  leaf_dict,masks

### mask 转数字label
def rgbmask2label(mask,leaf_dict=None):
    leaf_dict,masks = rgb2cls_masks(mask,leaf_dict)
    base = [np.zeros(mask.shape)]
    base.extend(masks)
    masks0 = np.array(base)[:,:,:,:1] 
    masks1 = np.argmax(masks0.transpose((3,1,2,0))[0],2)
    
    return leaf_dict,masks1

### 数字label转mask 
def label2rgbmask(label,label_colormap=None):
    lbl_pil = Image.fromarray(label.astype(np.uint8), mode="P")
    
    if label_colormap is None: colormap = imgviz.label_colormap()
    else: colormap = label_colormap
    lbl_pil = lbl_pil.putpalette(colormap.flatten())
    return lbl_pil

def image_diff(image_t1,image_t2):
    diff = image_t1-image_t2
    diff = cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
    plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(diff>100,'gray')

def label2rgbmask2(label,class_n=3,colormap=None):
    if colormap is None: colormap = imgviz.label_colormap(class_n)
    mask_img = np.zeros((label.shape[0],label.shape[1],3),np.uint8) 
    for i,color in enumerate(colormap):
        mask_img[label==i]=color
    return mask_img    