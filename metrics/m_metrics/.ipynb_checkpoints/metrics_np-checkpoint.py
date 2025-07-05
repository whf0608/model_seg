# from hausdorff import hausdorff_distance
# from init import print_log
import random
import numpy as np
import cv2
from pathlib import Path

print_log =print

######## 指标
smooth = 1e-5

# 正样本正确率
# def TP(img,mask):
#     return img[mask>0].sum()

def TP(img,mask):
    return ((img*mask).sum()+smooth)/(mask.sum()+smooth)

#负样本正确率

# def TN(img,mask):
#     return (img[mask_img==0]==0).sum()
def TN(img,mask):
    return (((1-img)*(1-mask)).sum()+smooth)/((mask==0).sum()+smooth)

### 多余不部分
# def FP(img,mask):
#     return img[mask==0].sum()

def FP(img,mask):
    return ((img*(1-mask)).sum()+smooth)/(img.sum()+smooth)

### 缺检测
# def FN(img,mask):
#     return (mask.sum()-img[mask>0].sum())

def FN(img,mask):
    return (mask.sum()-(img*mask).sum()+smooth)/(mask.sum()+smooth)

###精确度
def PPV(img,mask):
    return TP(img,mask)/(TP(img,mask)+FP(img,mask))

###recall Sensitivity
def TPR(img,mask):
    return (TP(img,mask))/(TP(img,mask)+FN(img,mask))

### 特异度
def TNR(img,mask):
    return (TN(img,mask))/(TN(img,mask)+FP(img,mask))

### f1 == Dice
def F1(img,mask):
    return (2*TP(img,mask))/(2*TP(img,mask)+FP(img,mask)+FN(img,mask))

### dice==f1
def Dice(img,mask):
    return  ((img*mask).sum())/((img.sum()+mask.sum())/2.0)

### IOU
def IOU(img,mask):
    return (TP(img,mask))/(FP(img,mask)+TP(img,mask)+FN(img,mask))

# def IOU(img,mask):
#     return ((img&mask).sum()+smooth)/((img|mask).sum()+smooth)


def metrics(img,mask):
    print('正样本正确率(TP):',TP(img,mask))
    print('负样本正确率(TN):',TN(img,mask))
    print('多余部分(FP):', FP(img,mask))
    print('漏检部分(FN):',FN(img,mask))
    print('精度(PPV):', PPV(img,mask))
    print('recall(TPR):',TPR(img,mask))
    print('特异度(TNR):',TNR(img,mask))
    print('f1:',F1(img,mask))
    print('dice(f1):',Dice(img,mask))
    print('IOU:',IOU(img,mask))
    
    
class SegRunningScore(object):

    def __init__(self, n_classes):
        self.n_classes =n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)

        return hist

    def update(self, label_preds, label_trues):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def _get_scores(self):
    
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return acc, acc_cls, fwavacc, mean_iu, cls_iu

    def get_mean_iou(self):
        return self._get_scores()[3]

    def get_cls_iou(self):
        return self._get_scores()[4]

    def get_pixel_acc(self):
        return self._get_scores()[0]

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
        

############ 输出
class  TexFormat:
    def __init__(self,path='tex.txt',keys=['TP','TN','FP','FN','PPV','TPR','TNR','f1','dice','IOU']):
        self.path=path
        self.keys=keys
        with open(path,'w') as f:
            f.write('name\t')
            for k in keys:
                f.write(k+'\t\t')
            f.write('\n')
            
    def init_head(self):
        with open(self.path,'a') as f:
            f.write('name\t')
            for k in self.keys:
                f.write(k+'\t\t')
            f.write('\n')
    def texformat(self,name='',values=[0.1245,0.2545,0.141,0.2626,0.2155,0.1152,0.554,0.1515,0.5151,0.15151]):
        assert len(self.keys)==len(values),'keys lenght: '+str(len(self.keys))+' value lenght: '+str(len(values))
        
        with open(self.path,'a') as f:
            f.write(name+'\t &')
            for v in values:
                f.write(str('%.2f'%(v*100))+'%\t &')
            f.write('\\\\ \n')


#### 语义分割
def img_label(img,mask_img):
 
    f1=img[...,0]==201
    f2=img[...,1]==30
    f3=img[...,2]==248 
    img[f1&f2&f3]=[0,0,0]
    img= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img[img>0]=255

    mask_img = cv2.cvtColor(mask_img,cv2.COLOR_RGB2GRAY)

    img= img/255
    mask_img =mask_img/255

    img = np.array(img,np.int0)
    mask_img  = np.array(mask_img ,np.int0)
    return img,mask_img

###二分割
def img_label(img,mask_img):
     
    img= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img[img>0]=255

    mask_img = cv2.cvtColor(mask_img,cv2.COLOR_RGB2GRAY)
    
    mask_img[mask_img>0] =255
    img= img/255
    mask_img =mask_img/255

    img = np.array(img,np.int0)
    mask_img  = np.array(mask_img ,np.int0)
    return img,mask_img

class Metrics:
    def __init__(self,show_texformt=True,texformt_path='./tex.txt',single=False):
        
        self.texformt = TexFormat(path=texformt_path)
        self.show_texformt=show_texformt

        if single:
            self.TPs, self.TNs, self.FPs, self.FNs, self.PPVs, self.TPRs, self.TPRs, self.TNRs, self.F1s, self.Dices, self.IOUs = [], [], [], [], [], [], [], [], [], [], []

    def update(self,img,mask):
        # print(img.shape,mask.shape)
        assert img.max()<=1 and mask.max()<=1  and img.min()>=0 and mask.min()>=0
        self.TPs.append(TP(img,mask))
        self.TNs.append(TN(img,mask))
        self.FPs.append(FP(img,mask))
        self.FNs.append(FN(img,mask))
        self.PPVs.append(PPV(img,mask))
        self.TPRs.append(TPR(img,mask))
        self.TNRs.append(TNR(img,mask))
        self.F1s.append(F1(img,mask))
        self.Dices.append(Dice(img,mask))
        self.IOUs.append(IOU(img,mask))
    def get_metrics(self,index=None):
        return [sum(self.TPs)/(len(self.TPs)+smooth),
                sum(self.TNs)/(len(self.TNs)+smooth),
                sum(self.FPs)/(len(self.FPs)+smooth),
                sum(self.FNs)/(len(self.FNs)+smooth),
                sum(self.PPVs)/(len(self.PPVs)+smooth),
                sum(self.TPRs)/(len( self.TPRs)+smooth),
                sum(self.TNRs)/(len(self.TNRs)+smooth),
                sum(self.F1s)/(len(self.F1s)+smooth),
                sum(self.Dices)/(len(self.Dices)+smooth),
                sum(self.IOUs)/(len(self.IOUs)+smooth)
               ]
    
    def show_metrics(self,name=''):
        values=self.get_metrics()
        
        print_log('dataset name: ',name, 'data num:',len(self.IOUs))
        print_log('metrics----------------- ')
        print_log('正样本正确率(TP):','%.5f'%(values[0]))
        print_log('负样本正确率(TN):','%.5f'%(values[1]))
        print_log('多余部分(FP):',    '%.5f'%(values[2]))
        print_log('漏检部分(FN):',    '%.5f'%(values[3]))
        print_log('精度(PPV):',       '%.5f'%(values[4]))
        print_log('recall(TPR):',     '%.5f'%(values[5]))
        print_log('特异度(TNR):',     '%.5f'%(values[6]))
        print_log('f1:',              '%.5f'%(values[7]))
        print_log('dice(f1):',        '%.5f'%(values[8]))
        print_log('IOU:',             '%.5f'%(values[9]))
        print_log('metrics----------------- ')
        
        self.texformt.texformat(name,values)
        
        
    def running(self,name='',paths='.',num_limt=-1,result_f='result4_2',mask_f='mask2_1',subfix=['.png','.png'],deal_img_func=None,report_empty_pre=False):
        paths=np.array(paths)
        self.TPs,self.TNs,self.FPs,self.FNs,self.PPVs,self.TPRs,self.TPRs,self.TNRs,self.F1s,self.Dices,self.IOUs=[],[],[],[],[],[],[],[],[],[],[]
        if num_limt==-1: num_limt=len(paths)
        
        if num_limt<5 or len(paths)<5: 
            print_log('data  more lower ---- no metrics')
            return False
        if len(paths)<num_limt: num_limt=len(paths)
        
        print_log('images num:'+str(num_limt))
        index = [random.randint(0,len(paths)-1)  for _  in range(num_limt)]
        mask_empty =0
        
        for path in paths[index]:
            img = cv2.imread(path)
            # img[img<10]=0
            if deal_img_func is not None: img = deal_img_func(img)
        
            mask_path = path.replace(result_f,mask_f).replace(subfix[0],subfix[1])
            if Path(mask_path).exists():
                mask_img = cv2.imread(mask_path)
            else:
                mask_img = np.zeros(img.shape,np.uint8)
                mask_empty +=1
                if report_empty_pre: print_log('mask empty:', mask_path)
            # print('Pre mask:',path,'label mask: ',mask_path)    
            mask_img  = cv2.resize(mask_img,(img.shape[1],img.shape[0]))
            img,mask_img = img_label(img,mask_img)
            self.update(img,mask_img)
            # print('metrics index:',self.get_metrics())
        print_log(name+' pre mask empty num:', mask_empty)
        if num_limt-mask_empty <5:
            print_log('pre data  more lower ---- no metrics')
        else:
            self.show_metrics(name)
        return True

    def running_online_single(self,img=None,mask_img=None):
        self.update(img, mask_img)
        # self.show_metrics(name)
        return True