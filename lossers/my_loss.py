from torch  import nn
from torch.nn import functional as F
import torch
from lossers import dice_loss

bcecriterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()

def unet_loss_func2(rs,data,mask_key=['mask1_1'],n_classes=1):
    image_mask1_2 = data[mask_key[0]].cuda()[:,0:1]
    r1 = rs[0]
    loss = criterion(image_mask1_2.float(), r1.sum(dim=1))
    return loss/image_mask1_2.shape.numel()

def unet_loss_func(rs,data,mask_key=['mask1_1'],n_classes=3):
    image_mask1_2 = data[mask_key[0]].cuda()[:,0:1]/255.0
    
    if type(rs) != tuple:
        r1 = rs
    else:
        r1 = rs[0]
    
    image_mask1_2 = F.interpolate(image_mask1_2.float(), mode='bilinear', size=r1.shape[-2:])
    
    pre = F.softmax(r1, dim=1).float()
    lbl = F.one_hot(image_mask1_2[:,0].long(), n_classes)
    lbl =lbl.permute(0, 3, 1, 2).float()
    # print(pre.shape,lbl.shape)
    loss= criterion(image_mask1_2.float(), r1.sum(dim=1))\
        + dice_loss(pre,lbl,multiclass=True)\
        + bcecriterion(image_mask1_2[:,0].float(),r1.sum(dim=1))
    return loss
    
def loss_cross_entropy_mutliclas1(rs,data,mask_key=['mask1_1'],n_classes=3):
    image_mask1_2 = data[mask_key[0]][:,0:1]
    if type(rs) != tuple:
        r1 = rs
    else:
        r1 = rs[0]

    image_mask1_2 = image_mask1_2.cuda()
    image_mask1_2 = F.interpolate(image_mask1_2.float(), mode='bilinear', size=r1.shape[-2:])
    pre = F.softmax(r1, dim=1).float()
    lbl = F.one_hot(image_mask1_2[:,0].long(), n_classes)
    lbl =lbl.permute(0, 3, 1, 2).float()
    
    loss= criterion(image_mask1_2[:,0].float(), r1.sum(dim=1))\
        + bcecriterion(image_mask1_2[:,0].float(),r1.sum(dim=1))\
        + dice_loss(pre,lbl,multiclass=True)
    return loss


cdloss = torch.nn.CrossEntropyLoss()

def loss_cross_entropy_mutliclas(rs,data,mask_key='label',n_classes=2):
    # image_mask1_2 = data[mask_key[0]][:,0:1]

    # label_oh = F.one_hot(label.long(), n_classes)
    # label_oh = label_oh.transpose(1,4).squeeze(4)
    # print(r1,label)




    
    # loss = cdloss(r1,label.long()) #,ignore_index=0

    return final_loss 


def dice_loss(prediction, target):
    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)
    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def multitask_loss_func(rs, images, n_classes=3, mask_key=['t1_b','mask1_b1',['mask1_1'],'cd_cd_1','cd_mask_1']):
    n_class  = n_classes
    t_b      =  images[mask_key[0]].cuda().float()/255.0
    mask_2   =  images[mask_key[2]].cuda()/255.0
    mask_b   =  images[mask_key[1]].cuda().float()/255.0
    mask_cd   =  images[mask_key[3]].cuda().float()/255.0
    mask_cd_mask =  images[mask_key[4]].cuda().float()/255.0

    r1, r2, r3, r4, r5 = rs
    n_class=2

    loss = criterion( r1.sum(dim=1), t_b[:,0:1])\
        + criterion( r2.sum(dim=1), mask_b[:,0:1])\
        + criterion( r3.sum(dim=1), mask_2[:,0:1].float())\
        + criterion( r4.sum(dim=1), mask_cd_mask[:,0:1])\
        + criterion( r5.sum(dim=1), mask_cd[:,0:1])\
        + dice_loss(F.softmax(r1, dim=1).float(),
                        F.one_hot(t_b.long()[:,0], n_class).permute(0, 3, 1, 2).float(),
                        multiclass=True) \
        +dice_loss(F.softmax(r2, dim=1).float(),
                        F.one_hot(mask_b.long()[:,0], n_class).permute(0, 3, 1, 2).float(),
                        multiclass=True)\
        + dice_loss(F.softmax(r3, dim=1).float(),
                        F.one_hot(mask_2.long()[:,0], n_class).permute(0, 3, 1, 2).float(),
                        multiclass=True)\
        + dice_loss(F.softmax(r4, dim=1).float(),
                                               F.one_hot(mask_cd.long()[:,0], n_class).permute(0, 3, 1, 2).float(),
                                               multiclass=True)\
        + dice_loss(F.softmax(r5, dim=1).float(),
                                               F.one_hot(mask_cd_mask.long()[:,0], n_class).permute(0, 3, 1, 2).float(),
                                               multiclass=True)
       
    return loss

def rdd_3_loss_func(rs, images, mask_key=['mask1_1'], n_classes=3):
    mask3 = images[mask_key[0]].cuda()[:, 0]
    r1, r2, r3 = rs
    
    pre = F.softmax(r3, dim=1).float()
    lbl = F.one_hot(mask3.long(), n_classes + 1).permute(0, 3, 1, 2).float()
    loss = criterion(mask3.float(),r3.sum(dim=1))+dice_loss(pre, lbl, multiclass=True)

    return loss

def rdd_3_loss_func_muilt(rs, images,  n_classes=3,mask_key=['t1_b','mask1_b1',['mask1_1']]):
    t_b      =  images[mask_key[0]][:, 0].cuda() 
    mask_b   =  images[mask_key[1]][:, 0].cuda()
    mask   =  images[mask_key[2]][:, 0].cuda()
    
    t_b  = torch.nn.functional.normalize(t_b.cuda().float()).contiguous()
    mask_b = torch.nn.functional.normalize(mask_b.cuda().float()).contiguous()
    
    r1, r2, r3 = rs
    w1,w2,w3 = 1,1,1
    pre = F.softmax(r3, dim=1).float()
    lbl = F.one_hot(mask.long(), n_classes + 1).permute(0, 3, 1, 2).float()
    loss = w1*criterion(r1.sum(dim=1), t_b.float())+\
            w2*criterion(r2.sum(dim=1), mask_b.float())+\
            w3*criterion(r3.sum(dim=1), mask.float())+w3*dice_loss(pre, lbl, multiclass=True)

    return loss

def rdd_5_3_loss_func(rs, images, n_classes=2, mask_key=['t1_b', 'mask1_b1', 'mask1_1', 'cd_cd_1', 'cd_mask_1']):
    n_class = n_classes
    mask_2 = images[mask_key[2]][:, 0:1]
    mask_2[mask_2>1]=1
    mask_2 = mask_2.cuda()
    
    mask_cd = images[mask_key[3]][:, 0:1].cuda().float() / 255.0
    mask_cd_mask = images[mask_key[4]][:, 0:1].cuda().float() / 255.0
    
    
    r1, r2, r3, r4, r5 = rs
    pre_label = F.softmax(r3, dim=1).float()
    mask_2 = F.interpolate(mask_2.float(), mode='bilinear', size=r3.shape[-2:])
    label = F.one_hot(mask_2[:,0].long(), n_class).permute(0, 3, 1, 2).float()

    loss = criterion(mask_2[:,0].float(), r3.sum(dim=1)) \
           + bcecriterion(mask_2[:,0].float(), r3.sum(dim=1)) \
           + dice_loss(pre_label, label, multiclass=True)\
    
    label = F.one_hot(mask_2.long(), n_class).float()
    pred = r3.transpose(1, 2).transpose(2, 3).contiguous().view(-1, r3.shape[1])
    label = label.view(-1,r3.shape[1])
    loss = loss +  cross_entropy(pred, label)
    
    if torch.sum(mask_cd_mask) != 0:
        pre_label = F.softmax(r5, dim=1).float()
        mask_cd_mask = F.interpolate(mask_cd_mask.float(), mode='bilinear', size=r3.shape[-2:])
        label = F.one_hot(mask_cd_mask[:,0].long(), n_class).permute(0, 3, 1, 2).float()
        loss = loss + (criterion(mask_cd[:,0], r4.sum(dim=1)) + criterion(mask_cd_mask[:,0], r5.sum(dim=1))) \
               + bcecriterion(mask_cd[:,0].float(), r4.sum(dim=1)) + bcecriterion(mask_cd_mask[:,0].float(), r5.sum(dim=1)) \
               + dice_loss(pre_label, label, multiclass=True)

        label = F.one_hot(mask_cd_mask.long(), n_class).float()
        pred  = r5.transpose(1, 2).transpose(2, 3).contiguous().reshape(-1, r5.shape[1])
        label = label.view(-1,r5.shape[1])
        loss  = loss+ cross_entropy(pred, label)
  
    
    return loss