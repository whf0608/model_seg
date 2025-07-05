import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from .offset_helper import DTOffsetHelper

bcecriterion = nn.BCEWithLogitsLoss()
edgecriterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.85))
cecriterion=nn.CrossEntropyLoss(ignore_index=-1)

def loss(result,gt,mask_key=['true_masks','mask_deg_1','mask_depth_1']):

    remasks_pred, masks_pred, pred2, pred3, pred4, pred5, edge1, edge2, edge3, edge4, direction  = result

    true_masks = gt[mask_key[0]][:,0:1].cuda()

    dis_masks = gt[mask_key[1]][:,0].cuda().float()/180.0
    dir_masks = gt[mask_key[2]][:,0].cuda().float()/255.0

    _, dir_masks = DTOffsetHelper.align_angle(dir_masks, num_classes=8,
                                                 return_tensor=True)
    true_masks = true_masks >0
    edge_masks = (dis_masks < 5)
    dir_masks[edge_masks == 0] = -1
    # print("===============",direction.shape,dir_masks.shape,edge_masks.shape, true_masks.shape,remasks_pred.shape)
    loss =bcecriterion(masks_pred.squeeze(), torch.sigmoid(remasks_pred).squeeze().float())+ \
                      bcecriterion(remasks_pred.squeeze(), true_masks.squeeze().float())+\
                      bcecriterion(masks_pred.squeeze(), true_masks.squeeze().float())+\
                      edgecriterion(edge1.squeeze(), edge_masks.squeeze().float())+ \
                      cecriterion(direction,dir_masks.long())#+\
                      # 0.25*bcecriterion(pred2.squeeze(), F.interpolate(true_masks.float(),mode='bilinear',size=(320,320)).squeeze().float())+ \
                      # 0.25*bcecriterion(pred3.squeeze(), F.interpolate(true_masks.float(),mode='bilinear',size=(160,160)).squeeze().float())+ \
                      # 0.25*bcecriterion(pred4.squeeze(), F.interpolate(true_masks.float(),mode='bilinear',size=(80,80)).squeeze().float())+ \
                      # 0.25*bcecriterion(pred5.squeeze(), F.interpolate(true_masks.float(),mode='bilinear',size=(40,40)).squeeze().float())+ \
                      # 0.25*edgecriterion(edge2.squeeze(), F.interpolate(edge_masks.unsqueeze(1).float(),mode='bilinear',size=(320,320)).squeeze().float())+ \
                      # 0.25*edgecriterion(edge3.squeeze(), F.interpolate(edge_masks.unsqueeze(1).float(),mode='bilinear',size=(160,160)).squeeze().float())+ \
                      # 0.25*edgecriterion(edge4.squeeze(), F.interpolate(edge_masks.unsqueeze(1).float(),mode='bilinear',size=(80,80)).squeeze().float())
    return loss


def loass_1(result,batch,mask_key=[]):

    true_masks = batch['mask'] > 0
    dir_masks = batch['direction_map']
    dis_masks = batch['distance_map']

    true_masks = true_masks.cuda().float()
    dir_masks = dir_masks.cuda().float()
    edge_masks = (dis_masks < 5).cuda().float()
    dir_masks[edge_masks == 0] = -1

    remasks_pred, masks_pred, pred2, pred3, pred4, pred5, edge1, edge2, edge3, edge4, direction = result
    loss = bcecriterion(masks_pred.squeeze(), torch.sigmoid(remasks_pred).squeeze().float()) + \
           bcecriterion(remasks_pred.squeeze(), true_masks.squeeze().float()) + \
           bcecriterion(masks_pred.squeeze(), true_masks.squeeze().float()) + \
           edgecriterion(edge1.squeeze(), edge_masks.squeeze().float()) + \
           cecriterion(direction, dir_masks.long()) + \
           0.25 * bcecriterion(pred2.squeeze(),
                               F.interpolate(true_masks, mode='bilinear', size=(256, 256)).squeeze().float()) + \
           0.25 * bcecriterion(pred3.squeeze(),
                               F.interpolate(true_masks, mode='bilinear', size=(128, 128)).squeeze().float()) + \
           0.25 * bcecriterion(pred4.squeeze(),
                               F.interpolate(true_masks, mode='bilinear', size=(64, 64)).squeeze().float()) + \
           0.25 * bcecriterion(pred5.squeeze(),
                               F.interpolate(true_masks, mode='bilinear', size=(32, 32)).squeeze().float()) + \
           0.25 * edgecriterion(edge2.squeeze(), F.interpolate(edge_masks.unsqueeze(1), mode='bilinear',
                                                               size=(256, 256)).squeeze().float()) + \
           0.25 * edgecriterion(edge3.squeeze(), F.interpolate(edge_masks.unsqueeze(1), mode='bilinear',
                                                               size=(128, 128)).squeeze().float()) + \
           0.25 * edgecriterion(edge4.squeeze(), F.interpolate(edge_masks.unsqueeze(1), mode='bilinear',
                                                               size=(64, 64)).squeeze().float())
    return   loss

def loss_multi(result,gt,mask_key=['true_masks','mask_mat','edge_masks','dir_masks']):

    remasks_pred, masks_pred, pred2, pred3, pred4, pred5, edge1, edge2, edge3, edge4, direction = result
    true_masks = gt[mask_key[0]]
    mask_mat = gt[mask_key[1]]

    edge_masks = mask_mat['edge_masks']
    dir_masks = mask_mat['dir_masks']



    loss =bcecriterion(masks_pred.squeeze(), torch.sigmoid(remasks_pred).squeeze().float())+ \
                      bcecriterion(remasks_pred.squeeze(), true_masks.squeeze().float())+ \
                      bcecriterion(masks_pred.squeeze(), true_masks.squeeze().float())+ \
                      edgecriterion(edge1.squeeze(), edge_masks.squeeze().float())+ \
                      cecriterion(direction,dir_masks.long())+ \
                      0.25*bcecriterion(pred2.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(256,256)).squeeze().float())+ \
                      0.25*bcecriterion(pred3.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(128,128)).squeeze().float())+ \
                      0.25*bcecriterion(pred4.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(64,64)).squeeze().float())+ \
                      0.25*bcecriterion(pred5.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(32,32)).squeeze().float())+ \
                      0.25*edgecriterion(edge2.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(256,256)).squeeze().float())+ \
                      0.25*edgecriterion(edge3.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(128,128)).squeeze().float())+ \
                      0.25*edgecriterion(edge4.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(64,64)).squeeze().float())
    return loss