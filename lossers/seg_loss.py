import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weight_reduce_loss
import torch
import torch.nn.functional as F





from torch import optim
from tqdm import tqdm
from torch import nn
from torch import Tensor
import torch.nn.functional as F
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]






def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def hed_loss(pred,
             label,
             weight=None,
             reduction='mean',
             avg_factor=None,
             class_weight=None):
    """Calculate the binary CrossEntropy loss with weights.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
    Returns:
        torch.Tensor: The calculated loss
    """
    if weight is not None:
        weight = weight.float()

    total_loss = 0
    label = label.unsqueeze(1)
    batch, channel_num, imh, imw = pred.shape
    for b_i in range(batch):
        p = pred[b_i, :, :, :].unsqueeze(1)
        t = label[b_i, :, :, :].unsqueeze(1)
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        class_weight = torch.zeros_like(mask)
        class_weight[t > 0.5] = num_neg / (num_pos + num_neg)
        class_weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # weighted element-wise losses
        loss = F.binary_cross_entropy(p, t.float(), weight=class_weight, reduction='none')
        # do the reduction for the weighted loss
        #loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
        loss = torch.sum(loss)
        total_loss = total_loss + loss

    return total_loss


class HEDLoss(nn.Module):
    """HEDLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(HEDLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.cls_criterion = hed_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:

            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls




def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask, reduction='sum')

    return cost






def hed_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.1).float()).float()
    num_negative = torch.sum((mask <= 0.).float()).float()

    mask[mask > 0.1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), targets.float())

    return l_weight*torch.sum(cost)


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost

def bdcn_lossORI(inputs, targets, l_weigts=1.1,cuda=False):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    # print(cuda)
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * 1.1 / valid  # balance = 1.1
    weights = torch.Tensor(weights)
    # if cuda:
    weights = weights.cuda()
    inputs = torch.sigmoid(inputs)
    loss = torch.nn.BCELoss(weights, reduction='sum')(inputs.float(), targets.float())
    return l_weigts*loss

def rcf_loss(inputs, label):

    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0.5).float()).float() # ==1.
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0.
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())

    return 1.*torch.sum(cost)

# ------------ cats losses ----------

def bdrloss(prediction, label, radius,device='cpu'):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)
    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()



def textureloss(prediction, label, mask_radius, device='cpu'):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss)


def cats_loss(prediction, label, l_weight=[0.,0.], device='cpu'):
    # tracingLoss
    tex_factor,bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    prediction = torch.sigmoid(prediction)
    # print('bce')
    cost = torch.sum(torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False))
    label_w = (label != 0).float()
    # print('tex')
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

    return cost + bdr_factor * bdrcost + tex_factor * textcost

def cross_entropy_loss2d(inputs, targets, cuda=False, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in xrange(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    inputs = F.sigmoid(inputs)
    loss = nn.BCELoss(weights, size_average=False)(inputs, targets)
    return loss

def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import dice

def clip_by_value(t, t_min, t_max):
    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

def attention_loss2(output,target):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0
    eps = 1e-14
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

    weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
             (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
    weight=weight.detach()

    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.sum(loss)
    return loss


class AttentionLoss2(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        total_loss = 0
        for i in range(len(output)):
            o = output[i]
            l = label[i]
            loss_focal = attention_loss2(o, l)
            total_loss = total_loss + loss_focal
        total_loss = total_loss / batch_size
        return total_loss


class AttentionLossSingleMap(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLossSingleMap, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        loss_focal = attention_loss2(output, label)
        total_loss = loss_focal / batch_size
        return total_loss


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss2d(inputs, targets, cuda=False, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    #loss = nn.BCELoss(weights, size_average=False)(inputs, targets)
    loss = nn.BCELoss(weights, reduction='sum')(inputs, targets)
    return loss

def bdcn_loss(prediction, label, cuda=False, balance=1.1):
    total_loss = 0
    b, c, w, h = label.shape
    for j in range(c):
        p = prediction[:, j, :, :].unsqueeze(1)
        l = label[:, j, :, :].unsqueeze(1)
        loss = cross_entropy_loss2d(p, l, cuda, balance)
        total_loss = total_loss + loss

    total_loss = total_loss / b * 1.0
    return total_loss

def bdcn_loss_edge(prediction, label, cuda=False, balance=1.1):
    b, c, w, h = label.shape
    loss = cross_entropy_loss2d(prediction, label, cuda, balance)
    total_loss = loss / b * 1.0
    return total_loss



"""Calculate Multi-label Loss (Semantic Loss)"""
import torch
from torch.nn.modules.loss import _Loss

torch_ver = torch.__version__[:3]

__all__ = ['EdgeDetectionReweightedLosses', 'EdgeDetectionReweightedLosses_CPU']


class WeightedCrossEntropyWithLogits(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(WeightedCrossEntropyWithLogits, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        loss_total = 0
        for i in range(targets.size(0)): # iterate for batch size
            pred = inputs[i]
            target = targets[i]

            for j in range(pred.size(0)):
                p = pred[j]
                t = target[j]

                num_pos = torch.sum(t == 1).float()
                num_neg = torch.sum(t == 0).float()
                num_total = num_neg + num_pos  # true total number
                pos_weight = (num_neg / num_pos).clamp(min=1, max=num_total)  # compute a pos_weight for each image

                max_val = (-p).clamp(min=0)
                log_weight = 1 + (pos_weight - 1) * t
                loss = p - p * t + log_weight * (
                            max_val + ((-max_val).exp() + (-p - max_val).exp()).log())

                loss = torch.sum(loss)
                loss_total = loss_total + loss

        loss_total = loss_total / targets.size(0)
        return loss_total

class EdgeDetectionReweightedLosses(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pre, target = tuple(inputs)
        side5, fuse = tuple(pre)

        loss_side5 = super(EdgeDetectionReweightedLosses, self).forward(side5, target)
        loss_fuse = super(EdgeDetectionReweightedLosses, self).forward(fuse, target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss

class EdgeDetectionReweightedLosses_CPU(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    """CPU version used to dubug"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses_CPU, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pred, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedLosses_CPU, self).forward(pred[0], target)
        loss_fuse = super(EdgeDetectionReweightedLosses_CPU, self).forward(pred[1], target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss


##########
class WeightedCrossEntropyWithLogitsSingle(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(WeightedCrossEntropyWithLogitsSingle, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        loss_total = 0
        for i in range(targets.size(0)): # iterate for batch size
            pred = inputs[i]
            target = targets[i]
            #print(pred.shape, target.shape)

            num_pos = torch.sum(target == 1).float()
            num_neg = torch.sum(target == 0).float()
            num_total = num_neg + num_pos  # true total number
            pos_weight = (num_neg / num_pos).clamp(min=1, max=num_total) # compute a pos_weight for each image

            max_val = (-pred).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target
            loss = pred - pred * target + log_weight * (max_val + ((-max_val).exp() + (-pred - max_val).exp()).log())

            loss = loss.sum()
            loss_total = loss_total + loss

        loss_total = loss_total / targets.size(0)
        return loss_total

class EdgeDetectionReweightedLossesSingle(WeightedCrossEntropyWithLogitsSingle):
    """docstring for EdgeDetectionReweightedLosses"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLossesSingle, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pre, target = tuple(inputs)
        side5, fuse = tuple(pre)

        #print(side5.shape, fuse.shape, target.shape)

        loss_side5 = super(EdgeDetectionReweightedLossesSingle, self).forward(side5, target)
        loss_fuse = super(EdgeDetectionReweightedLossesSingle, self).forward(fuse, target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss

class EdgeDetectionReweightedLossesSingle_CPU(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    """CPU version used to dubug"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLossesSingle_CPU, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pred, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedLossesSingle_CPU, self).forward(pred[0], target)
        loss_fuse = super(EdgeDetectionReweightedLossesSingle_CPU, self).forward(pred[1], target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss


from typing import Tuple
from torch import nn, Tensor
import torch
from torch import nn
import torch.nn.functional as F

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:,1,:,:].unsqueeze(1))
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss

class LabelCircleLossModel(nn.Module):
    def __init__(self, num_classes, m=0.35, gamma=30, feature_dim=192):
        super(LabelCircleLossModel, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.weight = torch.nn.Parameter(torch.randn(feature_dim, num_classes, requires_grad=True))
        self.labels = torch.tensor([x for x in range(num_classes)])
        self.classes = num_classes
        self.init_weights()
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin
        self.loss = nn.CrossEntropyLoss()
    def init_weights(self, pretrained=None):
        self.weight.data.normal_()

    def _forward_train(self, feat, label):
        normed_feat = torch.nn.functional.normalize(feat)
        normed_weight = torch.nn.functional.normalize(self.weight,dim=0)

        bs = label.size(0)
        mask = label.expand(self.classes, bs).t().eq(self.labels.expand(bs,self.classes)).float()
        y_true = torch.zeros((bs,self.classes)).scatter_(1,label.view(-1,1),1)
        y_pred = torch.mm(normed_feat,normed_weight)
        y_pred = y_pred.clamp(-1,1)
        sp = y_pred[mask == 1]
        sn = y_pred[mask == 0]

        alpha_p = (self.O_p - y_pred.detach()).clamp(min=0)
        alpha_n = (y_pred.detach() - self.O_n).clamp(min=0)

        y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
                    (1-y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
        loss = self.loss(y_pred,label)

        return loss, sp, sn

    def forward(self, input, label,  mode='train'):
            if mode == 'train':
                return self._forward_train(input, label)
            elif mode == 'val':
                raise KeyError

class CircleLoss2(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss2, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        m = labels.size(0)
        mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import dice

def clip_by_value(t, t_min, t_max):
    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

def attention_loss2(output,target):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0
    eps = 1e-14
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

    weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
             (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
    weight=weight.detach()

    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.sum(loss)
    return loss


class AttentionLoss2(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        total_loss = 0
        for i in range(len(output)):
            o = output[i]
            l = label[i]
            loss_focal = attention_loss2(o, l)
            total_loss = total_loss + loss_focal
        total_loss = total_loss / batch_size
        return total_loss


class AttentionLossSingleMap(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLossSingleMap, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        loss_focal = attention_loss2(output, label)
        total_loss = loss_focal / batch_size
        return total_loss

import torch
import torch.nn as nn


def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    total_loss = 0
    batch, channel_num, imh, imw = edges.shape
    for b_i in range(batch):
        p = preds[b_i, :, :, :].unsqueeze(1)
        t = edges[b_i, :, :, :].unsqueeze(1)
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight[t > 0.5] = num_neg / (num_pos + num_neg)
        weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # Calculate loss.
        loss = torch.nn.functional.binary_cross_entropy(p.float(), t.float(), weight=weight, reduction='none')
        loss = torch.sum(loss)
        total_loss = total_loss + loss
    return total_loss

def weighted_cross_entropy_loss2(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    num_pos = torch.sum(edges == 1).float()
    num_neg = torch.sum(edges == 0).float()
    weight = torch.zeros_like(edges)
    mask = (edges > 0.5).float()
    weight[edges > 0.5] = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    loss = torch.nn.functional.binary_cross_entropy(preds, edges, weight=weight, reduction='none')
    loss = torch.sum(loss)
    return loss

def weighted_cross_entropy_loss3(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    total_loss = 0
    batch, channel_num, imh, imw = edges.shape
    for b_i in range(batch):
        p = preds[b_i, :, :, :].unsqueeze(0)
        t = edges[b_i, :, :, :].unsqueeze(0)
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight[t > 0.5] = num_neg / (num_pos + num_neg)
        weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # Calculate loss.
        loss = torch.nn.functional.binary_cross_entropy(p.float(), t.float(), weight=weight, reduction='none')
        loss = torch.sum(loss)
        total_loss = total_loss + loss
    return total_loss


class HED_Loss(nn.Module):
    def __init__(self):
        super(HED_Loss, self).__init__()

    def forward(self,output,label):#,depth_gad):
        """
        output: [N,4,H,W]
        label: [N,4,H,W]
        """
        total_loss = 0
        b,c,w,h = label.shape
        for j in range(c):
            p = output[:, j, :, :].unsqueeze(1)
            t = label[:, j, :, :].unsqueeze(1)
            loss = weighted_cross_entropy_loss(p, t)
            total_loss = total_loss + loss

        total_loss=total_loss/b*1.0
        return total_loss

class HED_Loss2(nn.Module):
    def __init__(self):
        super(HED_Loss2, self).__init__()

    def forward(self,output,label):#,depth_gad):
        """
        output: [N,4,H,W]
        label: [N,4,H,W]
        """
        b, c, w, h = label.shape
        loss = weighted_cross_entropy_loss2(output, label)
        total_loss = loss / b * 1.0
        return total_loss

class HED_Loss3(nn.Module):
    def __init__(self):
        super(HED_Loss3, self).__init__()

    def forward(self,output,label):#,depth_gad):
        """
        output: [N,4,H,W]
        label: [N,4,H,W]
        """
        b, c, w, h = label.shape
        loss = weighted_cross_entropy_loss3(output, label)
        total_loss = loss / b * 1.0
        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
def clip_by_value(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """

    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'focal2':
            return self.FocalLoss2
        elif mode == 'attention':
            print('attention loss')
            return self.AttentionLoss
        else:
            raise NotImplementedError

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='sum')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        loss /= n

        return loss

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss