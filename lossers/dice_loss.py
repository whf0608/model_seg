# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


# @weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


# @LOSSES.register_module()
class DiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .focal_loss import BinaryFocalLoss


def make_one_hot(input, num_classes=None):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Shapes:
        predict: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with predict
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    if num_classes is None:
        num_classes = input.max() + 1
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu().long(), 1)
    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_dice = False  # treat a large map when True
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target, use_sigmoid=True):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        output: A tensor of shape [N, C, *]
        target: A tensor of same shape with output
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

    def forward(self, output, target):
        assert output.shape == target.shape, 'output & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        output = F.softmax(output, dim=1)
        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = dice(output[:, i], target[:, i], use_sigmoid=False)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += (dice_loss)
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss


class WBCEWithLogitLoss(nn.Module):
    """
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1.
    To decrease the number of false positives, set β<1.
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, weight=1.0, ignore_index=None, reduction='mean'):
        super(WBCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        weight = float(weight)
        self.weight = weight
        self.reduction = reduction
        self.smooth = 0.01

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = torch.sigmoid(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

        # loss = self.bce(output, target)
        loss = -self.weight * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


class WBCE_DiceLoss(nn.Module):
    def __init__(self, alpha=1.0, weight=1.0, ignore_index=None, reduction='mean'):
        """
        combination of Weight Binary Cross Entropy and Binary Dice Loss
        Args:
            @param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            @param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            @param alpha: weight between WBCE('Weight Binary Cross Entropy') and binary dice, apply on WBCE
        Shapes:
            output: A tensor of shape [N, *] without sigmoid activation function applied
            target: A tensor of shape same with output
        """
        super(WBCE_DiceLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        assert 0 <= alpha <= 1, '`alpha` should in [0,1]'
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dice = BinaryDiceLoss(ignore_index=ignore_index, reduction=reduction, general=True)
        self.wbce = WBCEWithLogitLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.dice_loss = None
        self.wbce_loss = None

    def forward(self, output, target):
        self.dice_loss = self.dice(output, target)
        self.dice_loss = -torch.log(1 - self.dice_loss)
        self.wbce_loss = self.wbce(output, target)
        loss = self.alpha * self.wbce_loss + self.dice_loss
        return loss


class Binary_Focal_Dice(nn.Module):
    def __init__(self, **kwargs):
        super(Binary_Focal_Dice, self).__init__()
        self.dice = BinaryDiceLoss(**kwargs)
        self.focal = BinaryFocalLoss(**kwargs)

    def forward(self, logits, target):
        dice_loss = self.dice(logits, target)
        dice_loss = -torch.log(1 - dice_loss)
        focal_loss = self.focal(logits, target)
        loss = dice_loss + focal_loss
        return loss, (dice_loss.detach(), focal_loss.detach())


def test():
    input = torch.rand((3, 1, 32, 32, 32))
    model = nn.Conv3d(1, 4, 3, padding=1)
    target = torch.randint(0, 4, (3, 1, 32, 32, 32)).float()
    target = make_one_hot(target, num_classes=4)
    criterion = DiceLoss(ignore_index=[2, 3], reduction='mean')
    loss = criterion(model(input), target)
    loss.backward()
    print(loss.item())

    # input = torch.zeros((1, 2, 32, 32, 32))
    # input[:, 0, ...] = 1
    # target = torch.ones((1, 1, 32, 32, 32)).long()
    # target_one_hot = make_one_hot(target, num_classes=2)
    # # print(target_one_hot.size())
    # criterion = DiceLoss()
    # loss = criterion(input, target_one_hot)
    # print(loss.item())


if __name__ == '__main__':
    test()


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
   """
   net_output must be (b, c, x, y(, z)))
   gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
   if mask is provided it must have shape (b, 1, x, y(, z)))
   :param net_output:
   :param gt:
   :param axes:
   :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
   :param square: if True then fp, tp and fn will be squared before summation
   :return:
   """
   if axes is None:
       axes = tuple(range(2, len(net_output.size())))

   shp_x = net_output.shape
   shp_y = gt.shape

   with torch.no_grad():
       if len(shp_x) != len(shp_y):
           gt = gt.view((shp_y[0], 1, *shp_y[1:]))

       if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
           # if this is the case then gt is probably already a one hot encoding
           y_onehot = gt
       else:
           gt = gt.long()
           y_onehot = torch.zeros(shp_x)
           if net_output.device.type == "cuda":
               y_onehot = y_onehot.cuda(net_output.device.index)
           y_onehot.scatter_(1, gt, 1)

   tp = net_output * y_onehot
   fp = net_output * (1 - y_onehot)
   fn = (1 - net_output) * y_onehot

   if mask is not None:
       tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
       fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
       fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

   if square:
       tp = tp ** 2
       fp = fp ** 2
       fn = fn ** 2

   tp = sum_tensor(tp, axes, keepdim=False)
   fp = sum_tensor(fp, axes, keepdim=False)
   fn = sum_tensor(fn, axes, keepdim=False)

   return tp, fp, fn


class SoftDiceLoss(nn.Module):
   def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                square=False):
       """
       paper: https://arxiv.org/pdf/1606.04797.pdf
       """
       super(SoftDiceLoss, self).__init__()

       self.square = square
       self.do_bg = do_bg
       self.batch_dice = batch_dice
       self.apply_nonlin = apply_nonlin
       self.smooth = smooth

   def forward(self, x, y, loss_mask=None):
       shp_x = x.shape

       if self.batch_dice:
           axes = [0] + list(range(2, len(shp_x)))
       else:
           axes = list(range(2, len(shp_x)))

       if self.apply_nonlin is not None:
           x = self.apply_nonlin(x)

       tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

       dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

       if not self.do_bg:
           if self.batch_dice:
               dc = dc[1:]
           else:
               dc = dc[:, 1:]
       dc = dc.mean()

       return -dc