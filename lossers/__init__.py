from inspect import isfunction,isclass
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

from .jaccard import JaccardLoss
from .dice import DiceLoss
from .focal import FocalLoss
from .lovasz import LovaszLoss
from .soft_bce import SoftBCEWithLogitsLoss
from .soft_ce import SoftCrossEntropyLoss
from .tversky import TverskyLoss
from .mcc import MCCLoss
from .seg_loss import dice_loss
from .my_loss import unet_loss_func, rdd_3_loss_func,rdd_3_loss_func_muilt,multitask_loss_func,rdd_5_loss_func, loss_cross_entropy,loss_cross_entropy_mutliclas,rdd_5_3_loss_func,rdd_5_loss_func1,unet_loss_func2,loss_cross_entropy_lovasz
from .cbrnet_loss import loss as cbrnet_loss
from .cbrnet_loss import loass_1 as cbrnet_loss_1
import functools
from .useful_loss import EdgeLoss

from .losses import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
)

loss_dic = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
    'JaccardLoss': JaccardLoss,
     'DiceLoss': DiceLoss,
     'FocalLoss': FocalLoss,
     'LovaszLoss': LovaszLoss,
    'SoftBCEWithLogitsLoss': SoftBCEWithLogitsLoss,
    'SoftCrossEntropyLoss': SoftCrossEntropyLoss,
    'TverskyLoss': TverskyLoss,
    'MCCLoss': MCCLoss,
    'dice_loss': dice_loss,
    'unet_loss': unet_loss_func,
    'rdd_3_loss': rdd_3_loss_func,
    'rdd_3_loss_func_muilt': rdd_3_loss_func_muilt,
    'multitask_loss_func': multitask_loss_func,
    'rdd_5_loss_func': rdd_5_loss_func,
    'rdd_5_3_loss_func': rdd_5_3_loss_func,
    'rdd_5_loss_func1': rdd_5_loss_func1,
    'cbrnet_loss': cbrnet_loss,
    'cbrnet_loss_1': cbrnet_loss_1,
    'edgeLoss': EdgeLoss,
    'loss_cross_entropy': loss_cross_entropy,
    'unet_loss_func2':unet_loss_func2,
    'loss_cross_entropy_mutliclas': loss_cross_entropy_mutliclas,
    'loss_cross_entropy_lovasz':loss_cross_entropy_lovasz
}


def get_loss_function(cfg):
    if cfg is None:
        return cross_entropy2d

    else:
        loss_dict = cfg
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in loss_dic:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        loss = loss_dic[loss_name]
        print("init loss :", loss_name," params:", loss_params)
        if isfunction(loss):
            return functools.partial(loss, **loss_params)
        if isclass(loss):
            return loss(**loss_params)

