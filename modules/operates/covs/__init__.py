from torch import nn
from .ghostcov import GhostConv
from .dwconv import DWConv
from .conv2dpadding import Conv2dStaticSamePadding,Conv2dDynamicSamePadding
from .cov_base import Conv
from .dorefaconv2d import DorefaConv2d,BNFold_DorefaConv2d,BNFold_Conv2d_Q,QuantizedConv2d
from .mixconv2d import MixConv2dv1,MixConv2dv2
from .separableconv import SeparableConv2d


