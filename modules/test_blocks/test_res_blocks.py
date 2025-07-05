from torch import nn
from backbone_airchitecture.blocks.res_blocks import Bottle2neck,Bottleneck,BasicBlock



from  timm.models.res2net import res2next50
import torch
from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS, flop_count, FlopCountAnalysis
# model = res2next50()

imgs=torch.ones((1,64,512,512))

# # r= model(imgs)
# flop_dict2, _ = flop_count(model, (imgs,))
# print(flop_dict2)
# print(_DEFAULT_SUPPORTED_OPS)


b = Bottle2neck(64,16,norm_layer=nn.BatchNorm2d)
print(b(imgs).shape)

b = Bottleneck(64,16,norm_layer=nn.BatchNorm2d)
print(b(imgs).shape)

b = BasicBlock(64,64,norm_layer=nn.BatchNorm2d)
print(b(imgs).shape)
