from torch import nn
from modules.blocks.blocks import DoubleConv, Down, Up, OutConv, Concat,Conv
from modules.blocks.blocks import Up_same as Ups
import torch.nn.functional as F
import torch
import sys
try:
    from .modeling.image_encoder import ImageEncoderViT
except:
    print("no loading vit")

class Model(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False,**arg):
        super(Model, self).__init__()
        self.feature = ImageEncoderViT(embed_dim=1280,depth=32,num_heads=16,global_attn_indexes=[7,15,23,31],img_size=1024,qkv_bias=True,use_rel_pos=True,window_size=14,out_chans=256)

        factor = 2 if bilinear else 1
        self.cov3 = Conv(256,128)
        self.up3 = Ups(256, 128 // factor, bilinear,input_same=True)
        # self.cov4 = Conv(128,64)
        # self.up4 = Up(128, 64, bilinear, input_same=True)
        # self.outc = OutConv(64, n_classes)
        
        
        self.m1 =M(128,32,first=True,n_classes=n_classes)
        self.m2 =M(128,32,n_classes=n_classes)
        self.m3 =M(128,32,n_classes=n_classes)
        self.m4 =M(128,32,end=True,n_classes=n_classes)

        self.m5 =M0(128,32,n_classes=n_classes)

        self.sam_checkpoint = sam_checkpoint
        if self.sam_checkpoint:
            self.param()

    def param(self):
        sam = torch.load(self.sam_checkpoint, map_location='cpu')
        self.feature.load_state_dict(sam)
        for param in self.feature.parameters():
            param.requires_grad = False

    def forward(self, x, show=None):
        size0 = x.shape[-2:]
        with torch.no_grad():
            x = self.feature(x)
        
        x_ = self.cov3(x)
        outx = self.up3(x, x_)
        size = outx.shape[-2:]
        x,logits1 = self.m1([outx])
        x,logits2 = self.m2([outx,x])
        x,logits3 = self.m3([outx, x])
        x1,logits4 = self.m5(outx)
        x1 = F.interpolate(x1, size=size)
        x, logits5 = self.m4([outx, x,x1])
        
        logits1 = F.interpolate(logits1, size=size0)
        logits2 = F.interpolate(logits2, size=size0)
        logits3 = F.interpolate(logits3, size=size0)
        logits4 = F.interpolate(logits4, size=size0)
        logits5 = F.interpolate(logits5, size=size0)
        return logits1,logits2,logits3,logits4,logits5
    
    
class M0(nn.Module):
    def __init__(self,in_channels, out_channels,n_classes, first=False,end=False):
        super(M0, self).__init__()

        c = 32
        self.out = Conv(in_channels, c)

        self.cov = Conv(c, out_channels)
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x):
        x = self.out(x)
        x1 = self.cov(x)
        logits = self.outc(x1)
        return x, logits


class M(nn.Module):
    def __init__(self,in_channels, out_channels,n_classes, first=False,end=False):
        super(M, self).__init__()

        self.out = Conv(in_channels, out_channels)
        c0 = out_channels if first else out_channels*2
        c = out_channels*3 if end else c0
        self.concat = Concat()

        self.cov = Conv(c, out_channels)
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, xs):
        x = self.out(xs[0])

        xs0 = [x]
        for _ in range(1,len(xs)):
            xs0.append(xs[_])
        x1 = self.cov(self.concat(xs0))
        logits2 = self.outc(x1)

        return x, logits2
