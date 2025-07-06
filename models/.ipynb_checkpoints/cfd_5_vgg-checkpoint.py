import torch
from torch import nn
from modules.blocks.blocks import DoubleConv, Down, Up, OutConv, Concat,Conv
import torch.nn.functional as F
from torchvision import models

class UNet_(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet_, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 64
        self.down2 = vgg16_bn.features[12:22]  # 64
        self.down3 = vgg16_bn.features[22:32]  # 64
        self.down4 = vgg16_bn.features[32:42]  # 64
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 128, bilinear)



    def forward(self, x,show=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x6 = self.up2(x, x3)
        x = self.up3(x6, x2)
        outx = self.up4(x, x1)
        return outx,x6



class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.UNet_ = UNet_(n_channels=n_channels)


        self.m1 =M(128,32,first=True,n_classes=n_classes)
        self.m2 =M(128,32,n_classes=n_classes)
        self.m3 =M(128,32,n_classes=n_classes)
        self.m4 =M(128,32,end=True,n_classes=n_classes)

        self.m5 =M0(256,32,n_classes=n_classes)


    def forward(self, x,show=False):
        size = x.shape[-2:]
        outx,x5 = self.UNet_(x)

        x,logits1 = self.m1([outx])

        x,logits2 = self.m2([outx,x])

        x,logits3 = self.m3([outx, x])

        x1,logits4 = self.m5(x5)
        x1 = F.interpolate(x1, size=size)
        logits4 = F.interpolate(logits4, size=size)

        x, logits5 = self.m4([outx, x,x1*0.01])
            
        return logits1, logits2, logits3, logits4, logits5

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

if __name__ =='__main__':
    model = UNet(3,3)
    imgs = torch.zeros((1,3,640,640))

    rs = model(imgs)

    for r in rs:
        print(r.shape)