import torch
from torch import nn
from modules.blocks.blocks import DoubleConv, Down, Up, OutConv, Concat,Conv
import torch.nn.functional as F


class UNet_(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet_, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
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
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        outx = self.up4(x, x1)
        return outx
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.UNet_ = UNet_(n_channels=n_channels)

        self.m1 =M(128,32,first=True,n_classes=n_classes)
        self.m2 =M(128,32,n_classes=n_classes)
        self.m3 =M(128,32,n_classes=n_classes)


    def forward(self, x,show=False):

        outx = self.UNet_(x)

        x,logits1 = self.m1([outx])

        x,logits2 = self.m2([outx,x])

        x,logits3 = self.m3([outx, x])

    
        return logits1, logits2, logits3

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
        x = F.interpolate(x, size=(640, 640))
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