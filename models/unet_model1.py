from torch import nn
from modules.blocks.blocks import DoubleConv,Down,OutConv,Up
# from modules.blocks.blocks import segnetUp3 as Up
import torch

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False,**arg):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor,self.bilinear)
        self.up2 = Up(512, 256 // factor,self.bilinear)
        self.up3 = Up(256, 128 // factor,self.bilinear)
        self.up4 = Up(128, 64,self.bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x,show=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # print("x0======max: ",x.max(),"x0=======min:",x.min())
        x = self.up4(x, x1)
        # print("x======max: ",x.max(),"x=======min:",x.min())
        if show:
            torch.save(x,show+'out1.pt')
        logits = self.outc(x)
        # print("max: ",logits.max(),"min:",logits.min())
        return logits