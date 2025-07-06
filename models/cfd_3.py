import torch
from torch import nn
from modules.blocks.blocks import DoubleConv, Down, Up, OutConv, Concat,Conv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
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
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 128, bilinear)
        
        self.concat = Concat()
        
        self.out1 = DoubleConv(256, 64)
        self.outc1 = OutConv(64, n_classes)
        self.cov1 = Conv(256+64,256)
        
        
        self.out2 = DoubleConv(128, 64)
        
        self.outc2 = OutConv(64, n_classes)
        self.cov2 = Conv(128+64,128)
            
            
        self.out3 = DoubleConv(128, 64)
        self.outc3 = OutConv(64, n_classes)
        
        
    def forward(self, x, show=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x_ = self.out1(x)
        logits1 = self.outc1(x_)
        x = self.cov1(self.concat([x_, x]))
        x = self.up3(x, x2)
        x_ = self.out2(x)
        logits2 = self.outc2(x_)
        x = self.cov2(self.concat([x_, x]))
        
        x= self.up4(x, x1)
        x = self.out3(x)
        logits3 = self.outc3(x)

        
        return logits1, logits2, logits3