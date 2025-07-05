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

        self.out1 = DoubleConv(128, 32,mid_channels=64)
        self.outc = OutConv(32, n_classes)

        self.out2 = DoubleConv(128, 32,mid_channels=64)
        self.concat = Concat()
        self.cov2 = Conv(64,32)
        self.outc2 = OutConv(32, n_classes)

        self.out3 = DoubleConv(128, 32,mid_channels=64)
        self.cov3 = Conv(64, 32)
        self.outc3 = OutConv(32, n_classes)
        
        self.out4 = DoubleConv(128, 32,mid_channels=64)
        self.cov4 = Conv(64, 32)
        self.outc4 = OutConv(32, n_classes)
        
        self.out5 = DoubleConv(128, 32,mid_channels=64)
        self.cov5 = Conv(64, 32)
        self.outc5 = OutConv(32, n_classes)
        
        self.blocks = nn.Sequential(Conv(32,32),Conv(32,32),Conv(32,32),Conv(32,32))
        
        
        
    def forward(self, x, show=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        outx = self.up4(x, x1)
        x = self.out1(outx)

        logits = self.outc(x)

        if show is not None:
            torch.save(x, show +'out1.pt')
            
        x_ = self.out2(outx)
        x_ = self.blocks(x_)
        if show is not None:
            torch.save(x_, show +'out2.pt')
            
        x = self.cov2(self.concat([x_, x]))
       
        logits2 = self.outc2(x)
        
        x_ = self.out3(outx)
        x_ = self.blocks(x_)
        if show is not None:
            torch.save(x_, show +'out3.pt')
        
        x = self.cov3(self.concat([x_, x]))
       
        logits3 = self.outc3(x)
        
        x_ = self.out4(outx)
        x_ = self.blocks(x_)
        if show is not None:
            torch.save(x_, show +'out4.pt')
        
        x = self.cov4(self.concat([x_, x]))
       
        logits4 = self.outc4(x)
        
        x_ = self.out5(outx)
        x_ = self.blocks(x_)
        if show is not None:
            torch.save(x_, show +'out5.pt')
        
        x = self.cov5(self.concat([x_, x]))
       
        logits5 = self.outc5(x)
        
        return logits, logits2, logits3,logits4,logits5