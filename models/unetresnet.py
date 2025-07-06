import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.blocks.blocks import DoubleConv,Down,Up,OutConv
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self,n_channels=3, n_classes=2, bilinear=False,**arg):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        state=[64,128,256,512]
        bilinear = True
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        factor = 2 if bilinear else 1
        self.up1 = Up(state[3]+state[2],state[2], bilinear)
        self.up2 = Up(state[2]+state[1],state[1], bilinear)
        self.up3 = Up(state[1]+state[0],state[0], bilinear)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = Up(state[0]+state[0],state[0], bilinear)
        self.head = nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        
    def forward(self, x,gts=None):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x1 = self.up(x1)
        x = self.up4(x,x1)
        x1 = self.up(x1)
        x = self.up4(x,x1)
        direction=self.head(x)
        return direction,direction