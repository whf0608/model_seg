from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.blocks.blocks import DoubleConv,Down,Up,OutConv

# class fix_seg(nn.Module):
#     def __init__(self):
#         super(fix_seg, self).__init__()
#         self.conv0=nn.Conv2d(1,8,3,1,1,bias=False)
#         self.conv0.weight = nn.Parameter(torch.tensor([[[[0,0, 0], [1, 0, 0], [0, 0, 0]]],
#                                                        [[[1,0, 0], [0, 0, 0], [0, 0, 0]]],
#                                                        [[[0,1, 0], [0, 0, 0], [0, 0, 0]]],
#                                                        [[[0,0, 1], [0, 0, 0], [0, 0, 0]]],
#                                                        [[[0,0, 0], [0, 0, 1], [0, 0, 0]]],
#                                                        [[[0,0, 0], [0, 0, 0], [0, 0, 1]]],
#                                                        [[[0,0, 0], [0, 0, 0], [0, 1, 0]]],
#                                                        [[[0,0, 0], [0, 0, 0], [1, 0, 0]]]]).float())
#     def forward(self,direc_pred,masks_pred,edge_pred):
#         direc_pred=direc_pred.softmax(1)
#         edge_mask=1*(torch.sigmoid(edge_pred).detach()>0.5)
#         refined_mask_pred=(self.conv0(masks_pred)*direc_pred).sum(1).unsqueeze(1)*edge_mask+masks_pred*(1-edge_mask)
#         return refined_mask_pred

class CBRNet(nn.Module):
    def __init__(self,n_channels=3, n_classes=2, bilinear=False,**arg):
        super(CBRNet, self).__init__()
        self.n_classes = n_classes
        bilinear = True
        state=[64,128,256,512,1024]
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 64
        self.down2 = vgg16_bn.features[12:22]  # 64
        self.down3 = vgg16_bn.features[22:32]  # 64
        self.down4 = vgg16_bn.features[32:42]  # 64
        del vgg16_bn
        self.bcecriterion = nn.BCEWithLogitsLoss()
        self.edgecriterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.85))
        self.cecriterion=nn.CrossEntropyLoss(ignore_index=-1)
        factor = 2 if bilinear else 1
        self.up1 = Up(state[4],state[3] // factor, bilinear)
        # self.up2 = Up(state[3],state[2] // factor, bilinear)
        # self.up3 = Up(state[2],state[1] // factor, bilinear)
        self.up4 = Up(state[3],state[0], bilinear)
        self.interpo=nn.Upsample(scale_factor=2, mode='bilinear')
        self.dir_head = nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))

    def forward(self, x,gts=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        # x = self.up2(x,  x3)
        # x = self.up3(x, x2)
        x = self.up4(x, x1)
   
        direction=self.dir_head(x)
        
        return direction
