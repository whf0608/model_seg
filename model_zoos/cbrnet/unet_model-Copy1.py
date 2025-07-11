from .unet_parts import *
from torchvision import models
class fix_seg(nn.Module):
    def __init__(self):
        super(fix_seg, self).__init__()
        self.conv0=nn.Conv2d(1,8,3,1,1,bias=False)
        self.conv0.weight = nn.Parameter(torch.tensor([[[[0,0, 0], [1, 0, 0], [0, 0, 0]]],
                                                       [[[1,0, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0,1, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0,0, 1], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0,0, 0], [0, 0, 1], [0, 0, 0]]],
                                                       [[[0,0, 0], [0, 0, 0], [0, 0, 1]]],
                                                       [[[0,0, 0], [0, 0, 0], [0, 1, 0]]],
                                                       [[[0,0, 0], [0, 0, 0], [1, 0, 0]]]]).float())
    def forward(self,direc_pred,masks_pred,edge_pred):
        direc_pred=direc_pred.softmax(1)
        edge_mask=1*(torch.sigmoid(edge_pred).detach()>0.5)
        refined_mask_pred=(self.conv0(masks_pred)*direc_pred).sum(1).unsqueeze(1)*edge_mask+masks_pred*(1-edge_mask)
        return refined_mask_pred
class CBRNet(nn.Module):
    def __init__(self,  nc=3, n_classes = 3):
        super(CBRNet, self).__init__()
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
        self.classifier1=nn.Sequential(nn.Conv2d(512,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.up1 = Up(state[4],state[3] // factor, bilinear)
        self.classifier2=nn.Sequential(nn.Conv2d(256+1,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.classifier2_2=nn.Sequential(nn.Conv2d(256,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.up2 = Up(state[3],state[2] // factor, bilinear)
        self.classifier3=nn.Sequential(nn.Conv2d(128+1,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.classifier3_2=nn.Sequential(nn.Conv2d(128,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.up3 = Up(state[2],state[1] // factor, bilinear)
        self.classifier4=nn.Sequential(nn.Conv2d(64+1,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.classifier4_2=nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.up4 = Up(state[1],state[0], bilinear)
        self.classifier5=nn.Sequential(nn.Conv2d(64+1,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.classifier5_2=nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,n_classes,1,1))
        self.interpo=nn.Upsample(scale_factor=2, mode='bilinear')
        self.dir_head = nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,8*n_classes,1,1))
        # self.conv = nn.Conv2d(64, n_classes, 3, 1, 1, bias=False)
        self.fixer=fix_seg()

    def forward(self, x,gts=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        seg5=self.classifier1(x5)
        x = self.up1(x5, x4)
        edge4=self.classifier2_2(x)
        seg4=self.classifier2(torch.cat((x,self.interpo(seg5)),1))
        x = self.up2(x,  x3)
        edge3=self.classifier3_2(x)
        seg3=self.classifier3(torch.cat((x,self.interpo(seg4)),1))
        x = self.up3(x, x2)
        edge2=self.classifier4_2(x)
        seg2=self.classifier4(torch.cat((x,self.interpo(seg3)),1))
        x = self.up4(x, x1)
        edge1=self.classifier5_2(x)
        seg1=self.classifier5(torch.cat((x,self.interpo(seg2)),1))
        direction=self.dir_head(x)
        if self.training:
            return self.conv(x)
        else:
            r_x=self.fixer(direction,seg1,edge1)
            return r_x,seg1,seg2,seg3,seg4,seg5,edge1,edge2,edge3,edge4,direction
