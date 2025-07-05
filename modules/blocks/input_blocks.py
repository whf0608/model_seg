from torch import nn
import torch
import torch.nn.functional as F
from .fusionblocks import Focus
from .blocks import Conv

class MultiKernelInput(nn.Module):
    def __init__(self,c1=3,c2=64):   ### X (1,3,640,640)
        super(MultiKernelInput, self).__init__()
        self.cov1x1_2 = Conv(c1, c2 // 8, k=1, s=2)
        self.cov3x3_2 = Conv(c1, c2 // 8, k=3, s=2)
        self.cov5x5_2 = Conv(c1, c2 // 8, k=5, s=2)
        self.cov7x7_2 = Conv(c1, c2 // 8, k=7, s=2)

        self.cov3x3_1 = Conv(c2 // 2, c2, k=3, s=1)

    def forward(self,x):
        r1 = self.cov1x1_2(x)
        r2 = self.cov3x3_2(x)
        r3 = self.cov5x5_2(x)

        r4 = self.cov7x7_2(x)
        x = torch.cat([r1, r2, r3, r4], 1)
        x = self.cov3x3_1(x)
        return x

class MultiKernelConv(nn.Module):
    def __init__(self,c1, c2):
        super(MultiKernelConv, self).__init__()
        self.cov1x1_2 = Conv(c1,c2//8,k=1,s=2)
        self.cov3x3_2 = Conv(c1,c2//8,k=3,s=2)
        self.cov5x5_2 = Conv(c1,c2//8,k=5,s=2)
        self.cov7x7_2 = Conv(c1,c2//8,k=7,s=2)

        self.cov3x3_1 = Conv(c2//2,c2,k=3,s=1)

    def forward(self,x):
        r1 = self.cov1x1_2(x)
        r2 = self.cov3x3_2(x)
        r3 = self.cov5x5_2(x)

        r4 = self.cov7x7_2(x)
        x = torch.cat([r1, r2, r3, r4], 1)
        x = self.cov3x3_1(x)

        return x


"""
1x1 
    3x3
        3x3
            3x3
"""
class MultiConv(nn.Module):
    def __init__(self,c1, c2):
        super(MultiConv, self).__init__()
        self.cov1x1_2 = Conv(c1,c2//8,k=1,s=2)
        self.cov3x3_1 = Conv(c1,c2//8,k=3,s=2)
        self.cov3x3_2 = Conv(c1,c2//8,k=3,s=2)
        self.cov3x3_3 = Conv(c1,c2//8,k=3,s=2)

        self.cov3x3_4 = Conv(c2//2,c2,k=3,s=1)

    def forward(self,x):
        r1 = self.cov1x1_2(x)
        r2 = self.cov3x3_1(r1)
        r3 = self.cov3x3_2(r2)
        r4 = self.cov3x3_3(r3)
        x = torch.cat([r1, r2, r3, r4], 1)
        x = self.cov3x3_4(x)

        return x


class MultiPool(nn.Module):
    def __init__(self,c1, c2):
        super(MultiPool, self).__init__()
        self.p = nn.MaxPool2d(3, 2, 1)
        self.cov1x1_1 = Conv(c1,c2//8, k=1, s=1)
        self.cov1x1_2 = Conv(c1,c2//8, k=1, s=1)
        self.cov1x1_3 = Conv(c1,c2//8, k=1, s=1)

        self.cov3x3_2 = Conv(c1,c2//8, k=3, s=2)
        self.cov3x3_1 = Conv(c2//2,c2, k=3, s=1)

    def forward(self,x):
        r1 = self.p(x)
        r2 = self.p(r1)
        r3 = self.p(r2)

        r1 = self.cov1x1_1(r1)
        r4 = self.cov3x3_2(x)

        r2 = F.interpolate(r2, r1.shape[2:])
        r3 = F.interpolate(r3, r1.shape[2:])

        r2 = self.cov1x1_2(r2)
        r3 = self.cov1x1_3(r3)

        x = torch.cat([r1, r2, r3, r4], 1)
        x = self.cov3x3_1(x)
        return x

class MultiPoolTranspose(nn.Module):
    def __init__(self,c1, c2):
        super(MultiPoolTranspose, self).__init__()
        self.p = nn.MaxPool2d(3, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(c1, c1, 3, 2, 1, 1,False)
        self.dconv2 = nn.ConvTranspose2d(c1, c1, 3, 2, 1, 1,False)

        self.cov1x1_1 = Conv(c1,c2//8, k=1, s=1)
        self.cov1x1_2 = Conv(c1,c2//8, k=1, s=1)
        self.cov1x1_3 = Conv(c1,c2//8, k=1, s=1)
        self.cov3x3_2 = Conv(c1,c2//8, k=3, s=2)
        self.cov3x3_1 = Conv(c2//2,c2, k=3, s=1)

    def forward(self,x):
        r1 = self.p(x)
        r2 = self.p(r1)
        r3 = self.p(r2)

        r2 = self.dconv1(r2)
        r3 = self.dconv2(r3)
        r3 = self.dconv2(r3)

        r1 = self.cov1x1_1(r1)
        r2 = self.cov1x1_2(r2)
        r3 = self.cov1x1_2(r3)
        r4 = self.cov3x3_2(x)

        x = torch.cat([r1, r2, r3, r4], 1)
        x = self.cov3x3_1(x)
        return x


class CovAndHW(nn.Module):
    def __init__(self, c1, c2, size=160):
        super(CovAndHW, self).__init__()
        self.size = size
        self.img_size = (size, size)

        self.cov_a = Conv(c1, c1, k=(size, size), p=0)
        self.cov_b = Conv(c1, c1, k=(1, size), p=0)
        self.cov_c = Conv(c1, c1, k=(size, 1), p=0)

    def forward(self, x):
        self.img_size = x.shape[2:]
        x = F.interpolate(x, (self.size, self.size))

        b = self.cov_b(x)
        c = self.cov_c(x)

        w = (b @ c) * (c @ b)
        x = x + x @ w

        x = F.interpolate(x, self.img_size)

        return x


class Fusion2Branch(nn.Module):
    def __init__(self, c1, c2):
        super(Fusion2Branch, self).__init__()

        self.x1_cov = Conv(c1, c1, 1, 1)
        self.x2_cov = Conv(c1, c1, 1, 1)

        self.sub_cov = Conv(c1, c2, 1, 1)
        self.add_cov = Conv(c1, c2, 1, 1)

        self.cat_cov = Conv(c1 * 2, c2, 3, 1, 1)
        self.ac_cov = Conv(c1 * 2, c2, 3, 1, 1)
        self.sub_cat_cov = Conv(c1 * 3, c2, 3, 1, 1)

        self.cat_x1_cov = Conv(c1 * 2, c2, 3, 1, 1)
        self.cat_x2_cov = Conv(c1 * 2, c2, 3, 1, 1)

    def forward(self, x1, x2):
        x1 = self.x1_cov(x1)
        x2 = self.x2_cov(x2)

        sub = torch.sub(x1, x2)
        add = torch.add(x1, x2)

        sub = self.sub_cov(sub)
        add = self.add_cov(add)

        cat = torch.cat([x1, x2], 1)
        cat = self.cat_cov(cat)

        ac = torch.cat([add, cat], 1)
        ac = self.ac_cov(ac)

        sub = torch.cat([sub, torch.sub(ac, x1), torch.sub(ac, x2)], 1)
        sub = self.sub_cat_cov(sub)

        x1 = torch.cat([x1, sub], 1)
        x1 = self.cat_x1_cov(x1)

        x2 = torch.cat([x2, sub], 1)
        x2 = self.cat_x2_cov(x2)

        return x1, sub, x2

class Fusion3Branch(nn.Module):
    def __init__(self, c1, c2):
        super(Fusion3Branch, self).__init__()

        self.x1_cov = Conv(c1, c1, 1, 1)
        self.x2_cov = Conv(c1, c1, 1, 1)

        self.sub_cov = Conv(c1, c2, 1, 1)
        self.add_cov = Conv(c1, c2, 1, 1)

        self.cat_cov = Conv(c1 * 2, c2, 3, 1, 1)
        self.ac_cov = Conv(c1 * 2, c2, 3, 1, 1)
        self.sub_cat_cov = Conv(c1 * 3, c2, 3, 1, 1)
        self.sub_cat_cov2 = Conv(c1 * 2, c2, 3, 1, 1)
        self.cat_x1_cov = Conv(c1 * 2, c2, 3, 1, 1)
        self.cat_x2_cov = Conv(c1 * 2, c2, 3, 1, 1)

    def forward(self, x1, sub0, x2):
        x1 = self.x1_cov(x1)
        x2 = self.x2_cov(x2)

        sub = torch.sub(x1, x2)
        add = torch.add(x1, x2)
        sub = torch.cat([sub,sub0],1)
        sub =self.sub_cat_cov2(sub)
        # sub = self.sub_cov(sub)
        add = self.add_cov(add)

        cat = torch.cat([x1, x2], 1)
        cat = self.cat_cov(cat)

        ac = torch.cat([add, cat], 1)
        ac = self.ac_cov(ac)

        sub = torch.cat([sub, torch.sub(ac, x1), torch.sub(ac, x2)], 1)
        sub = self.sub_cat_cov(sub)

        x1 = torch.cat([x1, sub], 1)
        x1 = self.cat_x1_cov(x1)

        x2 = torch.cat([x2, sub], 1)
        x2 = self.cat_x2_cov(x2)

        return x1, sub, x2


