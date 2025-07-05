from torch import nn
from  .Activation import *
from torch.functional import F
import math
class SE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HardSigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# class Dscswf(nn.Module):
#     def __init__(self, layers):
#         super(Dscswf, self).__init__()
#         self.layers=layers
#         self.multiple = len(layers) > 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv3x3_256_512=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=2,groups=256)
#         self.conv3x3_512_1024=nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,stride=2,groups=512)
#         self.conv1x1_1024_512=nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,padding=0,stride=1)
#         self.conv1x1_512_256=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,padding=0,stride=1)
#         self.conv1x1_1024_256=nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1,padding=0,stride=1)
#
#
#     def forward(self, x,outputs):
#         b, c, h,w = x.size()
#         out=[x]
#         for i in self.layers:
#             b0, c0, h0,w0=outputs[i].shape
#             # print('output',outputs[i].shape)
#             if h<h0:
#                 out.append(self.Zoom_feature(outputs[i],None,shape=(c,h,w)))
#             else:
#                 out.append(self.Magnification(outputs[i], h/h0, shape=(c,h, w)))
#         for o in out:
#             print("============",o.shape)
#         catx=torch.cat(out,1)
#
#         channel_feature=self.channel(catx)
#         spatial_feature=self.spatial(catx)
#         # print(spatial_feature.shape)
#         # print(channel_feature.shape)
#         sc=torch.mul(spatial_feature,channel_feature)
#         # print(t0.shape)
#         result=catx*sc
#         # print(x.shape)
#         # print(result.shape)
#         x=torch.cat([x,result],1)
#         x=self.zoom_channel(x,c)
#         return x
#
#     def Zoom_feature(self,x,time,shape=None):
#         # print(shape,x.shape)
#         h=x.shape[2]
#         h0=shape[0]
#         while h/h0>1:
#             if x.shape[1]==256:
#                 x = self.conv3x3_256_512(x)
#             elif x.shape[1]==512:
#                 x=self.conv3x3_512_1024(x)
#             h=h/2
#         if x.shape[2]!=shape[1]:
#             x=F.interpolate(x,size=shape[1:])
#         print('zoom',x.shape)
#         return x
#     def Magnification(self,x,time,shape=None):
#         x = F.interpolate(x, size=shape[1:])
#         # print('magn0',x.shape)
#         if shape[0]==512 and x.shape[1]==1024:
#             x=self.conv1x1_1024_512(x)
#         elif shape[0]==256 and x.shape[1] == 512 :
#             x = self.conv1x1_512_256(x)
#         elif shape[0] == 256 and x.shape[1] ==1024:
#             x = self.conv1x1_1024_256(x)
#         # print('magn',shape,x.shape)
#
#         return  x
#
#     def channel(self,x):
#         x = F.interpolate(x, size=(13,13))
#         x=self.avg_pool(x)
#         # x=x.reshape(x.shape[:-1])
#         return x
#
#     def spatial(self,x):
#         x=nn.Conv2d(in_channels=x.shape[1],out_channels=1,kernel_size=1,stride=1)(x)
#         return x
#
#     def zoom_channel(self,x,out_channel):
#         return nn.Conv2d(in_channels=x.shape[1],out_channels=out_channel,kernel_size=1,padding=0,stride=1)(x)


# class Gloff(nn.Module):
#     def __init__(self,layers):
#         super(Gloff, self).__init__()
#         self.layers = layers
#         self.multiple = len(layers) > 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self,x,outputs):
#         b, c, h, w = x.size()
#         out = []
#         out.append(self.Zoom_feature(x, shape=(5,5)))
#         for i in self.layers:
#             out.append(self.Zoom_feature(outputs[i], shape=(5,5)))
#         x1 = torch.cat(out, 1)
#         x1 = nn.Conv2d(in_channels=x1.shape[1], out_channels=64, kernel_size=1, padding=0, stride=1)(x1)
#         x1=self.avg_pool(x1)
#         x1=self.upsamel(x1,[h,w])
#         x1=torch.cat([x,x1],1)
#
#         x1 = nn.Conv2d(in_channels=x1.shape[1], out_channels=c, kernel_size=1, padding=0, stride=1)(x1)
#         print("result", x1.shape)
#         return  x1
#     def Zoom_feature(self,x,shape=None):
#         h = x.shape[2]
#         h0 = shape[0]
#         while h / h0 > 1:
#             x = nn.Conv2d(in_channels=x.shape[1], out_channels=x.shape[1], kernel_size=3, padding=1, stride=2,
#                           groups=x.shape[1])(x)
#             h = h / 2
#         if x.shape[2] != shape[0]:
#             x = F.interpolate(x, size=shape)
#         print('zoom', x.shape)
#         return x
#
#     def upsamel(self,x,size):
#         rs=[]
#         for i in range(size[0]*size[1]):
#             r = nn.Conv2d(in_channels=x.shape[1], out_channels=32, kernel_size=1, padding=0, stride=1)(x)
#             rs.append(r)
#         x=torch.cat(rs,3)
#         sp=[]
#         shape=x.shape[:-2]
#         sp.extend(shape)
#         sp.extend(size)
#         x=x.reshape(sp)
#         return x

class Dscswf(nn.Module):
    def __init__(self, layers):
        super(Dscswf, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv512_1=nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        self.conv1792_1=nn.Conv2d(in_channels=1792, out_channels=1, kernel_size=1, stride=1)
        self.sigmoid=nn.Sigmoid()

    def channel(self, x):
        x = F.interpolate(x, size=(13, 13))
        x = self.avg_pool(x)
        return x

    def spatial(self, x):
        if x.shape[1] == 512:
            x = self.conv512_1(x)
        elif x.shape[1] == 1792:
            x = self.conv1792_1(x)
        return x

    def forward(self, x, outputs):
        channel_feature = self.channel(x)
        spatial_feature = self.spatial(x)
        channel_feature =self.sigmoid(channel_feature)
        spatial_feature = self.sigmoid(spatial_feature)
        sc = torch.mul(spatial_feature, channel_feature)
        sc = self.sigmoid(sc)
        result = x * sc
        return result


class Dscswf1(nn.Module):
    def __init__(self, layers):
        super(Dscswf, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1024_512 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.conv1x1_512_256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.conv1x1_1024_256 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.conv768_1=nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, stride=1)
        self.conv1536_1=nn.Conv2d(in_channels=1536, out_channels=1, kernel_size=1, stride=1)
        self.conv3072_1=nn.Conv2d(in_channels=3072, out_channels=1, kernel_size=1, stride=1)
        self.conv381_256=nn.Conv2d(in_channels=381, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.conv637_512=nn.Conv2d(in_channels=637, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.conv1792_125=nn.Conv2d(in_channels=1792, out_channels=125, kernel_size=1, padding=0, stride=1)
        self.conv1149_1024=nn.Conv2d(in_channels=1149, out_channels=1024, kernel_size=1, padding=0, stride=1)


    def forward(self, x, outputs):
        b, c, h, w = x.size()
        print(x.shape)
        out = [x]
        for i in self.layers:
            b0, c0, h0, w0 = outputs[i].shape
            # print('output',outputs[i].shape)
            if w < w0:
                out.append(self.Zoom_feature(outputs[i], None, shape=(c, h, w)))
            else:
                out.append(self.Magnification(outputs[i], h / h0, shape=(c, h, w)))
        # for o in out:
        #     print("============", o.shape)
        catx = torch.cat(out, 1)

        channel_feature = self.channel(catx)
        spatial_feature = self.spatial(catx)
        # print(spatial_feature.shape)
        # print(channel_feature.shape)
        sc = torch.mul(spatial_feature, channel_feature)
        # print(sc.shape)
        result = catx * sc
        # print(result.shape)
        # print(result.shape)
        result = self.zoom_channel(result, 1792)
        x = torch.cat([x, result], 1)
        print(x.shape)
        x = self.zoom_channel(x, c)
        return x

    def Zoom_feature(self, x, time, shape=None):
        # print(shape,x.shape)
        # w = x.shape[3]
        # w0 = shape[1]
        x = F.interpolate(x, size=shape[1:])
        # while w / w0 > 1:
        # if x.shape[1] == 256:
        #         x = self.conv3x3_256_512(x)
        # elif x.shape[1] == 512:
        #         x = self.conv3x3_512_1024(x)
        #     w = w / 2
        # if x.shape[2] != shape[1]:
        # if shape[0] == 512 and x.shape[1] == 256:
        #     x = self.conv1x1_256_512(x)
        # elif shape[0] == 1024 and x.shape[1] == 512:
        #     x = self.conv1x1_512_1024(x)
        # elif shape[0] == 1024 and x.shape[1] == 256:
        #     x = self.conv1x1_256_1024(x)
        # print('zoom', x.shape)
        return x

    def Magnification(self, x, time, shape=None):
        x = F.interpolate(x, size=shape[1:])
        # print('magn0',x.shape)
        # if shape[0] == 512 and x.shape[1] == 1024:
        #     x = self.conv1x1_1024_512(x)
        # elif shape[0] == 256 and x.shape[1] == 512:
        #     x = self.conv1x1_512_256(x)
        # elif shape[0] == 256 and x.shape[1] == 1024:
        #     x = self.conv1x1_1024_256(x)
        # print('magn',shape,x.shape)

        return x

    def channel(self, x):
        x = F.interpolate(x, size=(13, 13))
        x = self.avg_pool(x)
        # x=x.reshape(x.shape[:-1])
        return x

    def spatial(self, x):
        if x.shape[1]==768:
            x =self.conv768_1(x)
        elif x.shape[1]==1536:
            x =self.conv1536_1(x)
        elif x.shape[1]==3072:
            x =self.conv3072_1(x)
        return x

    def zoom_channel(self, x, out_channel):
        if out_channel==256:
            x=self.conv381_256(x)
        elif out_channel==512:
            x=self.conv637_512(x)
        elif out_channel==1024:
            x=self.conv1149_1024(x)

        elif out_channel==1792:
            x=self.conv1792_125(x)
        return x

class Gloff(nn.Module):
    def __init__(self, layers):
        super(Gloff, self).__init__()
        self.conv64_64= nn.Conv2d(64, 64, kernel_size=13, padding=0, stride=13)
        self.conv512_64 = nn.Conv2d(512,64, kernel_size=1, padding=0, stride=1)
        self.conv1792_64= nn.Conv2d(1792, 64, kernel_size=1, padding=0, stride=1)
        self.conv64_512 = nn.Conv2d( 64,512, kernel_size=1, padding=0, stride=1)
        self.conv64_1792 = nn.Conv2d( 64,1792, kernel_size=1, padding=0, stride=1)
    def forward(self, x, outputs):
        b, c, h, w =x.shape
        if x.shape[1] == 512:
            x = self.conv512_64(x)
        elif x.shape[1] == 1792:
            x = self.conv1792_64(x)
        x = F.interpolate(x, size=[13, 13])
        x = self.conv64_64(x)
        x = F.interpolate(x, size=[h, w])

        if c == 512:
            x = self.conv64_512(x)
        elif c == 1792:
            x = self.conv64_1792(x)
        print("==========",x.shape)
        return x

class Gloff1(nn.Module):
    def __init__(self, layers):
        super(Gloff, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv256 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2, groups=256)
        self.conv512 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, groups=512)
        self.conv1024 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2, groups=1024)
        # self.conv256_64 = nn.Conv2d(256, 64, kernel_size=1, padding=0, stride=1)
        # self.conv512_64 = nn.Conv2d(512, 64, kernel_size=1, padding=0, stride=1)
        self.conv1792_64 = nn.Conv2d(1792, 64, kernel_size=1, padding=0, stride=1)
        self.conv64_32 = nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1)
        self.conv288_256 = nn.Conv2d(288, 256, kernel_size=1, padding=0, stride=1)
        self.conv544_512 = nn.Conv2d(544, 512, kernel_size=1, padding=0, stride=1)
        self.conv1065_1024 = nn.Conv2d(1056, 1024, kernel_size=1, padding=0, stride=1)

    def forward(self, x, outputs):
        b, c, h, w = x.size()
        out = []
        out.append(self.Zoom_feature(x, shape=(5, 5)))
        for i in self.layers:
            out.append(self.Zoom_feature(outputs[i], shape=(5, 5)))
        x1 = torch.cat(out, 1)
        # x1 = nn.Conv2d(in_channels=x1.shape[1], out_channels=64, kernel_size=1, padding=0, stride=1)(x1)

        x1 = self.conv1792_64(x1)
        # x1 = self.avg_pool(x1)
        # x1 = self.upsamel(x1, [h, w])
        x1 = F.interpolate(x1, size=[h, w])
        x1=self.conv64_32(x1)
        x1 = torch.cat([x, x1], 1)

        if c == 1024:
            x1 = self.conv1065_1024(x1)
        elif c == 512:
            x1 = self.conv544_512(x1)
        elif c == 256:
            x1 = self.conv288_256(x1)
        # print("result", x1.shape)
        return x1

    def Zoom_feature(self, x, shape=None):
        h = x.shape[2]
        h0 = shape[0]
        # print(x)
        while h / h0 > 1:
            if x.shape[1] == 256:
                x = self.conv256(x)
            elif x.shape[1] == 512:
                x = self.conv512(x)
            elif x.shape[1] == 1024:
                x = self.conv1024(x)
            h = h / 2
        if x.shape[2] != shape[0]:
            x = F.interpolate(x, size=shape)
        # print('zoom', x.shape)
        return x

    # def upsamel(self, x, size):
    #     rs = []
    #     for i in range(size[0] * size[1]):
    #         if i % 1 == 0:
    #             r = self.conv641_32(x)
    #         elif i % 2 == 0:
    #             r = self.conv642_32(x)
    #         elif i % 3 == 0:
    #             r = self.conv643_32(x)
    #             r = self.conv643_32(x)
    #         rs.append(r)
    #     x = torch.cat(rs, 3)
    #     sp = []
    #     shape = x.shape[:-2]
    #     sp.extend(shape)
    #     sp.extend(size)
    #     x = x.reshape(sp)
    #     return x
