from torch import nn
from modules.blocks.blocks import DoubleConv,Down,Up,OutConv,Conv
from modules.blocks.blocks import Up_same as Up
import torch
import sys
import torch.nn.functional as F
try:
    from .modeling.image_encoder import ImageEncoderViT
except:
    print("no loading vit")
class Model(nn.Module):
    def __init__(self, n_channels=3, n_classes=2,sam_checkpoint=None, bilinear=False,**arg):
        super(Model, self).__init__()
        self.feature = ImageEncoderViT(embed_dim=1280,depth=32,num_heads=16,global_attn_indexes=[7,15,23,31],img_size=1024,qkv_bias=True,use_rel_pos=True,window_size=14,out_chans=256)

        factor = 2 if bilinear else 1
        self.cov3 = Conv(256,128)
        self.up1 = Up(256, 256, bilinear,input_same=True)
        self.up2 = Up(256, 256, bilinear, input_same=True)
        self.up3 = Up(256, 256, bilinear,input_same=True)
        self.up4 = Up(256, 64, bilinear, input_same=True)
        self.outc = OutConv(64, n_classes)
        self.sam_checkpoint = sam_checkpoint
        if self.sam_checkpoint:
            self.param()
        print('vit model init')

    def param(self):
        print('loading sam weight:',self.sam_checkpoint)
        sam = torch.load(self.sam_checkpoint, map_location='cpu')
        self.feature.load_state_dict(sam)
        for param in self.feature.parameters():
            param.requires_grad = False

    def forward(self, x, show=None):
        with torch.no_grad():
            x,interm_embeddings = self.feature(x)

        xs=[]
        for x in interm_embeddings:
            x = self.feature.neck(x.permute(0, 3, 1, 2))
            xs.append(x)

        x_ = F.interpolate(xs[0], size=(128,128))
        x_ = self.cov3(x_)
        
        x = self.up1(x, x_)
 
        x_ = F.interpolate(xs[1], size=(256,256))
        x_ = self.cov3(x_)
        x = self.up2(x, x_)
     
        x_ = F.interpolate(xs[2], size=(512,512))
        x_ = self.cov3(x_)
        x = self.up3(x, x_)
    
        x_ = F.interpolate(xs[3], size=(1024,1024))
        x_ = self.cov3(x_)
        x = self.up4(x, x_)
    
        logits = self.outc(x)
        
        return logits