from .unet1 import UNet
from .unet2 import UNet as UNet_up_cat
from .unetvgg import VGGUNet
from .unetresnet import UNet as ResUnet
from .unetvit import Model as VitUnet
from .cross_model_same_scale_multi_keral import Model as Unet_cross_model_multi_keral
from .cross_model_same_scale import Fusion2Backbone as Unet_cross_model


from .cfd_3 import UNet as cfd_3
from .cfd_3_vgg import UNet as cfd_3_vgg
from .cfd_3_down_up import UNet as cfd_3_down_up
from .cfd_5 import UNet as cfd_5
from .cfd_5_cov import UNet as cfd_5_cov
from .cfd_5_vgg import UNet as cfd_5_vgg
from .cfd_5_vit import UNet as cfd_5_vit

from .unet_multitask import UNet as Unet_multitask

from model_zoos.xview2_1st.models import Res34_Unet_Double,SeResNext50_Unet_Loc,SeResNext50_Unet_Double,Dpn92_Unet_Loc,Dpn92_Unet_Double,Res34_Unet_Loc,SeNet154_Unet_Loc,SeNet154_Unet_Double


model_dic = {'Unet_down_up':UNet,
             'UNet_up_cat': UNet_up_cat,
             'vggunet':VGGUNet,
             'ResUnet':ResUnet,
             'VitUnet':VitUnet,
             'Unet_cross_model':Unet_cross_model,
             'Unet_cross_model_multi_keral':Unet_cross_model_multi_keral,

             'Res34_Unet_Double':Res34_Unet_Double,
             'SeResNext50_Unet_Loc':SeResNext50_Unet_Loc,
             'SeResNext50_Unet_Double':SeResNext50_Unet_Double,
             'Dpn92_Unet_Loc':Dpn92_Unet_Loc,
             'Dpn92_Unet_Double':Dpn92_Unet_Double,
             'Res34_Unet_Loc':Res34_Unet_Loc,
             'SeNet154_Unet_Loc':SeNet154_Unet_Loc,
             'SeNet154_Unet_Double':SeNet154_Unet_Double,
             
             'cfd_3':cfd_3,
             'cfd_3_vgg':cfd_3_vgg,
             'cfd_3_down_up':cfd_3_down_up,
             'cfd_5':cfd_5,
             'cfd_5_cov':cfd_5_cov,
             'cfd_5_vgg':cfd_5_vgg,
             'cfd_5_vit':cfd_5_vit,

             'Unet_multitask': Unet_multitask
          
            }

from model_zoos import model_dic as model_dic0

for k in model_dic0.keys():
    model_dic[k] = model_dic0[k]

# print(list(model_dic.keys()))

def get_model(model_name):
    print("init model :", model_name)
    return model_dic[model_name]