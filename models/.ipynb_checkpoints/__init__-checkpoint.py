from .unet1 import UNet
from .unet2 import UNet as UNet_up_cat
from .unetvgg import VGGUNet
from .unetresnet import UNet as ResUnet
from .unetvit import Model as VitUnet
from .cross_model_same_scale_multi_keral import Model as Unet_cross_model_multi_keral
from .cross_model_same_scale import Fusion2Backbone as Unet_cross_model


from .FDD_model_3_3 import UNet as FDD_3_3
from .FDD_model_3_3_2 import UNet as FDD_model_3_3_2
from .FDD_model_5_3 import UNet as FDD_model_5_3

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
             
            'FDD_3_3':FDD_3_3,
            'FDD_model_3_3_2': FDD_model_3_3_2,
             'FDD_model_5_3':FDD_model_5_3,

             'Unet_multitask': Unet_multitask
          
            }

from model_zoos import model_dic as model_dic0

for k in model_dic0.keys():
    model_dic[k] = model_dic0[k]

# print(list(model_dic.keys()))

def get_model(model_name):
    print("init model :", model_name)
    return model_dic[model_name]