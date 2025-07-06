from .unet_model1 import UNet as base_unet
from .unet_model2 import CBRNet as vgg_unet
from .unet_model3 import CBRNet as vgg_unet1
from .FDD_model_3_3 import UNet as FDD_3_3
from .FDD_model_3_3_2 import UNet as FDD_model_3_3_2
from .FDD_model_multitask import UNet as FDD_model_multitask
from .FDD_model_5_3 import UNet as FDD_model_5_3
# from model_zoos.xview2_1st.models import Res34_Unet_Double
from .vit_model import Model as vit_model
from .unet_resnet import UNet as resnet_unet
from .vit_model_5_3 import Model as vit_model_5_3

model_dic = {'base_unet':base_unet,
            'vgg_unet':vgg_unet,
             'vgg_unet1':vgg_unet1,
             'res_unet':resnet_unet,
            'FDD_3_3':FDD_3_3,
            'FDD_model_3_3_2': FDD_model_3_3_2,
             'FDD_model_multitask': FDD_model_multitask,
             'FDD_model_5_3':FDD_model_5_3,
             # 'Res34_Unet_Double': Res34_Unet_Double,
             'vit_model': vit_model,
             'vit_model_5_3': vit_model_5_3
            }

from model_zoos import model_dic as model_dic0

for k in model_dic0.keys():
    model_dic[k] = model_dic0[k]

# print(list(model_dic.keys()))

def get_model(model_name):
    print("init model :", model_name)
    return model_dic[model_name]