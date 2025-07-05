from .cbrnet.unet_model import CBRNet
from .cbrnet.unet_model_mutilcls import CBRNet as CBRNetMC
from .SiamCRNN import SiamCRNN
from .mamba.models.STMambaBDA import STMambaBDA
from glob import glob
model_dic = {'cbrnet': CBRNet,
             'cbrnetmc':CBRNetMC,
             'SiamCRNN':SiamCRNN,
             'STMambaBDA':STMambaBDA
             }
try:
    from .GeoSeg.geoseg.models.A2FPN import A2FPN
    from .GeoSeg.geoseg.models.BANet import BANet
    from .GeoSeg.geoseg.models.MANet import MANet
    from .GeoSeg.geoseg.models.FTUNetFormer import FTUNetFormer
    from .GeoSeg.geoseg.models.UNetFormer import UNetFormer
    from .GeoSeg.geoseg.models.ABCNet import ABCNet
    from .GeoSeg.geoseg.models.DCSwin import DCSwin
    model_dic_geoseg={'ABCNet': ABCNet,
    'A2FPN': A2FPN,
    'BANet': BANet,
    'MANet': MANet,
    'FTUNetFormer': FTUNetFormer,
    'UNetFormer': UNetFormer,
    'DCSwin':DCSwin
    }
    model_dic.update(model_dic_geoseg)
except:
    print("no load geoseg")


try:
    import timm
    model_names = timm.list_models() 
    class UNetFormerBackbone:
        def __init__(self,backbone_name='swsl_resnet18'):
            self.backbone_name=backbone_name
        
        def get_model_backbone(self,**arg):
            if 'backbone_name' not in arg.keys():
                arg['backbone_name']=self.backbone_name
            else:
                self.backbone_name = arg['backbone_name']
            print("============================================")
            print("loading backbone: ",self.backbone_name, arg)
            return UNetFormer(**arg)
        
    for model_name in model_names:
        model_dic['UNetFormer'+model_name] = UNetFormerBackbone(backbone_name=model_name).get_model_backbone
except:
    print("no load timm")

# try:
if True:
    import os
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)



    from mmseg.apis import inference_model, init_model
    from mmengine.model import revert_sync_batchnorm
    class Model_C:
        def __init__(self,model_config):
            self.model_config = model_config
        def get_model(self,dim=3,n_classes=3,revert_sync_batchnorm=False,**arg):
            # print('mmseg model config',self.model_config)
            # os.system("cat "+self.model_config)
            model = init_model(config=self.model_config)
            if revert_sync_batchnorm: model  = revert_sync_batchnorm(model)
            return model
        
    model_configs = glob(f'{current_dir}/mmsegmentation/configs/_base_/models/*.py')
    model_dic_mmseg = {}
    for model_config in model_configs:
        model_dic[model_config.split('/')[-1][:-3]] = Model_C(model_config).get_model
        
    model_configs = glob(f'{current_dir}/mmsegmentation/configs/*/*.py')
    
    for model_config in model_configs:
        if "_base_" == model_config.split('/')[-2]: continue
        model_dic[model_config.split('/')[-1][:-3]] = Model_C(model_config).get_model
# except:
#     print("no load mmseg")
try:
    sys.path.append(f'{current_dir}/vit-pytorch/vit_pytorch')
    # glob('/home/wanghaifeng/whf_work/vit/vit-pytorch/vit_pytorch/*')
except:
    pass
    
def test_all_models(model_dic=None):
    try:
        model_dic[model_name](None)
        print('ok: ',model_name)
    except:
        print('erro: ',model_name)






    