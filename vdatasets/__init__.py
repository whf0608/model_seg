# from .coco import COCO
# from .voc import VOC
# from .ade20k import ADE20K
# from .cityscapes import CityScapes
import json
from torch.utils import data
from .pascal_voc_loader import pascalVOCLoader
from .camvid_loader import camvidLoader
from .ade20k_loader import ADE20KLoader
from .mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
from .cityscapes_loader import cityscapesLoader
from .nyuv2_loader import NYUv2Loader
from .sunrgbd_loader import SUNRGBDLoader
from .mapillary_vistas_loader import mapillaryVistasLoader
from .disaster import DisasterDataset
from .generate_img import GenerateDataset
from .seg_dataset import SegDataset
from .segmuilt_dataset import SegMuiltDataset
from .disasters_dataset import DisasterDataset as DisasterDataset1
from .cbr_dataset import SegfixDataset
from .map_dataset import MapDataset
from .db_dataset import DamageAssessmentDatset
data_dic = {
        "disaster_v1": DisasterDataset,
        "generate": GenerateDataset,
        "seg_dataset": SegDataset,
        "segmuilt_dataset":SegMuiltDataset,
        "disaster_v2": DisasterDataset1,
        "disaster_dataset": DisasterDataset1,
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "segfixDataset": SegfixDataset,
        "mapdataset":MapDataset,
        "DamageAssessmentDatset":DamageAssessmentDatset
    }

# import sys
# sys.path.append('../segmention_buildings/BuildFormer')
# try:
#     import geoseg
#     from geoseg.datasets.dataset import Dataset as GeoDatset
#     data_dic["geodatset"]=  GeoDatset
# except:
#     print('not import geoseg')
#     print('Geodatset erro ')


def get_loader(name):
    """get_loader

    :param name:
    """
    print("init dataset: ", name)
    return data_dic[name]



def get_dataloader(cfg,model="training"):
    data_loader = get_loader(cfg["data"]["dataset"])
    print("-----------------------",cfg['data']['path'])
    print('data config: ',cfg)
    if model=="training":
        train_dataset = data_loader(**cfg['data'])
        trainloader = data.DataLoader(train_dataset, batch_size=cfg[model]["batch_size"],num_workers=cfg[model]["n_workers"],shuffle=True,drop_last=True)

        if "valing" in cfg.keys():
            val_dataset = data_loader(**cfg['data'])
            valloader = data.DataLoader(val_dataset, batch_size=cfg['valing']["batch_size"], num_workers=cfg['valing']["n_workers"])
        else:
            val_dataset = train_dataset
            valloader = data.DataLoader(val_dataset, batch_size=cfg[model]["batch_size"], num_workers=cfg[model]["n_workers"])
        return trainloader, valloader
    
    if model=="testing":
        test_dataset = data_loader(data_root=cfg['data']['path'], img_size=cfg['testing']["img_size"],**cfg['testing']['data'])
        testloader = data.DataLoader(test_dataset, batch_size=cfg['testing']["batch_size"], num_workers=cfg['testing']["n_workers"])
        return testloader