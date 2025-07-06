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
from .generate_img import GenerateDataset
from .seg_dataset import SegDataset
from .cbr_dataset import SegfixDataset
from .map_dataset import MapDataset
from .bdseg_dataset import BDsegDataset
data_dic = {
        "generate": GenerateDataset,
        "SegDataset": SegDataset,
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
        "BDsegDataset":BDsegDataset
    }


def get_loader(name):
    print("init dataset: ", name)
    return data_dic[name]

def get_dataloader(cfg,model="training"):

    if model=="training":
        data_loader = get_loader(cfg['training']["data"]["dataset"])
        print("-----------------------",cfg['training']["data"]["dataset"])
        train_dataset = data_loader(**cfg['training']['data'])
        trainloader = data.DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"],num_workers=cfg["training"]["n_workers"],shuffle=True,drop_last=True)
        data_loader = get_loader(cfg['val']["data"]["dataset"])
        val_dataset = data_loader(**cfg['val']['data'])
        valloader = data.DataLoader(val_dataset, batch_size=cfg['val']["batch_size"], num_workers=cfg['val']["n_workers"])
       
    return trainloader, valloader
    
    if model=="testing":
        data_loader = get_loader(cfg['testing']["data"]["dataset"])
        test_dataset = data_loader(**cfg['testing']['data'])
        testloader = data.DataLoader(test_dataset, batch_size=cfg['testing']["batch_size"], num_workers=cfg['testing']["n_workers"])
        return testloader