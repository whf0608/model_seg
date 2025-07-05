import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from  pathlib import Path
import json
import sys
from torch  import nn
from tqdm import tqdm
from init import *
from train_model import train_model
from models.FDD_model_5_3 import UNet
from lossers import dice_loss
from torch import optim

config_path = r'model_config/seg_dataset.json'
cfg = json.load(open(config_path))
cfg['data']['path']=r'/home/wanghaifeng/project_work/datasets/changedetection_dataset/zaihai/imgs8'
cfg['data']['use_sub']=['t2','t2_b','mask2_b2','mask1_1','cd_cd_1','cd_mask_1']
cfg['data']['index']='t2'
cfg['data']['limt_num']=None
cfg['training']['batch_size']=4





bcecriterion = nn.BCEWithLogitsLoss()

    ###--------------------------------
    n_classes, trainloader, valloader = get_dataloader(cfg)
    n_classes=2
    
    load_weights='weights/model_test_v1_1/model.pt'
    # load_weights=''
    if not Path(load_weights).exists():
            load_weights=''
    model = UNet(3,n_classes)
    if len(load_weights)>0:
        model0 = UNet(3,n_classes)
        model0 = torch.nn.parallel.DataParallel(model0)
        
        print('loading weights',load_weights)
        model0.load_state_dict(torch.load(load_weights, map_location='cpu'))
        state_dict = model0.module.state_dict()
        model.load_state_dict(state_dict)

    model = torch.nn.parallel.DataParallel(model.to(device))  
    
    optimizer, scheduler = get_optimizer_scheduler(model,cfg)
    
    learning_rate: float = 1e-5
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer,1,0.7)
    
    
    train_model(model,start_epoch=0,epochs=200,amp = False,device=device,n_classes=n_classes, trainloader=trainloader, valloader=valloader,optimizer=optimizer, scheduler=scheduler,get_loss_func=get_loss_func,get_val_func=get_loss_func,train_proccesing=train_proccesing,per_n=2)

