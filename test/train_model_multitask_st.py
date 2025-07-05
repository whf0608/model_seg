import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from  pathlib import Path
import json
import sys
from torch  import nn
from tqdm import tqdm
from init import *
from train_model import train_model
from models.FDD_model_multitask  import UNet
from lossers import dice_loss
from torch import optim

config_path = r'model_config/seg_dataset.json'
cfg = json.load(open(config_path))
cfg['data']['path']=r'/home/wanghaifeng/project_work/datasets/changedetection_dataset/zaihai/imgs8'
cfg['data']['use_sub']=['t2', 't2_b', 'cd_mask_1','cd_cd_1']
cfg['data']['index']='cd_cd_1'
cfg['data']['limt_num']=None
cfg['training']['batch_size']=1



def train(show):
    save_path =init_file(model_name='model_test_',flag='v1_1')
            
    model = UNet(3,2)
    n_classes, trainloader, valloader = get_dataloader(cfg)
    optimizer, scheduler = get_optimizer_scheduler(model,cfg)
    
    get_train_loss_func =  get_loss_function(cfg["training"]["loss"])
    get_val_loss_func =  get_loss_function(cfg["training"]["loss"])
    
    learning_rate: float = 1e-5
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer,1,0.7)
    
    train_model(model,start_epoch=0,epochs=200,amp = False,device=device,n_classes=n_classes, trainloader=trainloader, valloader=valloader,optimizer=optimizer, scheduler=scheduler,get_loss_func=get_loss_func,get_val_func=get_loss_func,train_proccesing=train_proccesing,per_n=50)

