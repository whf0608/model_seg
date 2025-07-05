import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from  pathlib import Path
import json
import sys
from torch  import nn
from tqdm import tqdm
from init import *
from train_model import train_model
from tran_process import Train_Precess
from models import get_model
from torch import optim
from lossers import get_loss_function
    
config_path = r'model_config/model_vgg_unet_config.json'
work_space = './model_test'
flag=None

def train(show):
    ##1. 配置参数
    cfg = json.load(open(config_path))
    n_classes=cfg['data']['n_classes']+1
    save_path = init_file(work_space=work_space,model_name=cfg['model'],flag=flag)
    with open(save_path+'/model_config.json','w') as f:
        f.write(json.dumps(cfg))
    
    
    ###2. 加载数据
    trainloader, valloader = get_dataloader(cfg)
    
    ###3. 加载权重
    model = torch.nn.parallel.DataParallel(get_model(cfg['model'])(3, n_classes))
    
    if flag is not None:
        with open(save_path+'/log.txt','r') as f:
            line = f.readlines()[-1]
            start_epoch=int(line.split(" ")[1])
        load_weights= save_path+'/model.pt'
        if Path(load_weights).exists():
            print('loading weights', load_weights)
            model.load_state_dict(torch.load(load_weights, map_location='cpu'))
     
    ### 4. 初始化损失和优化器
    optimizer, scheduler = get_optimizer_scheduler(model,cfg)
 
    get_train_loss_func =  get_loss_function(cfg["training"]["loss"])
    get_val_loss_func =  get_loss_function(cfg["training"]["loss"])
    
    ### 5. 训练过程处理
    train_proccesing = Train_Precess(save_path=save_path,on_show=show).train_proccesing
    
    train_model(model,start_epoch=0,epochs=200,amp = False,device=device,n_classes=n_classes, trainloader=trainloader, valloader=valloader,optimizer=optimizer, scheduler=scheduler,get_loss_func=get_train_loss_func,get_val_func=get_val_loss_func,train_proccesing=train_proccesing,per_n=1)

