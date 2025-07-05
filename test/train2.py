import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json
import sys
from torch  import nn
from tqdm import tqdm
from init import *
from  train_model import train_model
import cv2
from torch import optim
import torch.nn.functional as F
from lossers import dice_loss
from  datasets.tools import *
from metrics.m_metrics.metrics_np import *
from models.FDD_model_5_3 import UNet


def get_loss_func(rs,images):
    mask2 = images['mask2_1'][:,0:1,:,:].to(device)  # .half()
    mask1 = images['cd_mask_1'][:,0:1,:,:].to(device)#.half()
    if torch.sum(mask1)==0:
        cd_w=0
    else:
        cd_w=1 
    cd_w=0
    r1, r2, r3,r4,r5 =rs

    n_class=2
    r3=r3[:,1,:,:]+ bcecriterion(r3.squeeze(), mask2.squeeze().float())\


def get_val_func(rs,images):
    hist=fast_hist(rs.flatten().cpu().detach().int().numpy(),true_masks[:,0,...].flatten().cpu().int().numpy(),2)
    lss=(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))[-1]
    return lss

def main():
    args = parse_args()
    config_path = args.config
    work_space = args.work_dir
    flag = args.resume_from
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### 1. 配置参数
    cfg = json.load(open(config_path))
    save_path = init_file(work_space=work_space,model_name=cfg['model'],flag=flag)
    with open(save_path+'/model_config.json','w') as f:
        f.write(json.dumps(cfg))
    train_proccesing = Train_Precess(save_path=save_path).train_proccesing

    ### 2. 损失函数
    criterion = nn.MultiLabelSoftMarginLoss()
    bcecriterion = nn.BCEWithLogitsLoss()
    cecriterion=nn.CrossEntropyLoss(ignore_index=-1)
    
    ### 3.加载数据 
    n_classes, trainloader, valloader = get_dataloader(cfg)
    n_classes=2

    model = UNet(3,n_classes)    
    load_weights= save_path+'/model.pt'
    model = torch.nn.parallel.DataParallel(model)
    if Path(load_weights).exists():
        with open(save_path+'/log.txt','r') as f:
            line = f.readlines()[-1]
            start_epoch=int(line.split(" ")[1])
        print('loading weights', load_weights)
        model.load_state_dict(torch.load(load_weights, map_location='cpu'))
      
    optimizer, scheduler = get_optimizer_scheduler(model,cfg)
    learning_rate: float = 1e-5
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer,1,0.7)

    get_val_func = get_loss_func
    train_model(model,start_epoch=100,epochs=200,amp = False,device=device,n_classes=n_classes,
                trainloader=trainloader, valloader=valloader,testloader=testloader,optimizer=optimizer, 
                scheduler=scheduler,get_loss_func=get_loss_func,get_val_func=get_val_func,
                get_test_func=get_test_func,train_proccesing=train_proccesing,per_n=1)


def parse_args():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()