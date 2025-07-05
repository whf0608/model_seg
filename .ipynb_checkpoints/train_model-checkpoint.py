from tqdm import tqdm
import torch
import cv2
import numpy as np
from torch.amp import autocast
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

def train_model(model,device=None,start_epoch=0,epochs=2,amp = True,n_classes=3, trainloader=None, valloader=None, testloader=None,optimizer=None, scheduler=None,get_loss_func=None,get_val_func=None,get_test_func=None,train_proccesing=None,per_n=1,model_vis=False, update_lr_paramter=None):
    if model_vis:
        model_vis=train_proccesing
    else:
        model_vis = False
    
    model.train()
    model = model.cuda().float()
    grad_scaler = torch.amp.GradScaler('cuda',enabled=amp)
    optimizer.zero_grad(set_to_none=True)
    if per_n==1:
        grad_steps=8
    else:
        grad_steps=1
    for epoch in range(start_epoch, epochs):
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{epochs}', unit=' img') as pbar:
            model.train()
            lss = []
            for i, images in enumerate(trainloader):
                for _ in range(per_n):
                    img = images[list(images.keys())[0]].cuda().float()
                    img = torch.nn.functional.normalize(img).contiguous()
                    with autocast('cuda',enabled=amp):
                        rs=model(img)
                        loss = get_loss_func(rs,images)
                        grad_scaler.scale(loss).backward()
                        if (i%grad_steps)==0:
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            lss.append(loss.item())
                            pbar.update(img.shape[0])
                            pbar.set_postfix(**{'loss (batch)': loss.item()})
                   
                    
                    if train_proccesing: train_proccesing(tag='epoch_images',v=[i,epoch,images])
                    if train_proccesing: train_proccesing(tag='epoch_loss',v=[i,epoch,loss])
                    if train_proccesing: train_proccesing(tag='epoch_rs',v=[i,epoch,rs])
                if i%10000==0 and train_proccesing: train_proccesing(tag='epoch_i_model',v=[i,epoch,model.state_dict()])
            if train_proccesing: train_proccesing(tag='epoch_lss',v=[0,epoch,lss])
            
            scheduler.step()   
            if update_lr_paramter and epoch%10==0 and epoch>10:
                optimizer,scheduler=update_lr_paramter(model,epoch)
            
            if train_proccesing: train_proccesing(tag='epoch_model',v=[i,epoch,model.state_dict()])
            # if train_proccesing is not None: train_proccesing(tag='epoch_lss',v=[i,epoch,lss])
            if valloader:
                vallss = val_model(model,valloader=valloader,get_val_func=get_val_func,device=device,per_n=1,train_proccesing=train_proccesing,epoch=epoch)
                if train_proccesing:
                    train_proccesing(tag='epoch_train_val_lss',v=[i,epoch,lss,vallss])
                    model_best,eq_loss = train_proccesing(tag='epoch_vallss_model',v=[i,epoch,vallss,model.state_dict()])
                    # if model_best:
                    #     print('updating model paramter')
                    #     model.load_state_dict(model_best)
                    if eq_loss:
                         if update_lr_paramter:optimizer,scheduler=update_lr_paramter(model,epoch)
  
                    
            pbar.set_postfix(**{'loss': sum(lss) / len(lss),'valloss': sum(vallss) / len(vallss)})
    if testloader is not None: test_model_1(model,testloader,get_test_func,device,per_n=1)
            
def val_model(model,valloader=None,get_val_func=None,device=None,per_n=4,train_proccesing=None,epoch=0):
    model.eval()
    lss = []
    for i, images in enumerate(valloader):
        img = images[list(images.keys())[0]].to(device).float()
        img = torch.nn.functional.normalize(img).contiguous()
        with torch.no_grad():
            for _ in range(per_n):
                rs=model(img)
                if train_proccesing: train_proccesing(tag='epoch_val_rs',v=[0,epoch,rs])
                loss = get_val_func(rs, images)
                lss.append(loss.item())
    return lss   

def test_model_1(model, testloader=None, get_test_func=None, device=None, per_n=4):
    model.eval()
    print('testing model')
    lss = []
    for i, images in enumerate(testloader):
        img = images[list(images.keys())[0]].to(device).float()
        img = torch.nn.functional.normalize(img).contiguous()
        with torch.no_grad():
            for _ in range(per_n):
                rs=model(img)
                loss = get_test_func(rs,images)
                
                lss.append(loss.item())
    return lss


def test_model(model, testloader=None, get_test_func=None, device=None, img_size=(640,640), test_proccesing=None):
    model.eval()
    print('... test model ...')
    
    for dataset_name, dataset_path, img_ps in testloader():
        print('dataset_name: ', dataset_name, 'imgs num:', len(img_ps))
        for i,img_p in enumerate(img_ps):
            img = cv2.imread(img_p)
            if img is None: continue
            img = cv2.resize(img, img_size)

            img = img.transpose(2, 0, 1)
            img = np.array([img])
            img = torch.from_numpy(img).to(device)
            img = img/255

            with torch.no_grad():
                rs = model(img)
                if test_proccesing:  test_proccesing(tag='epoch_test_images',v=[i, img_p, rs])

        if get_test_func: get_test_func(dataset_name, dataset_path)
        
    if get_test_func: get_test_func(dataset_name, dataset_path, test_end=True)
    if test_proccesing: test_proccesing(tag='epoch_test_end', v=[i, img_p, rs])
    return 0

