from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, StepLR, ConstantLR, LinearLR, \
    ExponentialLR, CosineAnnealingLR, ChainedScheduler, SequentialLR,ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts
from torch import optim
from .schedulers import WarmUpLR, ConstantLR, PolynomialLR
from .optimizers import *

scheduler_dic = {
    "constant_lr": ConstantLR,
    "poly_lr": PolynomialLR,
    "multi_step": MultiStepLR,
    "cosine_annealing": CosineAnnealingLR,
    "exp_lr": ExponentialLR,
    'lambdalr': LambdaLR,
    'steplr': StepLR,
    'linearlr': LinearLR,
    'chainedscheduler': ChainedScheduler,
    'sequentiallr': SequentialLR,
    'reduceLronplateau': ReduceLROnPlateau,
    'cycliclr': CyclicLR,
    'Cosineannealingwarmrestars': CosineAnnealingWarmRestarts,

}

def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        return ConstantLR(optimizer)
    s_type = scheduler_dict["name"]
    
    
    scheduler_dict.pop("name")
    warmup_dict = {}
    if "warmup_iters" in scheduler_dic:
        # This can be done in a more pythonic way...
        warmup_dict["warmup_iters"] = scheduler_dict.get("warmup_iters", 100)
        warmup_dict["mode"] = scheduler_dict.get("warmup_mode", "linear")
        warmup_dict["gamma"] = scheduler_dict.get("warmup_factor", 0.2)
        scheduler_dict.pop("warmup_iters", None)
        scheduler_dict.pop("warmup_mode", None)
        scheduler_dict.pop("warmup_factor", None)
        base_scheduler = scheduler_dic[s_type](optimizer, **scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)
    
    print("init scheduler: ", s_type, 'params: ', scheduler_dict )
    return scheduler_dic[s_type](optimizer, **scheduler_dict)


def get_optimizer_scheduler(model,cfg):
   
    
    optimizer_cls = get_optimizer(cfg["training"]["optimizer"]["name"])
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    print("init optimizer :", cfg["training"]["optimizer"]["name"], "params: ",optimizer_params)
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    return optimizer, scheduler 


def update_paramter(model, epoch):
    print("update paramter")
    learning_rate: float = 1e-5
    learning_rate1: float = 1e-7
    
    if epoch>100:
         learning_rate= learning_rate-epoch*1e-7*0.5
    
    optimizer = optim.RMSprop(model.module.UNet_.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer.add_param_group({'params': model.module.m1.parameters(), 'lr':learning_rate1, 'weight_decay':1e-8,'momentum': 0.9})
    optimizer.add_param_group({'params': model.module.m2.parameters(), 'lr':learning_rate1, 'weight_decay':1e-8,'momentum': 0.9})
    optimizer.add_param_group({'params': model.module.m3.parameters(), 'lr':learning_rate1,'weight_decay':1e-8,'momentum': 0.9})
    optimizer.add_param_group({'params': model.module.m4.parameters(), 'lr':learning_rate1,'weight_decay':1e-8,'momentum': 0.9})
    optimizer.add_param_group({'params': model.module.m5.parameters(), 'lr':learning_rate1,'weight_decay':1e-8,'momentum': 0.9})
    # optimizer.param_groups[0]['lr']=learning_rate-epoch*1e-7*0.5
    # optimizer.param_groups[int(1+epoch/5%5)]['lr']=learning_rate-epoch*1e-7*0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer,1,0.7)
    return optimizer,scheduler