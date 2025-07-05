import torch
import random
import os
import logging
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
import time
import argparse


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--config', help='train config file path', default= "model_config/model_unetbase_xbdeq_bachsize1_imgs640.json")
    parser.add_argument('--work-dir', help='the dir to save logs and models', default="model_works")
    parser.add_argument('--start-epoch', type=int, help='start epoch', default=0)
    parser.add_argument('--end-epoch', type=int, help='end epoch', default=200)
    parser.add_argument('--per-n', type=int, help='end epoch', default=1)
    parser.add_argument('--img-size', type=int, help='end epoch', default=640)
    parser.add_argument('--batch-size', type=int, help='end epoch', default=1)
    parser.add_argument('--pretrain', type=bool, help='pretrain', default=False)
    parser.add_argument('--weight-path', type=str, help='wegith path', default='')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from', default=False)
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=0,
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


def init_file(work_space='weights/', model_name='model', flag=None):
    if flag is None or not flag:
        flag = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = work_space + '/' + model_name

    if not Path(save_path).exists():
        Path(save_path).mkdir()

    save_path = save_path + '/' + model_name + '_' + flag

    if not Path(save_path).exists():
        Path(save_path).mkdir()

    return save_path + '/'


def init_seeds(seed=0):
        # Initialize random number generator (RNG) seeds
        random.seed(seed)
        np.random.seed(seed)
        init_torch_seeds(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.enabled = False
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False

init_seeds(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################

# config_path = r'model_config/seg_dataset.json'
# cfg = json.load(open(config_path))
# cfg['data']['use_sub']=['t2', 't2_b','t1', 't1_b',  'mask2', 'mask1']
# cfg['data']['index']='t2'
# cfg['data']['limt_num']=10
# cfg['training']['batch_size']=2

##################


# root_path =r'/home/wanghaifeng/project_work/datasets/changedetection_dataset/'  
# dataset_roots=[root_path+'zaihai/imgs*',root_path+'*']

# def get_test_data():
#     dd=Datasets_Deal(dataset_roots,tt_pre=['t2',save_metric])
#     dd.clearn_subfile(ttv=[save_metric])
#     testloader = dd.get_data(img_flg='/train/t2/*.png',num_limt=5000,min_num_limt=100)
#     return testloader
# testloader=get_test_data

# def fast_hist(label_pred, label_true, num_classes):
#     mask = (label_true >= 0) & (label_true < num_classes)
#     hist = np.bincount(
#         num_classes * label_true[mask].astype(int) +
#         label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
#     return hist


# metrics = Metrics(texformt_path=save_path+'/tex1.txt')
# def get_test_func(dataset_name,dataset_path,test_end=False):
#     paths=glob(dataset_path+'/train/'+save_metric+'/*2.png')
#     print('metrics: '+dataset_name,'result num: ',len(paths))
#     mr = metrics.running(name=dataset_name,paths=paths,
#                         num_limt=10000,result_f=save_metric,mask_f='mask2_1',subfix=['2.png','.png'],deal_img_func=None)
#     if test_end :
#         metrics.texformt.init_head()
#     return None

# log_print=False
# log_wirting=True
# save_path='_2'
# def print_log(k,v=''):
#     if type(v) is list:
#         b=''
#         for _ in v:
#             b+=(_+', ')
#         v=b
        
#     if type(v) is set:
#         b=''
#         for _ in v:
#             print(_)
#             b+=(_+', ')
#         v=b
        
#     if log_print:print(k,v)
#     if log_wirting:
#         with open('train_log'+save_path+'.txt','a') as f:
#             f.write(str(k)+str(v)+'\n')