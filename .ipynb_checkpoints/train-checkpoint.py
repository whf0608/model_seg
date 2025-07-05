import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
sys.path.append('../segmention_buildings/BuildFormer')
sys.path.append('/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/segment-anything/segment_anything')
from init import parse_args
from train_model_mian import train


if __name__ == '__main__':
    args = parse_args()
    args = args.__dict__
    # args['config'] = 'model_config/model_FDD_model_multitask_seg_dataset_imgs8.json'
    # args['config'] = 'model_works/vgg_unet/vgg_unet_2024_01_23_11_33_46/model_config.json'
    args['config'] = 'model_config/model_vit_model_seg_dataset_harvey.json'
    args['img_size'] = 1024
    args['batch_size'] = 4
    args["rdd"] = False
    args['resume_from'] = None
    args['per_n'] = 1
    args['weight_path'] = None
    args['pretrain'] =False
    train(args=args, show=None)
