from metrics.m_metrics.metrics_np  import Metrics
from glob import glob
m = Metrics(texformt_path='/home/wanghaifeng/whf_work/work_sync/satellite_data/valite_dataset_sys/result/unet_train01/base_unet/result/tex.txt')
img_fs = glob('/home/wanghaifeng/whf_work/work_sync/satellite_data/valite_dataset_sys/result/unet_train01/base_unet/result/*.png')
print(len(img_fs))
m.running(name='xDBeq',paths=img_fs,
          result_f='/home/wanghaifeng/whf_work/work_sync/satellite_data/valite_dataset_sys/result/unet_train01/base_unet/result',
          mask_f='/home/wanghaifeng/project_work/datasets/disaster_dataset/xBDearthquake/train/mask1',
          )



# metrics = Metrics(texformt_path=save_path+'/tex1.txt')
# def get_test_func(dataset_name,dataset_path,test_end=False):
#     paths=glob(dataset_path+'/train/'+save_metric+'/*2.png')
#     print('metrics: '+dataset_name,'result num: ',len(paths))
#     mr = metrics.running(name=dataset_name,paths=paths,
#                         num_limt=10000,result_f=save_metric,mask_f='mask2_1',subfix=['2.png','.png'],deal_img_func=None)
#     if test_end :
#         metrics.texformt.init_head()
#     return None