import cv2
import torch
import io
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as opjoin
from utiles.tensor2imgmask import show_result_hard_orig, show_img, show_mask, show_result

class Train_Precess:
    def __init__(self, save_path, on_show=None, end_i=5):
        self.save_path = save_path
        self.save_metric = opjoin(save_path, 'result')
        self.best_vallss = 0
        self.bad_vallss_n = 0
        self.eq_lss_n = 0
        self.on_show = True if on_show is not None else False
        self.on_show_fn = on_show
        self.end_i = end_i
        self.excute_step = 5
        self.dc1 = Draw_canv()
        self.dc2 = Draw_canv()
        self.dc3 = Draw_canv()

    def train_proccesing(self, tag='', v=None):
        if tag == 'epoch_rs' or tag == 'epoch_val_rs':
            i, epoch, rs = v
            if type(rs) != tuple:
                rs = tuple([rs])
            if not self.on_show and epoch % self.excute_step == 0 and i < self.end_i:
                for j, r in enumerate(rs[0:]):
                    cv2.imwrite(opjoin(self.save_path, str(epoch) + '_' + str(i) + '_' + 'r' + str(j) + '.png'), show_result_hard_orig(r))
            if self.on_show:
                for j, r in enumerate(rs):
                    self.on_show_fn(tag+str(j), show_result_hard_orig(r))
            
        if tag == 'epoch_images':
            i, epoch, images = v
            if not self.on_show and epoch % (self.excute_step*10) == 0 and i < self.end_i:
                for k in images:
                    if 'image' in k:
                        try:
                            cv2.imwrite(opjoin(self.save_path , str(epoch) + '_' + str(i) + '_' + k + '.png'), show_img(images[k]))
                        except:
                            pass
            if self.on_show:
                index_i,epoch,images = v
                for k in images:
                    if k in ['t1','t2','mask_deg_1','mask_depth_1','image_t1','image_t2',
                             'image','mask','cd_cd_1','cd_mask_1','t2_b','t1_b','mask_b','mask1_1','mask1_2',"mask2_b1", "mask2_1","cd_cd","cd_mask","mask1"]:
                        self.on_show_fn('images_'+k,show_img(images[k]))
                    elif k in ['mask1_2','mask2_2',"mask1_3","mask2_3"]:
                        self.on_show_fn('mask_'+k,show_mask(images[k]))
                        
        if tag=='epoch_loss':
            index_i, epoch, loss = v
            
        if tag == 'epoch_lss':
            index_i, epoch, lss = v
            if self.on_show:
                self.on_show_fn('images_loss', self.dc1.draw([_ for _ in range(len(lss))],lss,'-'))
                self.on_show_fn('images_lss', self.dc2.draw(epoch,sum(lss) / len(lss)))
                # with open(opjoin(self.save_path, 'log.txt'), 'a') as f:
                #         f.write('epoch:'+str(epoch)+'  '+str(sum(lss) / len(lss))+'\n')
                        
        if tag == 'epoch_vallss':
            index_i, epoch, lss = v
            if self.on_show:
                img = self.dc3.draw([_ for _ in range(len(lss))], lss, '-')
                self.on_show('images_loss', img)
                if epoch % 50 == 0:
                    cv2.imwrite(opjoin(self.save_path, str(epoch)+'lss.png'), img)
                
        if tag == 'epoch_train_val_lss':
            i, epoch, lss, vallss = v
            with open(opjoin(self.save_path , 'log.txt'), 'a') as f:
                f.write('epoch: ' + str(epoch) + '  train loss:' + str(sum(lss) / len(lss)) + '  val loss:' + str(
                    sum(vallss) / len(vallss)) + '\n')

        if tag == 'epoch_model':
            i, epoch, model = v
            torch.save(model, opjoin(self.save_path , 'model.pt'))
        if tag == 'epoch_i_model':
            i, epoch, model = v
            torch.save(model, opjoin(self.save_path , str(i)+'_model.pt'))
            
        if tag == 'epoch_test_images':
            i, img_p, rs = v
            if i < self.end_i:
                for ri, r in enumerate(rs):
                    img = show_result(r)
                    # if img.max()>0:
                    img[img > 10] = 255
                    img[img < 10] = 0
                    cv2.imwrite(opjoin(self.save_metric , img_p.split('/')[-1].replace('.png', str(ri + 0) + '.png')), img)

        if tag == 'epoch_test_end':
            i, img_p, rs = v
            # metrics.texformt.init_head()

        if tag == 'epoch_vallss_model':
            i, epoch, vallss, model = v
            vallss = sum(vallss) / len(vallss)
            if epoch == 10:
                self.best_vallss = vallss
                torch.save(model, opjoin(self.save_path , 'model_best.pt'))

            if epoch > 10:
                if vallss > self.best_vallss:
                    if self.bad_vallss_n > 3:
                        model = torch.load(opjoin(self.save_path, 'model_best.pt'))
                        self.bad_vallss_n = 0
                        return (None, None)
                    else:
                        self.bad_vallss_n += 1
                elif vallss < self.best_vallss:
                    torch.save(model, opjoin(self.save_path, 'model_best.pt'))
                    self.best_vallss = vallss
                elif self.eq_lss_n>5:
                    return (None, self.eq_lss_n)
                else:
                    self.eq_lss_n+=1
            # print(self.best_vallss)
            return (None, None)

class Draw_canv:
    def __init__(self):
        self.img = np.zeros((640, 640, 3))
        self.x = []
        self.y = []

    def draw(self, x, y, f='o'):
        self.x.append(x)
        self.y.append(y)
        fig = plt.figure(figsize=(6, 6), dpi=100)
        if f == '0':
            ax = plt.plot(self.x, self.y, f)
        else:
            for x, y in zip(self.x, self.y):
                ax = plt.plot(x, y, f)
        # img=np.array(fig.canvas.renderer.buffer_rgba())
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
