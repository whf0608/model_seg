def update_paramter(model,epoch):
    learning_rate: float = 1e-5
    learning_rate1: float = 1e-7
    
    if epoch>100:
         learning_rate1= learning_rate-epoch*1e-7*0.5

    optimizer = optim.RMSprop(model.module.UNet_.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer.add_param_group({'params': model.module.m1.parameters(), 'lr':learning_rate1, 'weight_decay':1e-8,'momentum': 0.9})
    optimizer.add_param_group({'params': model.module.m2.parameters(), 'lr':learning_rate1, 'weight_decay':1e-8,'momentum': 0.9})
    optimizer.add_param_group({'params': model.module.m3.parameters(), 'lr':learning_rate1,'weight_decay':1e-8,'momentum': 0.9})
    optimizer.add_param_group({'params': model.module.m4.parameters(), 'lr':learning_rate1,'weight_decay':1e-8,'momentum': 0.9})
    optimizer.add_param_group({'params': model.module.m5.parameters(), 'lr':learning_rate1,'weight_decay':1e-8,'momentum': 0.9})
    optimizer.param_groups[0]['lr']=learning_rate-epoch*1e-7*0.5
    optimizer.param_groups[int(1+epoch/5%5)]['lr']=learning_rate-epoch*1e-7*0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer,1,0.7)
    return optimizer,scheduler

def test_paramter():
    m0,m1,m2,m3,m4,m5 =[],[],[],[],[],[]
    ms=[m0,m1,m2,m3,m4,m5]
    for epoch in range(200):
        if epoch%5==0:
            optimizer,scheduler=update_paramter(model,epoch)

        for m,d in zip(ms,optimizer.state_dict()['param_groups']):
            m.append(d['lr']*10000)
        if epoch%10==0 and epoch>10:
            optimizer.step()
            scheduler.step()
    return ms

            
def vis_paramter(ms):
    ms=np.array(ms)
    plt.figure(figsize=(15,7))
    plt.subplot(321)
    plt.plot(ms[0]/10000)

    plt.subplot(322)
    plt.plot(ms[1]/10000,color='red')
    plt.subplot(323)
    plt.plot(ms[2]/10000,color='green')
    plt.subplot(324)
    plt.plot(ms[3]/10000,color='yellow')
    plt.subplot(325)
    plt.plot(ms[4]/10000,color='#3232CD')
    plt.subplot(326)
    plt.plot(ms[5]/10000,color='#238E68')

    plt.savefig('test.png')