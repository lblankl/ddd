import os
import argparse
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import Dataset
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import model
import dataset
paddle.in_dynamic_mode()
from paddle.io import DataLoader
paddle.set_device("gpu")
import paddle.distributed as dist #第1处改动，import库





print("-----------------------------zip successfully-------------------------------------")



BATCH_SIZE =128
# 学习率
glr = 2e-4
dlr=2e-4
# 训练epoch数
epochs = 104
# 噪音dim
nz = 160
cuda = True
# 打印训练情况频率（单位step）
print_every = 100
gradient_acc=8
print("-----------------------------set parameters successfully-------------------------------------")
criterion=nn.BCELoss()
images_path="/root/paddlejob/workspace/code/dd"
tags_path="tops_labels.csv"
dist.init_parallel_env()
modelG=model.G_NET()
modelDs=[model.D_NET64(),model.D_NET128(),model.D_NET256()]

modelG=paddle.DataParallel(modelG)
for i in range(3):
    modelDs[i]=paddle.DataParallel(modelDs[i])
print("-----------------------------model successfully-------------------------------------")
    
train_dataset=dataset.MyDataSet(images_path,tags_path)

train_loader =  DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True,drop_last=True,num_workers =4)


optim_ds=[]
for i in range(3):
    od=paddle.optimizer.Adam(learning_rate=dlr,beta1=0.5,parameters=modelDs[i].parameters())
    optim_ds.append(od)


optim_g = paddle.optimizer.Adam(learning_rate=glr,beta1=0.5,parameters=modelG.parameters())
real_labels = paddle.ones(shape=[BATCH_SIZE,1])
fake_labels = paddle.zeros(shape=[BATCH_SIZE,1])

from visualdl import LogWriter

import IPython.display as display
DPaht ="/root/paddlejob/workspace/output/vi/errD"  
GPaht ="/root/paddlejob/workspace/output/vi/errG"  
Dwriter = LogWriter(logdir=DPaht)  #判别器可视化路径
Gwriter = LogWriter(logdir=GPaht)  #生成器可视化路径
TD="/root/paddlejob/workspace/output/vi/total_errD"
TG="/root/paddlejob/workspace/output/vi/total_errG"
TDwriter = LogWriter(logdir=TD)  #判别器可视化路径
TGwriter = LogWriter(logdir=TG)  #生成器可视化路径

gd="/root/paddlejob/workspace/code/model/G"
dd="/root/paddlejob/workspace/code/model/D"
print("-----------------------------visual successfully-------------------------------------")
class train_the_Gan(object):
    def __init__(self, data_loader,Gd,Dd,pre_train=0):
        self.G_dir=Gd
        self.D_dir=Dd
        self.num_Ds=3
        if pre_train==1:
            Gen_p=paddle.load(os.path.join(gd,"model_g"))
            Dis_p1=paddle.load(os.path.join(dd,'0',"model_d"))
            Dis_p2=paddle.load(os.path.join(dd,'1',"model_d"))
            Dis_p3=paddle.load(os.path.join(dd,'2',"model_d"))
            modelG.set_state_dict(Gen_p)
            modelDs[0].set_state_dict(Dis_p1)
            modelDs[1].set_state_dict(Dis_p2)
            modelDs[2].set_state_dict(Dis_p3)
        
        
    def train_Dnet(self, idx, count,real_imgs,wrong_imgs,fake_imgs,c_code):
       
        flag = count % 100
        batch_size = BATCH_SIZE
        fake_imgs.stop_gradient=True

   
        
        # Forward
        
        # for real
        real_logits = modelDs[idx](real_imgs, c_code)
        wrong_logits = modelDs[idx](wrong_imgs, c_code)
        fake_logits = modelDs[idx](fake_imgs, c_code)
        #
        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)
        
        errD_real_uncond = criterion(real_logits[1], real_labels)
        errD_wrong_uncond = criterion(wrong_logits[1], real_labels)
        errD_fake_uncond = criterion(fake_logits[1], fake_labels)
            #
        errD_real = errD_real + errD_real_uncond
        errD_wrong = errD_wrong + errD_wrong_uncond
        errD_fake = errD_fake + errD_fake_uncond
            #
        errD = errD_real + errD_wrong + errD_fake
        
        #errD = errD_real + 0.5 * (errD_wrong + errD_fake)
        # backward
        errD=errD/gradient_acc
        errD.backward()
        # update parameters
        if count==0:
            optim_ds[idx].step()
            optim_ds[idx].clear_grad()
        # log
        
                
        Dwriter.add_scalar(tag="train/Dloss%d" %idx, step=count//100, value=errD)
        return errD
        

    def train_Gnet(self, count,fake_imgs,c_code):
      
        
        errG_total = 0
        flag = count % 100
        batch_size = BATCH_SIZE
        
        
        for i in range(3):
            fake_imgs[i].stop_gradient=False
            outputs = modelDs[i](fake_imgs[i],c_code)
            errG = criterion(outputs[0], real_labels)
            
            errG_patch = criterion(outputs[1], real_labels)
            errG = errG + errG_patch
            errG_total = errG_total + errG
            
                    
            Gwriter.add_scalar(tag="train/Gloss%d" % i, step=count//100, value=errG)

        
        errG_total=errG_total/gradient_acc
        errG_total.backward()
        if count==0:
            optim_g.step()
            optim_g.clear_grad()
        return errG_total
    def train(self):
        modelG.train()
        modelDs[0].train()
        modelDs[1].train()
        modelDs[2].train()
        count=0
        optim_g.clear_grad()
        for i in range(3):
            optim_ds[i].clear_grad()
            
        for epoch_idx in range(epochs):
            
            ri=[1,1,1]
            wi=[1,1,1]
            for batch_idx, (ri[0], ri[1], ri[2],wi[0],wi[1],wi[2],c_code) in enumerate(train_loader):
                count=count+1
                
                noise = paddle.normal(shape=[BATCH_SIZE,nz])
                fake_imgs,_=modelG(noise,c_code)
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, (batch_idx+1)%gradient_acc,ri[i],wi[i],fake_imgs[i],c_code)
                    errD_total += errD

                errG_total = self.train_Gnet((batch_idx+1)%gradient_acc,fake_imgs,c_code=c_code)
                
                TDwriter.add_scalar(tag="train/tDloss",step=count//100,value=errD_total)
                TGwriter.add_scalar(tag="train/tGloss",step=count//100,value=errG_total)
                if batch_idx % print_every == 0:
                    print("\t{:d} ({:d} / {:d}) D loss = {:.4f},G loss== {:.4f}".format(epoch_idx, batch_idx, len(train_loader), errD_total.item(),errG_total.item()))

            if epoch_idx%20==0:
                pre_im=paddle.transpose(fake_imgs[2],[0,2,3,1])
                n_row = min(int(np.sqrt(BATCH_SIZE)), 10)
                images = np.zeros((256 * n_row, 256 * n_row, 3))
                for h in range(n_row):
                    for w in range(n_row):
                        images[h * 256:(h + 1) * 256, w * 256:(w + 1) * 256] = pre_im[(h * n_row) + w]
                images=(images+1)/2
                plt.imsave('/root/paddlejob/workspace/output/tmp_images/{}.jpg'.format(epoch_idx), images)
             
                for i in range(3):
                    paddle.save(modelDs[i].state_dict(), os.path.join(self.D_dir,str(i),"model_d"))
                   
                    #+str(epoch_idx)
                
                paddle.save(modelG.state_dict(), os.path.join(self.G_dir,"model_g"+str(epoch_idx//20)))
                  

trainG=train_the_Gan(data_loader=train_loader,Gd="/root/paddlejob/workspace/output/model/G",Dd="/root/paddlejob/workspace/output/model/D",pre_train=1)
print("-------------------------------------------Start Train----------------------------------------")
trainG.train()

        
                
      
        

    
