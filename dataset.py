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

paddle.in_dynamic_mode()
class MyDataSet(Dataset):
    def __init__(self, images_path, tags_path):
        self.images_path = images_path
        self.tags_path = tags_path
        
        df_tags=pd.read_csv(tags_path,skiprows=1,header=None)
        self.images = [self.images_path+str(x) for x in df_tags[0]]

        self.c_code=df_tags.iloc[0:,1:].values.astype("float32")
        self.l=len(self.images)

        
    
    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):

        #print(idx)
        # 使用Pillow来读取图像数据
        # if(os.path.exists("/root/paddlejob/workspace/code/dd/dataset/bodies/000200a4-4fa7-4ac4-87ff-0f5bc6ac30b0.jpg")==False):
        #     print(os.getcwd())
        #     return 0
        # else:
        #     print("s")
        while(os.path.exists(self.images[idx])==False):
            print("f1")
            idx=(idx+1)%self.l;
        img = Image.open(self.images[idx])
        #img=np.array(img).astype("float64")
        ccode=self.c_code[idx]
        img=T.pad(img,padding=[44,0,45,0],padding_mode="edge")
        img=T.Normalize(mean=127.5, std=127.5,data_format="HWC")(img)
        trans1=T.Compose([T.Resize(size=(64,64)),T.Transpose()])
        trans2=T.Compose([T.Resize(size=(128,128)),T.Transpose()])
        trans3=T.Compose([T.Resize(size=(256,256)),T.Transpose()])
        while(os.path.exists(self.images[(idx+1)%self.l])==False):
            print("f2")
            idx=(idx+1)%self.l;

        fimg = Image.open(self.images[(idx+1)%self.l])
        #fimg=np.array(fimg).astype("float64")
        fimg=T.pad(fimg,padding=[44,0,45,0],padding_mode="edge")
        fimg=T.Normalize(mean=127.5, std=127.5,data_format="HWC")(fimg)
        #noise = paddle.normal(shape=[nz])
        

        return trans1(img),trans2(img),trans3(img),trans1(fimg),trans2(fimg),trans3(fimg),ccode
