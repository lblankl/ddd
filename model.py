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

Z_DIM=160
CONDITION_DIM=249 #249或 153
R_NUM=2
GF_DIM=64
DF_DIM=64

def hw_flatten(x):
    b, c, h, w = x.shape
    x =  paddle.reshape(x, shape=(b, c, h*w))
    return x
class Attention(nn.Layer):
    def __init__(self, out_channels):
        super(Attention, self).__init__()
        
        self.out_channel=out_channels
        self.F=nn.Conv2D(in_channels=out_channels,out_channels=out_channels//8,kernel_size=1,stride=1,padding='SAME')
        self.G=nn.Conv2D(in_channels=out_channels,out_channels=out_channels//8,kernel_size=1,stride=1,padding='SAME')
        self.H=nn.Conv2D(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,padding='SAME')
        self.O=nn.Conv2D(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,padding='SAME')
        self.softmax=nn.Softmax()
    def forward(self,x):
        
        n_, c_, h_, w_ = x.shape
        f=self.F(x)

        g = self.G(x)

        h = self.H(x)

        s = paddle.matmul(hw_flatten(f), hw_flatten(g), transpose_x=True)

        attention_ = self.softmax(s)

        o = paddle.matmul(hw_flatten(h), attention_) # b c n
        o = paddle.reshape(o, shape=(n_, self.out_channel, h_, w_)) # b c h w
        o = self.O(o)
        gamma = paddle.static.create_parameter(shape=[1], name='name1'+'gamma', dtype='float32', default_initializer=nn.initializer.Constant(value=0.0))
        x = gamma * o + x
        return x

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return  nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias_attr=False)
                
class GLU(paddle.nn.Layer):
    def __init__(self):
        super(GLU, self).__init__()
    def forward(self, x):
        return nn.functional.glu(x,axis=1)


def upBlock(in_planes, out_planes):
    block=nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes,out_planes*2),
        nn.BatchNorm2D(out_planes*2),
        GLU()
    )
    return block

def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2D(out_planes*2),
        GLU()
    )
    return block


class ResBlock(nn.Layer):
    def __init__(self,channel_num):
        super(ResBlock, self).__init__()
        self.block=nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2D(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2D(channel_num)
        )
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class INIT_STAGE_G(nn.Layer):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        
        self.in_dim = Z_DIM + CONDITION_DIM
        
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias_attr=False),
            nn.BatchNorm1D(ngf * 4 * 4 * 2),
            GLU())


        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.att=Attention(ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code=None):
      
        in_code = paddle.concat(x=[c_code, z_code], axis=1)
        
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code=paddle.reshape(out_code,[-1,self.gf_dim,4,4])
      
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.att(out_code)
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Layer):
    def __init__(self, ngf, num_residual=R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        
        self.ef_dim = CONDITION_DIM
       
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.att=Attention(ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.shape[2]
        c_code=paddle.reshape(c_code,shape=[-1,self.ef_dim,1,1])
        c_code=paddle.tile(c_code,repeat_times=[1,1,s_size,s_size])
        
        # state size (ngf+egf) x in_size x in_size
        h_c_code = paddle.concat([c_code, h_code], axis=1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code=self.att(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code


class GET_IMAGE_G(nn.Layer):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Layer):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = GF_DIM
        self.define_module()

    def define_module(self):
        
        #self.ca_net = CA_NET()

        
        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
        self.img_net1 = GET_IMAGE_G(self.gf_dim)
        
        self.h_net2 = NEXT_STAGE_G(self.gf_dim)
        self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        
        self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2)
        self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
        

    def forward(self, z_code, c_code):
        
       
        fake_imgs = []
        
        h_code1 = self.h_net1(z_code, c_code)
        fake_img1 = self.img_net1(h_code1)
        fake_imgs.append(fake_img1)
        
        h_code2 = self.h_net2(h_code1, c_code)
        fake_img2 = self.img_net2(h_code2)
        fake_imgs.append(fake_img2)
        
        h_code3 = self.h_net3(h_code2, c_code)
        fake_img3 = self.img_net3(h_code3)
        fake_imgs.append(fake_img3)
        

        return fake_imgs,c_code


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2D(out_planes),
        nn.LeakyReLU(0.2)
    )
    return block


def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2D(in_planes, out_planes, 4, 2, 1, bias_attr=False),
        nn.BatchNorm2D(out_planes),
        nn.LeakyReLU(0.2)
    )
    return block


def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2D(3, ndf, 4, 2, 1, bias_attr=False),
        nn.LeakyReLU(0.2),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2D(ndf, ndf * 2, 4, 2, 1, bias_attr=False),
        nn.BatchNorm2D(ndf * 2),
        nn.LeakyReLU(0.2),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2D(ndf * 2, ndf * 4, 4, 2, 1, bias_attr=False),
        nn.BatchNorm2D(ndf * 4),
        nn.LeakyReLU(0.2),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2D(ndf * 4, ndf * 8, 4, 2, 1, bias_attr=False),
        nn.BatchNorm2D(ndf * 8),
        nn.LeakyReLU(0.2)
    )
    return encode_img

class D_NET64(nn.Layer):
    def __init__(self):
        super(D_NET64, self).__init__()
        self.df_dim = DF_DIM
        self.ef_dim = CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.att=Attention(ndf*8)

        self.logits = nn.Sequential(
            nn.Conv2D(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
                nn.Conv2D(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code=self.att(x_code)
        
        c_code = paddle.reshape(c_code,[-1, self.ef_dim, 1, 1])
        c_code=paddle.tile(c_code,repeat_times=[1,1,4,4])
      
            # state size (ngf+egf) x 4 x 4
        h_c_code = paddle.concat(x=[c_code, x_code], axis=1)
            # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        

        output = self.logits(h_c_code)
        output=paddle.reshape(output,[-1,1])
        
        out_uncond = self.uncond_logits(x_code)

        out_uncond=paddle.reshape(out_uncond,[-1,1])
        return output, out_uncond


class D_NET128(nn.Layer):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = DF_DIM
        self.ef_dim = CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.att=Attention(ndf*8)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2D(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
        nn.Conv2D(ndf * 8, 1, kernel_size=4, stride=4),
        nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code=self.att(x_code)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        
        c_code = paddle.reshape(c_code,[-1, self.ef_dim, 1, 1])
        c_code=paddle.tile(c_code,repeat_times=[1,1,4,4])
            # state size (ngf+egf) x 4 x 4
        h_c_code = paddle.concat(x=[c_code, x_code], axis=1)
            # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
       

        output = self.logits(h_c_code)
        output=paddle.reshape(output,[-1,1])
    
        out_uncond = self.uncond_logits(x_code)
        out_uncond=paddle.reshape(out_uncond,[-1,1])
        return output, out_uncond
        


class D_NET256(nn.Layer):
    def __init__(self):
        super(D_NET256, self).__init__()
        self.df_dim = DF_DIM
        self.ef_dim = CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.att=Attention(ndf*8)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2D(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

       
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
                nn.Conv2D(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code=self.att(x_code)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        
        c_code = paddle.reshape(c_code,[-1, self.ef_dim, 1, 1])
        c_code=paddle.tile(c_code,repeat_times=[1,1,4,4])
            # state size (ngf+egf) x 4 x 4
        h_c_code = paddle.concat(x=[c_code, x_code], axis=1)
            # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        
        output = self.logits(h_c_code)
        output=paddle.reshape(output,[-1,1])
        
        out_uncond = self.uncond_logits(x_code)
        out_uncond=paddle.reshape(out_uncond,[-1,1])
        return output, out_uncond
        
