# coding:--utf-8--

import sys
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import os
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))

fatherpath = os.path.split(curPath)[0]
grandpath=os.path.split(fatherpath)[0]
greatgrandpath=os.path.split(grandpath)[0]
gggpath=os.path.split(greatgrandpath)[0]

sys.path.append(fatherpath)
sys.path.append(curPath)
sys.path.append(grandpath)
sys.path.append(greatgrandpath)
sys.path.append(gggpath)
sys.path.append(os.path.split(gggpath)[0])

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR
import pickle

from functools import reduce
from torch.nn import BatchNorm1d
import time
import gc
import pywt
from pytorch_wavelets import DWT1DForward,DWT1DInverse
np.random.seed(7)
torch.manual_seed(1)


class A2_Ratio(nn.Module):
    def __init__(self,window):
        super(A2_Ratio,self).__init__()
        self.dense1=nn.Linear(int(2*window),window)

        self.activation1=nn.Tanh()
        self.dense2=nn.Linear(window,1)
        self.activation2=nn.Sigmoid()

        self.ratio=nn.Sequential(self.dense1,self.activation1,self.dense2,self.activation2)

    def forward(self,x1,x2):
        x=torch.cat([x1,x2],dim=2) 
        relative_ratio=self.ratio(x).squeeze() #（b,f）

        return relative_ratio 


class A2_Reconstructor(nn.Module):
    def __init__(self, t_nc, window, c_in):
        super(A2_Reconstructor, self).__init__()
        dim_in=int(c_in*window)
        dim_out=int(t_nc * window)
        self.encoder = nn.Sequential(
            # input is (nc) x 64
            nn.Linear(dim_in, 2),
            nn.Tanh(),
            nn.Linear(2, 1),

        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.Tanh(),
            nn.Linear(2, dim_out)
        )


       

    def forward(self, temporal_x):
       
        z = self.encoder(temporal_x)
        x_rec = self.decoder(z)
  

        return x_rec #, weight

class ECG_Encoder(nn.Module):
    def __init__(self, opt):
        super(ECG_Encoder, self).__init__()

        self.main = nn.Sequential(
           
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv1d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
         
            nn.Conv1d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv1d(opt.ndf * 16, opt.nz, 10, 1, 0, bias=False),
           
        )

    def forward(self, input):

        output = self.main(input)

        return output

class ECG_Decoder(nn.Module):
    def __init__(self, opt):
        super(ECG_Decoder, self).__init__()

        self.main=nn.Sequential(
            
            nn.ConvTranspose1d(opt.nz,opt.ngf*16,10,1,0,bias=False),
            nn.BatchNorm1d(opt.ngf*16),
            nn.ReLU(True),
         
            nn.ConvTranspose1d(opt.ngf * 16, opt.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 8),
            nn.ReLU(True),
         
            nn.ConvTranspose1d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True),
          
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf*2),
            nn.ReLU(True),
          
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf , 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf ),
            nn.ReLU(True),
          
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
          
        )

    def forward(self, input):

        output = self.main(input)
        return output

class ECG_Reconstructor(nn.Module):
    def __init__(self, opt,ts_cin,nc):
        super(ECG_Reconstructor, self).__init__()
        self.encoder = ECG_Encoder(opt)
        self.decoder = ECG_Decoder(opt)

    def forward(self, temporal_x):
       
        z=self.encoder(temporal_x)
        x_rec=self.decoder(z)
    

        return x_rec

class ECG_Ratio(nn.Module):
    def __init__(self, window, ratio_nc):
        super(ECG_Ratio, self).__init__()
        self.conv1 = nn.Conv1d(2, ratio_nc, 4, 2, 1, bias=False)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(ratio_nc, ratio_nc * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv1d(ratio_nc * 2, ratio_nc * 3, 6, 4, 1, bias=False)
        self.conv4 = nn.Conv1d(ratio_nc * 3, ratio_nc * 4, 6, 4, 1, bias=False)
        self.conv5 = nn.Conv1d(ratio_nc * 4, ratio_nc * 5, 5, 1, 0, bias=False)
        self.conv6 = nn.Conv1d(ratio_nc * 5, 1, 1, 1, 0, bias=False)

        self.activation2 = nn.Sigmoid()

        self.ratio = nn.Sequential(self.conv1, self.activation1, self.conv2, self.activation1,
                                   self.conv3, self.activation1, self.conv4, self.activation1,
                                   self.conv5, self.activation1, self.conv6, self.activation2)

    def forward(self, x1, x2):  
        x = torch.cat([x1, x2], dim=1)  
        relative_ratio=self.ratio(x).squeeze() #（b,f）

        return relative_ratio 


class KPI_Ratio(nn.Module):
    def __init__(self, window, ratio_nc):
        super(KPI_Ratio, self).__init__()
        self.conv1 = nn.Conv1d(2, ratio_nc, 4, 2, 1, bias=False)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(ratio_nc, 2 * ratio_nc, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv1d(2 * ratio_nc, 3 * ratio_nc, 5, 3, 1, bias=False)
        self.conv4 = nn.Conv1d(3 * ratio_nc, 4 * ratio_nc, window // 12, 1, 0, bias=False)
        self.conv5 = nn.Conv1d(4 * ratio_nc, 1, 1, 1, 0, bias=False)
        self.activation2 = nn.Sigmoid()

        self.ratio = nn.Sequential(self.conv1, self.activation1, self.conv2, self.activation1,
                                   self.conv3, self.activation1, self.conv4, self.activation1,
                                   self.conv5, self.activation2)

    def forward(self, x1, x2):  
        x = torch.cat([x1, x2], dim=1)  
        relative_ratio=self.ratio(x).squeeze() #（b,f）

        return relative_ratio



class KPI_Reconstructor(nn.Module):
    def __init__(self, opt_ae,ts_cin,nc):
        super(KPI_Reconstructor, self).__init__()
        self.encoder = nn.Sequential(
      
            nn.Conv1d(opt_ae.c_in, opt_ae.freq_cout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
       
            nn.Conv1d(opt_ae.freq_cout, opt_ae.freq_cout * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt_ae.freq_cout * 2),
            nn.LeakyReLU(0.2, inplace=True),
         
            nn.Conv1d(opt_ae.freq_cout * 2, opt_ae.freq_cout * 4, 5, 3, 1, bias=False),
            nn.BatchNorm1d(opt_ae.freq_cout * 4),
            nn.LeakyReLU(0.2, inplace=True),
       



        
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4 * opt_ae.freq_cout, 2 * opt_ae.freq_cout, 5, 3, 1, bias=False),
            nn.BatchNorm1d(opt_ae.freq_cout * 2),
            nn.ReLU(True),
          
            nn.ConvTranspose1d(2 * opt_ae.freq_cout, opt_ae.freq_cout, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt_ae.freq_cout),
            nn.ReLU(True),
         
            nn.ConvTranspose1d(opt_ae.freq_cout, opt_ae.c_in, 4, 2, 1, bias=False),

            nn.Tanh()

        )

    def forward(self, temporal_x):
       
        z=self.encoder(temporal_x)
        x_rec=self.decoder(z)
       

        return x_rec

