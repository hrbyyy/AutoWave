# coding:--utf-8--
#
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
# extra_path='/home/ices/PycharmProject/shape_sequence_kpi/uts/conditional_conv/model_combine/'
sys.path.append(fatherpath)
sys.path.append(curPath)
sys.path.append(grandpath)
sys.path.append(greatgrandpath)
sys.path.append(gggpath)
sys.path.append(os.path.split(gggpath)[0])
# sys.path.append(extra_path)
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR
import pickle
# import matplotlib.pyplot as plt
from functools import reduce
from torch.nn import BatchNorm1d
import time
import gc
import pywt
from pytorch_wavelets import DWT1DForward,DWT1DInverse
import random


def normalization(arr): 

    fmin=arr.min()
    fmax=arr.max()
    result=-1+2*(arr-fmin)/(fmax-fmin)
    return result

def normalization2(arr): 
    fmin=arr.min()
    fmax=arr.max()
    result=(arr-fmin)/(fmax-fmin)
    return result


def gain_coef_v2(ts_tensor, xfm):

    yl,yh=xfm(ts_tensor)
    coef_list=[yl]+yh
    coef_tensor=torch.cat(coef_list,dim=2)  
  
    return  coef_tensor   #(b,1,len)

def shape_tune(coef_tensor,len_list):
   
    yl=coef_tensor[:,:,:len_list[0]]
    yh=[]
    start_idx=len_list[0]
    for i in range(1,len(len_list)):
        cur_coef=coef_tensor[:,:,start_idx:start_idx+len_list[i]]
        yh.append(cur_coef)
        start_idx+=len_list[i]
    result=(yl,yh)
    return result

def statistical_features(arr):#

    std = torch.std(arr, dim=2, keepdim=True)
    energy=arr.pow(2).sum(dim=2,keepdim=True) #
    # energy = torch.square(arr).sum(dim=2, keepdim=True)
    if arr.shape[2] > 2:
        dif = arr[:, :, 1:arr.shape[2]] - arr[:, :, 0:arr.shape[2] - 1]
        dif_second = dif[:, :, 1:dif.shape[2]] - dif[:, :, dif.shape[2] - 1]
        dif_sum = torch.abs(dif).sum(dim=2, keepdim=True)
        mobility_x = torch.sqrt(torch.var(dif, dim=2, keepdim=True) / torch.var(arr, dim=2, keepdim=True))
        mobility_dif = torch.sqrt(torch.var(dif_second) / torch.var(dif))
        form_factor = mobility_dif / mobility_x
        weight = std * energy * dif_sum * form_factor

    elif arr.shape[2] > 1:
        dif = arr[:, :, 1:arr.shape[2]] - arr[:, :, 0:arr.shape[2] - 1]
        dif_sum = torch.abs(dif).sum()
        weight = std * energy * dif_sum
    else:
        weight = energy  # 
    return weight

def process_coef(coef,len_list): #
    start_id=0
    end_id=-1
    result=torch.empty_like(coef)
    weight_tenor=torch.empty_like(coef)
    for i in range(len(len_list)):
        end_id=start_id+len_list[i]
        level_coef=coef[:,:,start_id:end_id]
        weight=statistical_features(level_coef)
        # weight=std*energy*dif_sum*form_factor
        if i==0:#
            weight/=(len(len_list)-1)
        else:  #
            weight/=i
        weight_tenor[:,:,start_id:end_id]=weight
        start_id=end_id

    norm_weight=normalization2(weight_tenor)  #
    result=coef*norm_weight
    return result
