# coding:--utf-8--
#对uts,其wavelet coef送入LSTM,输出等长概率，和原coef相乘后，逆变换回ts,求mse.
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


def normalization(arr): #对coef进行归一化 同ts处理方法一致，均归一化至[-1，1]

    fmin=arr.min()
    fmax=arr.max()
    result=-1+2*(arr-fmin)/(fmax-fmin)
    return result

def normalization2(arr): #对coef进行归一化 同ts处理方法一致，均归一化至[-1，1]

    fmin=arr.min()
    fmax=arr.max()
    result=(arr-fmin)/(fmax-fmin)
    return result


def gain_coef_v2(ts_tensor, xfm):

    yl,yh=xfm(ts_tensor)
    coef_list=[yl]+yh
    coef_tensor=torch.cat(coef_list,dim=2)  #在t（即len）维度拼接
    # coef_tensor=torch.transpose(coef_tensor,1,2)  #后续LSTM model适用
    # print(len(yh))
    # print(yh[0].shape,yh[1].shape,yh[2].shape)  #看DWT1D系数的shape (b,f=1,t) t=len(coef)
    # exit()
    # total_coef=[]
    #
    # for i in range(ts_tensor.shape[0]):
    #     coef_list=pywt.wavedec(ts_tensor[i, 0, :], xfm)
    #     new_list=[x.reshape(1,len(x)) for x in coef_list]
    #     new_array=np.concatenate(new_list,axis=1)
    #     new_array=new_array[None,:,:]  #batch升维
    #     total_coef.append(new_array)  #将所有系数拼接成一维array
    # final_coef=np.concatenate(total_coef,axis=0)
    # print(final_coef.shape)
    # with open(outp+indicator+'coef.pkl','wb') as fc:
    #     pickle.dump(final_coef,fc)
    return  coef_tensor   #(b,1,len)

def shape_tune(coef_tensor,len_list):
    #CNN不用转置
    # coef_tensor=torch.transpose(coef_tensor,1,2) #将LSTM结果转置
    yl=coef_tensor[:,:,:len_list[0]]
    yh=[]
    start_idx=len_list[0]
    for i in range(1,len(len_list)):
        cur_coef=coef_tensor[:,:,start_idx:start_idx+len_list[i]]
        yh.append(cur_coef)
        start_idx+=len_list[i]
    result=(yl,yh)
    return result

def statistical_features(arr):#接收某一级的coef_arr

    std = torch.std(arr, dim=2, keepdim=True)
    energy=arr.pow(2).sum(dim=2,keepdim=True) #1080无torch.square函数
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
        weight = energy  # 长度为1时，无std
    return weight

def process_coef(coef,len_list): #接收gain_coef_v2得到的coef和每级的coef长度，根据每级的统计特征处理coef
    start_id=0
    end_id=-1
    result=torch.empty_like(coef)
    weight_tenor=torch.empty_like(coef)
    for i in range(len(len_list)):
        end_id=start_id+len_list[i]
        level_coef=coef[:,:,start_id:end_id]
        weight=statistical_features(level_coef)
        # weight=std*energy*dif_sum*form_factor
        if i==0:#单独处理首个yl
            weight/=(len(len_list)-1)
        else:  #除以级数
            weight/=i
        weight_tenor[:,:,start_id:end_id]=weight
        start_id=end_id

    norm_weight=normalization2(weight_tenor)  #weight归一化至0-1,VS coef归一化至-1~1
    result=coef*norm_weight
    return result