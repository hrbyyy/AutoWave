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
from train import train,val
from predict import inference
from model import KPI_Reconstructor,KPI_Ratio


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything() 



def form_dataloader(batchsize,inpath):
    trainx_new=pickle.load(open(inpath+'trainx.pkl','rb'))
    val1x_new=pickle.load(open(inpath+'val1x.pkl','rb'))
    val2x_new = pickle.load(open(inpath + 'val2x.pkl', 'rb'))
    testx_new = pickle.load(open(inpath + 'testx.pkl', 'rb'))


    train_x = torch.from_numpy(trainx_new).type(torch.FloatTensor)
    val1_x = torch.from_numpy(val1x_new).type(torch.FloatTensor)
    val2_x = torch.from_numpy(val2x_new).type(torch.FloatTensor)
    test_x = torch.from_numpy(testx_new).type(torch.FloatTensor)
    
    torch_trainset = Data.TensorDataset(train_x, train_x)
    
    trainloader = Data.DataLoader(
        dataset=torch_trainset,  # torch TensorDataset format
        batch_size=batchsize,  # mini batch size
        shuffle=True  
    )
    val1data = Data.TensorDataset(val1_x, val1_x)
    val1loader = Data.DataLoader(dataset=val1data, shuffle=True, batch_size=batchsize)
    val2data = Data.TensorDataset(val2_x, val2_x)
    val2loader = Data.DataLoader(dataset=val2data, shuffle=False, batch_size=batchsize)
    testdata = Data.TensorDataset(test_x, test_x)
    testloader = Data.DataLoader(dataset=testdata, shuffle=False, batch_size=batchsize)

    return trainloader,val1loader,val2loader,testloader

class Option_pp:   
    def __init__(self,c_in,nc):
        self.c_in=c_in
        self.freq_cout=nc


if __name__ == "__main__":

    repeats = 5
    epoches = 100 

    d_clip = 0.01

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    batchsize = 128


    patience = 20  
    threshold = 2 

    inpath = '/mnt/A/dataset/kpi/final_for_model/original/April/reconstruction/'
    window = 60
    inpath=inpath+'window'+str(window)+'/'  

    wavelet_name = 'haar'
    wavelet = pywt.Wavelet('haar')
    wavelet_len = wavelet.dec_len 
    
    n_level=2
    xfm = DWT1DForward(J=n_level,wave='haar').cuda()
    ifm=DWT1DInverse(wave='haar').cuda()



    len_list = []
    level_list = list(range(n_level + 1))
    cur_len = window
    total_len = 0  
    for i in range(n_level):
        new_len = (cur_len + wavelet_len - 1) // 2
        len_list.append(new_len)
        total_len += new_len
        cur_len = new_len
    len_list=[cur_len]+len_list   
    total_len+=cur_len   
    print(total_len)   
   
    ts_cin=1
    nc=4
    ratio_nc=4
   
    opt_ae=Option_pp(c_in=1,nc=32)

    lr=5*1e-4


    p0 = '/mnt/A/PycharmProject/wavelet_rec/kpi/double_inputs/conv/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)

    p0=p0+'window'+str(window)+'/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)
    p0=p0+wavelet_name+'/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)
    p0=p0+'level'+str(n_level)+'/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)
  
    p0 = p0 + 'RMSprop/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)
    # for lr in lr_list:
    rootp=p0+'learning_rate'+str(lr)+'/'
    if os.path.exists(rootp)==False:
        os.mkdir(rootp,0o777)
    selected_ts = [16, 26, 27]  
    for ts_id in selected_ts:

        inpath_ts=inpath+str(ts_id)+'/'


     
        ppath = rootp+str(ts_id)+'/'
        if os.path.exists(ppath) == False:
            os.mkdir(ppath, 0o777)

        for r in range(repeats):
            outp = ppath + str(r) + '/'
            if os.path.exists(outp) == False:
                os.mkdir(outp, 0o777)

            reconstructor = KPI_Reconstructor(opt_ae,ts_cin,nc)

            reconstructor = nn.DataParallel(reconstructor)
            reconstructor.to(device)
            ratio_calculator=KPI_Ratio(window,ratio_nc)
            ratio_calculator=nn.DataParallel(ratio_calculator)
            ratio_calculator.to(device)
           
            params = ([a for a in reconstructor.parameters()] + [b for b in ratio_calculator.parameters()])

            optimizer = optim.RMSprop(params, lr= lr)

           

            vloss = float('inf') 
            losst = [] 

            lossv = []

            ep = -1
            counter = 0
            total_running = epoches 
            epoch_train_times = []
            for epoch in range(epoches):  
                # scheduler.step()  
                trainloader, val1loader, val2loader, testloader = form_dataloader(batchsize,inpath_ts)
                del val2loader
                del testloader
                gc.collect()
                torch.cuda.synchronize()
                start = time.clock()
                losst = train(reconstructor, xfm, ifm, device, trainloader, ratio_calculator,epoch, losst, optimizer, len_list) 
                del trainloader
                gc.collect()
                torch.cuda.synchronize()
                end = time.clock()
                epoch_train_time = end - start
                epoch_train_times.append(epoch_train_time)
              
                lossv,  vloss, ep, counter = val(reconstructor, xfm, ifm, window, device, ratio_calculator,val1loader, epoch, ep, lossv, vloss, counter, patience, threshold, batchsize,len_list,outp, optimizer)
                del val1loader
                gc.collect()
                if counter == patience:
                    total_running = epoch + 1 
                    break
                torch.cuda.empty_cache()
    

            time_data = np.asarray(epoch_train_times).T
            df_train_time = pd.DataFrame(columns=['epoch_train_time'], data=time_data)
            df_train_time.loc['mean', 'epoch_train_time'] = df_train_time['epoch_train_time'].mean()
            df_train_time.to_csv(outp + 'epoch_train_time.csv')

            trainloader, val1loader, val2loader, testloader = form_dataloader(batchsize, inpath_ts)
            del trainloader
            del val1loader
            gc.collect()
            inference(device,val2loader, opt_ae,ts_cin,nc, xfm,ifm,window,ratio_nc, batchsize,len_list, outp, 'val2')
            del val2loader
            inference(device, testloader,opt_ae,ts_cin,nc, xfm,ifm,window,ratio_nc,batchsize,len_list, outp, 'test')  
            del testloader
            gc.collect()
            data_lt = np.asarray(losst).reshape(len(losst), 1)
            data_lv = np.asarray(lossv).reshape(len(lossv), 1)

            data_l = np.hstack((data_lt, data_lv))
            df_loss = pd.DataFrame(data=data_l, columns=['train_loss', 'val_loss'])
            df_loss.to_csv(outp + 'epoch_loss.csv')
            torch.cuda.empty_cache()
            gc.collect()




