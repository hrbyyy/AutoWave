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


from train import train,val
from predict import inference
from model import A2_Reconstructor,A2_Ratio

np.random.seed(7)
torch.manual_seed(1)

datapath='/mnt/A/dataset/Yahoo A1A2/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/'



def form_dataloader(batchsize,inpath):
    trainx_new=pickle.load(open(inpath+'trainx.pkl','rb'))
    val1x_new=pickle.load(open(inpath+'val1x.pkl','rb'))
    val2x_new = pickle.load(open(inpath + 'val2x.pkl', 'rb'))
    testx_new = pickle.load(open(inpath + 'testx.pkl', 'rb'))


    train_x = torch.from_numpy(trainx_new).type(torch.FloatTensor)
    val1_x = torch.from_numpy(val1x_new).type(torch.FloatTensor)
    val2_x = torch.from_numpy(val2x_new).type(torch.FloatTensor)
    test_x = torch.from_numpy(testx_new).type(torch.FloatTensor)
    # print(train_x.shape)  (b,f=2,t)
    # exit()
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
    # with open(inpath + 'trainloader.pkl', 'wb') as f7:
    #     pickle.dump(trainloader, f7, protocol=4)
    # with open(inpath +  'val1loader.pkl', 'wb') as f8:
    #     pickle.dump(val1loader, f8, protocol=4)
    # with open(inpath +  'val2loader.pkl', 'wb') as f9:
    #     pickle.dump(val2loader, f9, protocol=4)
    # with open(inpath +  'testloader.pkl', 'wb') as f10:
    #     pickle.dump(testloader, f10, protocol=4)
    # return  train_x,val1_x,val2_x,test_x
    return trainloader,val1loader,val2loader,testloader

class Options:
    def __init__(self, nc,ndf,ngf,nz):
        self.nc = nc
        self.ndf=ndf
        self.ngf=ngf
        self.nz=nz



if __name__ == "__main__":

    repeats = 5
    epoches = 200  #

    d_clip = 0.01

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #

    batchsize = 64


    patience = 20  #
    threshold = 2  # 


    window =4
    ratio_nc=4
    datapath=datapath+'window'+str(window)+'/'
    wavelet_name = 'haar'
    wavelet = pywt.Wavelet('haar')
    wavelet_len = wavelet.dec_len  # 2

    # print(len(morl))
    # n_level = pywt.dwt_max_level(window, wavelet.dec_len)  # 6

    n_level=2 #
    xfm = DWT1DForward(J=n_level,wave='haar').cuda()
    ifm=DWT1DInverse(wave='haar').cuda()



    len_list = []
    level_list = list(range(n_level + 1))
    cur_len = window
    total_len = 0  # 
    for i in range(n_level):
        new_len = (cur_len + wavelet_len - 1) // 2
        len_list.append(new_len)
        total_len += new_len
        cur_len = new_len
    len_list=[cur_len]+len_list    
    total_len+=cur_len   
    print(total_len)   
    # len_list.append(cur_len)  
    # len_list.reverse()  

    # input_feature=len(scale_list)
    # hidden_units = 64
    # n_layer = 1
    # dropout = 0  #
    ts_cin=1
    freq_cout=4   #freq_tuner中对应的c_out
    nc=1
    lr=1e-3
  

    p0 = '/mnt/A/PycharmProject/wavelet_rec/A2/double_inputs/8-4-1/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)

    # p0 = p0 + 'lr' + str(lr) + '/'
    # if os.path.exists(p0) == False:
    #     os.mkdir(p0, 0o777)

    p0=p0+wavelet_name+'/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)
    p0=p0+'level'+str(n_level)+'/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)

  

    for r in range(repeats):
        outp = p0 + str(r) + '/'
        if os.path.exists(outp) == False:
            os.mkdir(outp, 0o777)

        reconstructor = A2_Reconstructor(nc, window, ts_cin)

        reconstructor = nn.DataParallel(reconstructor)
        reconstructor.to(device)
        ratio_calculator = A2_Ratio(window)
        ratio_calculator = nn.DataParallel(ratio_calculator)
        ratio_calculator.to(device)
        
        # optimizer = optim.Adam(reconstructor.parameters(), 1 * 1e-4,betas=(0.5,0.999))
        params = ([a for a in reconstructor.parameters()] + [b for b in ratio_calculator.parameters()])
        optimizer = optim.RMSprop(params, lr)

        vloss = float('inf')  # 
        losst = []  # 

        lossv = []

        ep = -1
        counter = 0
        total_running = epoches  # 
        epoch_train_times = []
        for epoch in range(epoches):  # 
            # scheduler.step()  #
            trainloader, val1loader, val2loader, testloader = form_dataloader(batchsize, datapath)

            del val2loader
            del testloader
            gc.collect()
            torch.cuda.synchronize()
            start = time.clock()
            losst = train(reconstructor, xfm, ifm, device, trainloader, ratio_calculator, epoch, losst, optimizer,
                          len_list)  #
            del trainloader
            gc.collect()
            torch.cuda.synchronize()
            end = time.clock()
            epoch_train_time = end - start
            epoch_train_times.append(epoch_train_time)
            # scheduler_g.step()       #
            # scheduler_d.step()
            lossv, vloss, ep, counter = val(reconstructor, xfm, ifm, window, ratio_calculator, device, val1loader,
                                            epoch, ep, lossv, vloss, counter, patience, threshold,batchsize,len_list, outp, optimizer)
            del val1loader
            gc.collect()
            if counter == patience:
                total_running = epoch + 1  # 
                break
            torch.cuda.empty_cache()
          
        time_data = np.asarray(epoch_train_times).T
        df_train_time = pd.DataFrame(columns=['epoch_train_time'], data=time_data)
        df_train_time.loc['mean', 'epoch_train_time'] = df_train_time['epoch_train_time'].mean()
        df_train_time.to_csv(outp + 'epoch_train_time.csv')

        trainloader, val1loader, val2loader, testloader = form_dataloader(batchsize, datapath)

        del trainloader
        del val1loader
        inference(device, val2loader, nc, ts_cin, freq_cout, xfm, ifm, window, ratio_nc, len_list,batchsize, outp, 'val2')

        inference(device, testloader, nc, ts_cin, freq_cout, xfm, ifm, window, ratio_nc, len_list, batchsize,outp, 'test')  # 该次inference时需要重新构造一个model(net)

        data_lt = np.asarray(losst).reshape(len(losst), 1)
        data_lv = np.asarray(lossv).reshape(len(lossv), 1)

        data_l = np.hstack((data_lt, data_lv))
        df_loss = pd.DataFrame(data=data_l, columns=['train_loss', 'val_loss'])
        df_loss.to_csv(outp + 'epoch_loss.csv')
        torch.cuda.empty_cache()
        gc.collect()






