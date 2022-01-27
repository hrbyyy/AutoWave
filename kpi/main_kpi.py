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
seed_everything() #可复现


#保存dataloader便于送入model训练，不必：返回tensor便于pytorch_wavelet求coef，取dataloader的data[0]即可。
def form_dataloader(batchsize,inpath):
    trainx_new=pickle.load(open(inpath+'trainx.pkl','rb'))
    val1x_new=pickle.load(open(inpath+'val1x.pkl','rb'))
    val2x_new = pickle.load(open(inpath + 'val2x.pkl', 'rb'))
    testx_new = pickle.load(open(inpath + 'testx.pkl', 'rb'))

#不是将x送入网络，而是将其wavelet coef送入LSTM,故此处不用swapaxes.
    train_x = torch.from_numpy(trainx_new).type(torch.FloatTensor)
    val1_x = torch.from_numpy(val1x_new).type(torch.FloatTensor)
    val2_x = torch.from_numpy(val2x_new).type(torch.FloatTensor)
    test_x = torch.from_numpy(testx_new).type(torch.FloatTensor)
    # print(train_x.shape)  (b,f=2,t)含label
    # exit()
    torch_trainset = Data.TensorDataset(train_x, train_x)
    # 把 dataset 放入 DataLoader
    trainloader = Data.DataLoader(
        dataset=torch_trainset,  # torch TensorDataset format
        batch_size=batchsize,  # mini batch size
        shuffle=True  # 要不要打乱数据 (打乱比较好)
    )
    val1data = Data.TensorDataset(val1_x, val1_x)
    val1loader = Data.DataLoader(dataset=val1data, shuffle=True, batch_size=batchsize)
    val2data = Data.TensorDataset(val2_x, val2_x)
    val2loader = Data.DataLoader(dataset=val2data, shuffle=False, batch_size=batchsize)
    testdata = Data.TensorDataset(test_x, test_x)
    testloader = Data.DataLoader(dataset=testdata, shuffle=False, batch_size=batchsize)

    return trainloader,val1loader,val2loader,testloader

class Option_pp:   #预处理和后处理block对应的参数
    def __init__(self,c_in,nc):
        self.c_in=c_in
        self.freq_cout=nc


if __name__ == "__main__":

    repeats = 5
    epoches = 100  # 由程序输入

    d_clip = 0.01

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练模型

    batchsize = 128


    patience = 20  # earlystopping指定轮数
    threshold = 2  # 每个batch val loss不再显著减小的值

    inpath = '/mnt/A/dataset/kpi/final_for_model/original/April/reconstruction/'
    window = 60
    inpath=inpath+'window'+str(window)+'/'  #利用kpi重构实验处理的数据。

    wavelet_name = 'haar'
    wavelet = pywt.Wavelet('haar')
    wavelet_len = wavelet.dec_len  # 2
    # print(len(morl))
    # n_level = pywt.dwt_max_level(window, wavelet.dec_len)  # 6
    n_level=2
    xfm = DWT1DForward(J=n_level,wave='haar').cuda()
    ifm=DWT1DInverse(wave='haar').cuda()



    len_list = []
    level_list = list(range(n_level + 1))
    cur_len = window
    total_len = 0  # 为后续可能需要准备
    for i in range(n_level):
        new_len = (cur_len + wavelet_len - 1) // 2
        len_list.append(new_len)
        total_len += new_len
        cur_len = new_len
    len_list=[cur_len]+len_list    #首位添加cur_len
    total_len+=cur_len   #还要加上最后一次approximate的长度！
    print(total_len)   #window_size总长为120时，coef总长度为130
    # len_list.append(cur_len)  # 最高level同时保留了approximate和detail
    # len_list.reverse()  # 从high到low，原地保存  #torch版pywt结果无需反向

    # input_feature=len(scale_list)
    # hidden_units = 64
    # n_layer = 1
    # dropout = 0  #0.5  注意：RNN 一层网络时，不用dropout
    ts_cin=1
    nc=4
    ratio_nc=4
    # #各opt参数为构建freq_tuner（inception架构）对应的参数
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
    # 单独保存每个ts_entity的结果
    p0 = p0 + 'RMSprop/'
    if os.path.exists(p0) == False:
        os.mkdir(p0, 0o777)
    # for lr in lr_list:
    rootp=p0+'learning_rate'+str(lr)+'/'
    if os.path.exists(rootp)==False:
        os.mkdir(rootp,0o777)
    selected_ts = [16, 26, 27]  #,15,17,19,21,12,  ,25,28,29
    for ts_id in selected_ts:

        inpath_ts=inpath+str(ts_id)+'/'


        # trainx,val1x,val2x,testx=form_dataloader(batchsize, inpath_ts)

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
            #可改为adam,并调整lr
            params = ([a for a in reconstructor.parameters()] + [b for b in ratio_calculator.parameters()])

            optimizer = optim.RMSprop(params, lr= lr)

            # scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, 20, gamma=0.1, last_epoch=-1)
            # scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, 20, gamma=0.1, last_epoch=-1)

            vloss = float('inf')  # vloss要设为无穷大
            losst = []  # 绘制loss曲线需要的training loss 和val loss

            lossv = []

            ep = -1
            counter = 0
            total_running = epoches  # 记录运行的总epoch数目，初始化为总epoches数目，early stopping时更新
            epoch_train_times = []
            for epoch in range(epoches):  # early stopping时，没有跑满epoches数目，需要记录停止时的epoch总数，便于绘制loss图
                # scheduler.step()  #改变学习率，后续尝试
                trainloader, val1loader, val2loader, testloader = form_dataloader(batchsize,inpath_ts)
                del val2loader
                del testloader
                gc.collect()
                torch.cuda.synchronize()
                start = time.clock()
                losst = train(reconstructor, xfm, ifm, device, trainloader, ratio_calculator,epoch, losst, optimizer, len_list)  # 每个epoch对该次repeat的同一个model进行训练
                del trainloader
                gc.collect()
                torch.cuda.synchronize()
                end = time.clock()
                epoch_train_time = end - start
                epoch_train_times.append(epoch_train_time)
                # scheduler_g.step()       #每个epoch对该次repeat的同一个model进行训练
                # scheduler_d.step()
                lossv,  vloss, ep, counter = val(reconstructor, xfm, ifm, window, device, ratio_calculator,val1loader, epoch, ep, lossv, vloss, counter, patience, threshold, batchsize,len_list,outp, optimizer)
                del val1loader
                gc.collect()
                if counter == patience:
                    total_running = epoch + 1  # epoch从0开始，记录early stopping时经历的epoch总数
                    break
                torch.cuda.empty_cache()
            # inference(device,val2_x,val2_y,num_inputs,num_outputs,int(i+1),batchfirst,dropout,outp,'val2')
            # inference(device,test_x,test_y,num_inputs,num_outputs,int(i+1),batchfirst,dropout,outp,'test')     #该次inference时需要重新构造一个model(net)

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
            inference(device, testloader,opt_ae,ts_cin,nc, xfm,ifm,window,ratio_nc,batchsize,len_list, outp, 'test')  # 该次inference时需要重新构造一个model(net)
            del testloader
            gc.collect()
            data_lt = np.asarray(losst).reshape(len(losst), 1)
            data_lv = np.asarray(lossv).reshape(len(lossv), 1)

            data_l = np.hstack((data_lt, data_lv))
            df_loss = pd.DataFrame(data=data_l, columns=['train_loss', 'val_loss'])
            df_loss.to_csv(outp + 'epoch_loss.csv')
            torch.cuda.empty_cache()
            gc.collect()




