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
from model import KPI_Reconstructor,KPI_Ratio
from utils import gain_coef_v2, normalization,process_coef,shape_tune


def inference(device, testloader, opt_ae,ts_cin,nc,xfm,ifm,window,ratio_nc, batchsize,len_list, outp, filename):
    n_samples=len(testloader.dataset)

    pred_loss=0
    real_seq=torch.empty(size=(n_samples,1,window),device=device)
    rec_seq=torch.empty(size=(n_samples,1,window),device=device)
    err_array=torch.empty(size=(n_samples,1,window),device=device)
    label_array=torch.empty(size=(n_samples,window),device=device)
    ratio_array = torch.empty(size=(n_samples, 1), device=device)
    # total_ratio = 0
    with torch.no_grad():
        netinf = KPI_Reconstructor(opt_ae,ts_cin,nc)
        # net = TemporalConvNet(num_inputs, num_channels)
        # net.to(device)
        netinf=nn.DataParallel(netinf)
        netinf.to(device)
        netinf.eval()
        ratio_calculator=KPI_Ratio(window,ratio_nc)
        ratio_calculator=nn.DataParallel(ratio_calculator)
        ratio_calculator.to(device)
        ratio_calculator.eval()

        criterion = torch.nn.MSELoss(reduction='none')
        criterion2 = torch.nn.MSELoss(reduction='none') 
       
        state = torch.load(outp + 'model.pth')


        netinf.load_state_dict(state['net'])
        ratio_calculator.load_state_dict(state['net_ratio'])
       
        torch.cuda.synchronize()
        t0 = time.clock()
        for batch_idx, data in enumerate(testloader):

            start_posi=batch_idx*batchsize
            end_posi= min(n_samples, start_posi+batchsize)

           
            inputs, targets = data[0][:, :1,  :], data[1][:, :1,  :]
            label=data[0][:,1,:].to(device)

            inputs, targets = inputs.to(device), targets.to(device)
            torch.cuda.synchronize()
            tstart = time.clock()
            coef = gain_coef_v2(inputs, xfm)
            normalized_coef = normalization(coef)
            trec_inputs = netinf(inputs) 
            new_coef = process_coef(normalized_coef, len_list)  # (b,1,t)

            rec_coef = shape_tune(new_coef, len_list)
            frec_inputs = ifm(rec_coef)
            ratio = ratio_calculator(trec_inputs, frec_inputs)
            ratio = torch.unsqueeze(ratio, dim=1)
          
            loss = (1-ratio)*criterion(trec_inputs, targets) + ratio * criterion(frec_inputs, targets)  
            loss = torch.sum(loss)
            err = criterion2(trec_inputs, targets)
            torch.cuda.synchronize()
            tend = time.clock()
            per_time=(tend-tstart)/len(inputs)
            label_array[start_posi:end_posi, :] = label
            err_array[start_posi:end_posi,:,:]=err
            real_seq[start_posi:end_posi,:,:]=inputs
            rec_seq[start_posi:end_posi,:,:]=trec_inputs
            ratio_array[start_posi:end_posi, :] = ratio
           
            pred_loss=pred_loss+loss

       
        torch.cuda.synchronize()
        t1 = time.clock()
        cpu_time = t1 - t0
        pred_error=err_array.detach().cpu().numpy()
        real_label=label_array.detach().cpu().numpy()
        real_seq=real_seq.detach().cpu().numpy()
        rec_seq=rec_seq.detach().cpu().numpy()
        ratio_array = ratio_array.detach().cpu().numpy()
       

        test_stats = {
            "error_vectors": pred_error,
            'true_labels':real_label,
            'real_seqs':real_seq,
            'rec_seqs':rec_seq,
            'ratio_array': ratio_array
          
        }
        pickle.dump(test_stats, open(outp + filename + '_errslabels.pkl', 'wb'))
        df = pd.DataFrame(columns=['prediction loss', 'cpu_time', 'per inference time'])
        df = df.append({'prediction loss': pred_loss.item(), 'cpu_time': cpu_time, 'per inference time': per_time},
                       ignore_index=True)
        df.to_csv(outp + filename + '_losstime.csv')
        torch.cuda.empty_cache()
    return
