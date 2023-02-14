
# --coding:utf-8--

import sys
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import multiprocessing
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
import numpy as np

import pandas as pd
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR
import pickle
import matplotlib.pyplot as plt
from functools import reduce
from torch.nn import BatchNorm1d
import time


import score_GAN_uts



def combine_para(lists):
    def my_fun(list1, list2):
        return [str(i)+','+str(j) for i in list1 for j in list2] 

    return reduce(my_fun, lists)


wavelet_name='haar'
n_level=4
repeats=5

p0='/mnt/A/PycharmProject/wavelet_rec/ecg/double_inputs/conv/haar/level4/adam10.0005/epoch100/'


def compute_oldmetric(true_labels,pred_labels,outp,indicator): 

   
    pred_labels, true_labels =pred_labels.astype(int),true_labels.astype(int)


    data = np.concatenate([pred_labels, true_labels], axis=1)
    df = pd.DataFrame(data=data, columns=['pred', 'real'])
    df.to_csv(outp +indicator+'pred_real_labels.csv')
    TP = len(df.loc[(df['pred'] == 1) & (df['real'] == 1)])
    FP = len(df.loc[(df['pred'] == 1) & (df['real'] == 0)])
    TN = len(df.loc[(df['pred'] == 0) & (df['real'] == 0)])
    FN = len(df.loc[(df['pred'] == 0) & (df['real'] == 1)])
    if TP == 0:
        P, R, F1 ,tpr= 0, 0, 0,0
    else:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R)  # 
        tpr = TP / (TP + FN)
    if FP==0:
        fpr=0
    else:
        fpr = FP / (FP + TN)
    if fpr == 0:
        rpr = 'inf'
    else:
        rpr = tpr / fpr
   
    df_out = pd.DataFrame(
        [{'P': P, "R": R, 'F1': F1, 'FP rate': fpr, 'relative positive ratio': rpr}])
    df_out.to_csv(outp +indicator+ 'pred_metrics.csv')
    return

def seqlevel_threshold(seqas_array, val_label, outp,score_indicator): 
  

    candidates=sorted(seqas_array, reverse=True)
  
    index=-1
    fmax=-1
   
    score_array=seqas_array.astype(int)
    
    val_label=val_label.astype(int)



    for j in range(len(candidates)):
        pred_label=np.where(score_array>=candidates[j],1,0)
      
        pred_label=pred_label.astype(int)
        data=np.hstack((pred_label, val_label))
        df=pd.DataFrame(data=data,columns=['pred','real'])
        TP=len(df.loc[(df['pred']==1)&(df['real']==1)])
        FP=len(df.loc[(df['pred']==1)&(df['real']==0)])
        TN=len(df.loc[(df['pred']==0)&(df['real']==0)])
        FN=len(df.loc[(df['pred']==0)&(df['real']==1)])
        if TP == 0:
            P, R, F1, tpr = 0, 0, 0, 0
        else:
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = 2 * P * R / (P + R)  # 
            tpr = TP / (TP + FN)
        if FP == 0:
            fpr = 0
        else:
            fpr = FP / (FP + TN)
        if fpr == 0:
            rpr = 'inf'
        else:
            rpr = tpr / fpr
        if F1>fmax:
            fmax=F1
            index=j
    df_out=pd.DataFrame(columns=['index','threshold','total_length'])
    df_out=df_out.append({'index':index,'threshold':candidates[index],'total_length':len(seqas_array)}, ignore_index=True)
    df_out.to_csv(outp+score_indicator+'_pointlevel_threshold.csv')
    return candidates[index]



def post_cal(rootp,repeats):

    ppath=rootp
   

    for r in range(repeats):
        outp = ppath + str(r) + '/'

        val2dict = pickle.load(open(outp + 'val2' + '_errslabels.pkl', 'rb'))
        val2_errvs=val2dict['error_vectors'].squeeze(axis=1)
        val2_labels = val2dict['true_labels'] 

        testdict = pickle.load(open(outp + 'test' + '_errslabels.pkl', 'rb'))
        test_errvs=testdict['error_vectors'].squeeze(axis=1)
        test_labels = testdict['true_labels']

        val2_predseqs = val2dict['rec_seqs']  
        val2_realseqs = val2dict['real_seqs']
      
        test_predseqs = testdict['rec_seqs']
        test_realseqs = testdict['real_seqs']
    

       
        raw_val2as = np.abs(val2_realseqs - val2_predseqs).squeeze(axis=1)  
        raw_testas = np.abs(test_realseqs - test_predseqs).squeeze(axis=1)
        val2_seqas=raw_val2as.sum(axis=1,keepdims=True)  
        test_seqas=raw_testas.sum(axis=1,keepdims=True)


        threshold = seqlevel_threshold(val2_seqas, val2_labels, outp, 'MAE')

        predict_seqlabel = np.where(test_seqas>threshold,1,0) 

        compute_oldmetric(test_labels, predict_seqlabel, outp, 'seqMAE')


    return

post_cal(p0,repeats)
