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


import score_uts




def combine_para(lists):
    def my_fun(list1, list2):
        return [str(i) + ',' + str(j) for i in list1 for j in
                list2]  

    return reduce(my_fun, lists)


window = 60  


repeats = 5


rate_threshold = 0.1

p0="/mnt/A/PycharmProject/wavelet_rec/kpi/double_inputs/conv/window60/haar/level2/RMSprop/learning_rate0.0005/"



def compute_oldmetric(true_labels, pred_labels, outp, indicator):  

    pred_labels = np.reshape(pred_labels, newshape=(len(pred_labels), 1))
    true_labels = np.reshape(true_labels, newshape=(len(true_labels), 1))

    pred_labels, true_labels = pred_labels.astype(int), true_labels.astype(int)

    data = np.concatenate([pred_labels, true_labels], axis=1)
    df = pd.DataFrame(data=data, columns=['pred', 'real'])
    df.to_csv(outp + indicator + 'pred_real_labels.csv')
    TP = len(df.loc[(df['pred'] == 1) & (df['real'] == 1)])
    FP = len(df.loc[(df['pred'] == 1) & (df['real'] == 0)])
    TN = len(df.loc[(df['pred'] == 0) & (df['real'] == 0)])
    FN = len(df.loc[(df['pred'] == 0) & (df['real'] == 1)])
    if TP == 0:
        P, R, F1, tpr = 0, 0, 0, 0
    else:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R) 
        tpr = TP / (TP + FN)
    if FP == 0:
        fpr = 0
    else:
        fpr = FP / (FP + TN)
    if fpr == 0:
        rpr = 'inf'
    else:
        rpr = tpr / fpr
   
    df_out = pd.DataFrame(
        [{'P': P, "R": R, 'F1': F1, 'FP rate': fpr, 'relative positive ratio': rpr}])
    df_out.to_csv(outp + indicator + 'pred_metrics.csv')
    return


def seqlevel_threshold(seqas_array, val_label, outp,
                       score_indicator):
  

    candidates = sorted(seqas_array, reverse=True)
  
    index = -1
    fmax = -1
    
    score_array = seqas_array.astype(int)
   
    val_label = val_label.astype(int)

  

    for j in range(len(candidates)):
        pred_label = np.where(score_array >= candidates[j], 1, 0)
       
        pred_label = pred_label.astype(int)
        data = np.hstack((pred_label, val_label))
        df = pd.DataFrame(data=data, columns=['pred', 'real'])
        TP = len(df.loc[(df['pred'] == 1) & (df['real'] == 1)])
        FP = len(df.loc[(df['pred'] == 1) & (df['real'] == 0)])
        TN = len(df.loc[(df['pred'] == 0) & (df['real'] == 0)])
        FN = len(df.loc[(df['pred'] == 0) & (df['real'] == 1)])
        if TP == 0:
            P, R, F1, tpr = 0, 0, 0, 0
        else:
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = 2 * P * R / (P + R) 
            tpr = TP / (TP + FN)
        if FP == 0:
            fpr = 0
        else:
            fpr = FP / (FP + TN)
        if fpr == 0:
            rpr = 'inf'
        else:
            rpr = tpr / fpr
        if F1 > fmax:
            fmax = F1
            index = j
    df_out = pd.DataFrame(columns=['index', 'threshold', 'total_length'])
    df_out = df_out.append({'index': index, 'threshold': candidates[index], 'total_length': len(seqas_array)},
                           ignore_index=True)
    df_out.to_csv(outp + score_indicator + '_pointlevel_threshold.csv')
    return candidates[index]


l = window // 2
selected_ts = [16, 26, 27] 


def post_cal(ts_id, rootp, repeats):

    ppath = rootp + str(ts_id) + '/'
    # ppath = rootp + 'encoder' + str(encoder_layer) + 'unit' + str(encoder_unit) + 'decoder' + str(decoder_layer) + 'unit' + str(decoder_unit) + "/"
    print(ppath)

    for r in range(repeats):
        outp = ppath + str(r) + '/'
        # print(outp)

        val2dict = pickle.load(open(outp + 'val2' + '_errslabels.pkl', 'rb'))
        val2_errvs = val2dict['error_vectors'].squeeze(axis=1)
        val2_labels = val2dict['true_labels']  

        testdict = pickle.load(open(outp + 'test' + '_errslabels.pkl', 'rb'))
        test_errvs = testdict['error_vectors'].squeeze(axis=1)
        test_labels = testdict['true_labels']

        val2_predseqs = val2dict['rec_seqs']  # (b,t,f) f=1
        val2_realseqs = val2dict['real_seqs']
        # val2_probs = val2dict['predict_probs']
        test_predseqs = testdict['rec_seqs']
        test_realseqs = testdict['real_seqs']
        # test_probs = testdict['predict_probs']

        raw_val2as = np.abs(val2_realseqs - val2_predseqs).squeeze(axis=1)  # (b,f,t) f=1
        raw_testas = np.abs(test_realseqs - test_predseqs).squeeze(axis=1)

        val2_seqas = raw_val2as.sum(axis=1).reshape(len(raw_val2as), 1)  
        test_seqas = raw_testas.sum(axis=1).reshape(len(raw_testas), 1)
      
        real_val2labels = score_uts.trans_reallabel(val2_labels, 1).reshape(len(val2_labels), 1)
        real_testlabels = score_uts.trans_reallabel(test_labels, 1).reshape(len(test_labels), 1)

        threshold = seqlevel_threshold(val2_seqas, real_val2labels, outp, 'MAE')
      
        predict_seqlabel = np.where(test_seqas > threshold, 1, 0) 
        # real_pointlabel = score_uts.true_pointlabel(test_labels)
        compute_oldmetric(real_testlabels, predict_seqlabel, outp, 'seqMAE')
      
    return


pool = multiprocessing.Pool(processes=3)
pool.apply_async(post_cal, (16, p0, repeats))
pool.apply_async(post_cal, (26, p0, repeats))
pool.apply_async(post_cal, (27, p0, repeats))

pool.close()
pool.join()
