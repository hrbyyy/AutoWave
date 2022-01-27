# --coding:utf-8--

import sys
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import os

extra_path = '/home/ices/PycharmProject/wavelet/uts_wavelet/kpi_inception/'
# extra_path2 = '/home/ices/PycharmProject/shape_sequence_kpi/uts/conditional_conv/model_combine/uts_WGAN_4score/frequency_condition/'
sys.path.append(extra_path)
# sys.path.append(extra_path2)
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

# import score_LSTM
# from synthetic_data.big_sin import score_LSTM
# from synthetic_data import preprocess
import score_GAN_uts
import multiprocessing


# import new_score_func


def combine_para(lists):
    def my_fun(list1, list2):
        return [str(i) + ',' + str(j) for i in list1 for j in
                list2]  # 

    return reduce(my_fun, lists)


pw = 4  # 



repeats = 5
per_val2 = 97
per_test = 718

# 
k = 2  #
rate_threshold = 0.1
# p0 = '/mnt/A/PycharmProject/wavelet_rec/A2/conv_inceptiony/haar/'
# p0='/mnt/A/PycharmProject/wavelet_rec/A2/only_temporal2/window4/'
p0='/mnt/A/PycharmProject/wavelet_rec/A2/double_inputs/8-4-1/haar/level2/'


# 

def compute_oldmetric(true_labels, pred_labels, indicator):  # 

    pred_labels = np.reshape(pred_labels, newshape=(len(pred_labels), 1))
    true_labels = np.reshape(true_labels, newshape=(len(true_labels), 1))

    pred_labels, true_labels = pred_labels.astype(int), true_labels.astype(int)

    data = np.concatenate([pred_labels, true_labels], axis=1)
    df = pd.DataFrame(data=data, columns=['pred', 'real'])
    # df.to_csv(outp +indicator+'pred_real_labels.csv')
    TP = len(df.loc[(df['pred'] == 1) & (df['real'] == 1)])
    FP = len(df.loc[(df['pred'] == 1) & (df['real'] == 0)])
    TN = len(df.loc[(df['pred'] == 0) & (df['real'] == 0)])
    FN = len(df.loc[(df['pred'] == 0) & (df['real'] == 1)])
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
  
    return P, R, F1, fpr, tpr


def pointlevel_threshold(average_score_list, val_label,
                         score_indicator):  # 
    # 

    point_score = np.asarray(average_score_list).reshape(len(average_score_list), 1)
    # 
    label_list = val_label[:, 0].tolist()
    last_list = val_label[-1, 1:].tolist()
    label_list.extend(last_list)
    label_array = np.asarray(label_list).reshape(len(label_list), 1).astype(int)

    candidates = sorted(point_score, reverse=True)
    # for i in range(len(val_score_list) - 1):
    #     candidates.append((val_score_list[i] + val_score_list[i + 1]) / 2)
    index = -1
    fmax = -1
    # score_array=np.asarray(sequence_score).reshape((len(sequence_score), 1))
    score_array = point_score
    # val_label=np.asarray(val_label).reshape((len(val_label), 1))
    val_label = label_array.astype(int)

    #

    for j in range(len(candidates)):
        pred_label = np.where(score_array >= candidates[j], 1, 0)
        # 
        pred_label = pred_label.astype(int)
        data = np.hstack((pred_label, val_label))
        df = pd.DataFrame(data=data, columns=['pred', 'real'])
        TP = len(df.loc[(df['pred'] == 1) & (df['real'] == 1)])
        FP = len(df.loc[(df['pred'] == 1) & (df['real'] == 0)])
        TN = len(df.loc[(df['pred'] == 0) & (df['real'] == 0)])
        FN = len(df.loc[(df['pred'] == 0) & (df['real'] == 1)])
        if TP == 0:
            P, R, F1 = 0, 0, 0
        else:
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = 2 * P * R / (P + R)  # 
        if F1 > fmax:
            fmax = F1
            index = j
    # df_out=pd.DataFrame(columns=['index','threshold','total_length'])
    # df_out=df_out.append({'index':index,'threshold':candidates[index],'total_length':len(val_score_array)}, ignore_index=True)
    # df_out.to_csv(outp+score_indicator+'_pointlevel_threshold.csv')
    return candidates[index]

def seqlevel_threshold(seqas_array, val_label, outp,score_indicator): #
  #

    candidates=sorted(seqas_array, reverse=True)
    # for i in range(len(val_score_list) - 1):
    #     candidates.append((val_score_list[i] + val_score_list[i + 1]) / 2)
    index=-1
    fmax=-1
    # score_array=np.asarray(sequence_score).reshape((len(sequence_score), 1))
    score_array=seqas_array.astype(int)
    # val_label=np.asarray(val_label).reshape((len(val_label), 1))
    val_label=val_label.astype(int)

    #

    for j in range(len(candidates)):
        pred_label=np.where(score_array>=candidates[j],1,0)
        #
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



# l = pw // 2

def post_cal(path):  #,level
    # level_p=path+'level'+str(level)+'/'
    for r in range(repeats):
        outp = path + str(r) + '/'

        val2dict = pickle.load(open(outp + 'val2' + '_errslabels.pkl', 'rb'))
        val2_errvs=val2dict['error_vectors']
        val2_labels = val2dict['true_labels']  # 
        testdict = pickle.load(open(outp + 'test' + '_errslabels.pkl', 'rb'))
        test_errvs=testdict['error_vectors']
        test_labels = testdict['true_labels']

        val2_predseqs = val2dict['rec_seqs']  # (b,t,f) f=1
        val2_realseqs = val2dict['real_seqs']
        # val2_probs = val2dict['predict_probs']
        test_predseqs = testdict['rec_seqs']
        test_realseqs = testdict['real_seqs']
        # test_probs = testdict['predict_probs']
        # 
        num_ts = len(val2_predseqs) // per_val2

        dfseqMAE_round = pd.DataFrame(columns=['P', 'R', 'F1', 'fpr', 'tpr'])
        for i in range(num_ts):
            val2_start = int(i * per_val2)
            val2_end = int((i + 1) * per_val2)
            test_start = int(i * per_test)
            test_end = int((i + 1) * per_test)
            per_val2predseq = val2_predseqs[val2_start:val2_end, :, :]
            per_val2realseq = val2_realseqs[val2_start:val2_end, :, :]
            # per_val2probs = val2_probs[val2_start:val2_end, :]
            per_val2labels = val2_labels[val2_start:val2_end, :]
            per_val2errvs = val2_errvs[val2_start:val2_end, :, :].squeeze(axis=1)
            per_testpredseq = test_predseqs[test_start:test_end, :, :]
            per_testrealseq = test_realseqs[test_start:test_end, :, :]
            # per_testprobs = test_probs[test_start:test_end, :]
            per_testlabels = test_labels[test_start:test_end, :]
            per_testerrvs=test_errvs[test_start:test_end, :,:].squeeze(axis=1)

    # 
            raw_val2as = np.abs(per_val2realseq - per_val2predseq).squeeze(axis=1)  # 
            raw_testas = np.abs(per_testrealseq - per_testpredseq).squeeze(axis=1)

    # 

            val2_seqas = raw_val2as.sum(axis=1).reshape(len(raw_val2as), 1)  # 
            test_seqas = raw_testas.sum(axis=1).reshape(len(raw_testas), 1)

        # 
            real_val2labels = score_GAN_uts.trans_reallabel(per_val2labels, 1).reshape(len(per_val2labels), 1)
            real_testlabels = score_GAN_uts.trans_reallabel(per_testlabels, 1).reshape(len(per_testlabels), 1)

            seqMAEthreshold = seqlevel_threshold(val2_seqas, real_val2labels, outp, 'seqMAE')

            predict_seqlabel = np.where(test_seqas >=seqMAEthreshold, 1, 0)  # 
            # real_pointlabel = score_GAN_uts.true_pointlabel(test_labels)
            p_sMAE, r_sMAE, f1_sMAE, fpr_sMAE, tpr_sMAE=compute_oldmetric(real_testlabels, predict_seqlabel, 'seqMAE')

            dfseqMAE_round = dfseqMAE_round.append({'P': p_sMAE, 'R': r_sMAE, 'F1': f1_sMAE, 'fpr': fpr_sMAE, 'tpr': tpr_sMAE}, ignore_index=True)

        dfseqMAE_round.loc['mean'] = dfseqMAE_round.apply(lambda x: x.mean())

        dfseqMAE_round.to_csv(outp + 'seqMAE_metrics.csv')
    return

post_cal(p0)
