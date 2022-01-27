
# --coding:utf-8--
#按hw为分组，不同fsize时固定层数，并调整dilationrate
# 采用的是将ts变化成回归形式的数据集，并按时间先后划分train-val-test
import sys
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import multiprocessing
import os
extra_path='/home/ices/PycharmProject/shape_sequence_kpi/uts/conditional_conv/model_combine/uts_WGAN_4score/'
extra_path2='/home/ices/PycharmProject/shape_sequence_kpi/uts/conditional_conv/model_combine/uts_WGAN_4score/frequency_condition/'
sys.path.append(extra_path)
sys.path.append(extra_path2)
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
#from synthetic_data import preprocess
import score_GAN_uts
# import new_score_func


def combine_para(lists):
    def my_fun(list1, list2):
        return [str(i)+','+str(j) for i in list1 for j in list2]  #采用str形式，后续利用split得到每个元素值，如果此处包装成[i，j],则会变成[[i,j],k]形式

    return reduce(my_fun, lists)

# pw=10    #预测步长,30,40
# hw=10        #288
# num_sim=1  #k_sim中k的数目
wavelet_name='haar'
n_level=4
repeats=5
# k=pw//2 #point_adjust对应的delay_tolerance
# rate_threshold=0.1
# p0 = '/mnt/A/PycharmProject/wavelet_rec/ecg/auto_statis_saveratio/haar/level4/adam10.0001/' #
p0='/mnt/A/PycharmProject/wavelet_rec/ecg/double_inputs/conv/haar/level4/adam10.0005/epoch100/'
# p0="/mnt/A/PycharmProject/wavelet_rec/ecg/auto_ratio2/"
# p0 = p0 + wavelet_name + '/'
# #
# p0 = p0 + 'level' + str(n_level) + '/adam1e-5/'
#p0=p0+'adam_beta/'
# 单独保存每个ts_entity的结果

def compute_oldmetric(true_labels,pred_labels,outp,indicator): #传入real_label和predict_label

    # pred_labels = np.reshape(pred_labels,newshape=(len(pred_labels), 1))
    # true_labels = np.reshape(true_labels,newshape=(len(true_labels), 1))

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
        F1 = 2 * P * R / (P + R)  # 尝试改用Fb(bata,定bata值)
        tpr = TP / (TP + FN)
    if FP==0:
        fpr=0
    else:
        fpr = FP / (FP + TN)
    if fpr == 0:
        rpr = 'inf'
    else:
        rpr = tpr / fpr
    # 用list of dict 初始化df,每个dict是一个记录
    df_out = pd.DataFrame(
        [{'P': P, "R": R, 'F1': F1, 'FP rate': fpr, 'relative positive ratio': rpr}])
    df_out.to_csv(outp +indicator+ 'pred_metrics.csv')
    return

def seqlevel_threshold(seqas_array, val_label, outp,score_indicator): #定义函数求threshold      传入VN2和VA的anomaly score(合为一个列表),以及val_y(list,真实0，1label,而非regression值)
  #原求point_score方法错误，应该用多个时刻的score均值，而非和label取法一致，将首列和末行结果拼接，只取一个时刻的值

    candidates=sorted(seqas_array, reverse=True)
    # for i in range(len(val_score_list) - 1):
    #     candidates.append((val_score_list[i] + val_score_list[i + 1]) / 2)
    index=-1
    fmax=-1
    # score_array=np.asarray(sequence_score).reshape((len(sequence_score), 1))
    score_array=seqas_array.astype(int)
    # val_label=np.asarray(val_label).reshape((len(val_label), 1))
    val_label=val_label.astype(int)

    #补充val_y变成sequence_label

    for j in range(len(candidates)):
        pred_label=np.where(score_array>=candidates[j],1,0)
        #补充pred_label变成sequence label
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
            F1 = 2 * P * R / (P + R)  # 尝试改用Fb(bata,定bata值)
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

def post_cal(rootp,repeats):

    ppath=rootp
    # ppath = rootp + 'encoder' + str(encoder_layer) + 'unit' + str(encoder_unit) + 'decoder' + str(decoder_layer) + 'unit' + str(decoder_unit) + "/"


    for r in range(repeats):
        outp = ppath + str(r) + '/'

        val2dict = pickle.load(open(outp + 'val2' + '_errslabels.pkl', 'rb'))
        val2_errvs=val2dict['error_vectors'].squeeze(axis=1)
        val2_labels = val2dict['true_labels']  # 从dataloader拼接而得，已经越过初始hw点

        testdict = pickle.load(open(outp + 'test' + '_errslabels.pkl', 'rb'))
        test_errvs=testdict['error_vectors'].squeeze(axis=1)
        test_labels = testdict['true_labels']

        val2_predseqs = val2dict['rec_seqs']  # (b,t,f) f=1
        val2_realseqs = val2dict['real_seqs']
        # val2_probs = val2dict['predict_probs']
        test_predseqs = testdict['rec_seqs']
        test_realseqs = testdict['real_seqs']
        # test_probs = testdict['predict_probs']

        # 原metric
        raw_val2as = np.abs(val2_realseqs - val2_predseqs).squeeze(axis=1)  # (b,f,t) f=1,需降维。
        raw_testas = np.abs(test_realseqs - test_predseqs).squeeze(axis=1)
        val2_seqas=raw_val2as.sum(axis=1,keepdims=True)  #保持二维形式，便于后续求metric
        test_seqas=raw_testas.sum(axis=1,keepdims=True)


        threshold = seqlevel_threshold(val2_seqas, val2_labels, outp, 'MAE')

        predict_seqlabel = np.where(test_seqas>threshold,1,0) #利用np.where,无需特意编写函数

        compute_oldmetric(test_labels, predict_seqlabel, outp, 'seqMAE')


    return

post_cal(p0,repeats)
