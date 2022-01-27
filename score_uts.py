import os
import numpy as np  
import pandas as pd
from sklearn.metrics import roc_curve,f1_score,precision_score,recall_score
import pickle
# import matplotlib.pyplot as plt
import torch
# from sklearn.metrics import auc

def score2label(as_list,threshold):
    pred_labels = []
    for i in range(len(as_list)):
        if as_list[i] > threshold:
            pred_labels.append(1)
        else:

            pred_labels.append(0)
    pred_labels=np.asarray(pred_labels)
    # pred_labels=pred_labels.reshape(len(pred_labels),1)
    return pred_labels


def point_adjust(predict_label, real_label, delay_tolerance):
    total = real_label.shape[0]
  
    real_label = real_label.astype(int)
    predict_label = predict_label.astype(int)

    real_string = ''.join(map(str, real_label))
    i = real_string.find('1')

    if i > -1:
        while i < total:
            while i < total and real_label[i] == 0:  # 
                i += 1  # 
            if i < total:
                j = i + 1
                while j < total and real_label[j] == 1:  # 
                    j += 1  # 
              
                if j < total:
                    slice_end = j
                else:
                    slice_end = -1
                current_predictstring = ''.join(map(str, predict_label[i:slice_end]))
                current_firstdetection = current_predictstring.find('1')

               
                if predict_label[
                   i:slice_end].sum() > 0 and delay_tolerance >= current_firstdetection >= 0:  # 
                    predict_label[i:slice_end] = 1
                else:
                    predict_label[i:slice_end] = 0
                i = j  # 
            else:
                break
    return predict_label


#
def average_score_vfast(score_array,pw):
    score=score_array
    count=np.ones_like(score_array)
    row,column=score_array.shape
    for i in range(pw-1):
        shift=i+1
        arr_shift=np.zeros_like(score_array)
        count_shift=np.zeros_like(score_array)
        arr_shift[shift:,:column-shift]=score_array[:row-shift,shift:]
        count_shift[shift:,:column-shift]=1
        score+=arr_shift
        count+=count_shift
    score=score/count
    #
    final_score=score[:,0].tolist()+score[-1,1:].tolist()
    return final_score

#
def average_score_fast(score_array,pw):
    b, s = score_array.shape
    timecoverage = b + s - 1

    score=np.zeros(shape=(timecoverage,))
    for t in range(timecoverage-(pw-1)):
        score_t=0
        if t<pw-1:
            num=t+1
        else:
            num=pw
        for i in range(num):
            score_t+=score_array[t-i,i]
        score_t/=num
        score[t]=score_t
    for tt in range(timecoverage-(pw-1),timecoverage):

        score_tt=0
        num_tt=timecoverage-tt#
        for j in range(num_tt):

            score_tt+=score_array[-1-j,j+1]
        score_tt/=num_tt
        score[tt]=score_tt
    return list(score)   #
# 
def average_score(score_array):
    b, s = score_array.shape
    timecoverage = b + s - 1
    scores = []
    for t in range(timecoverage):
        score = 0
        count = 0
        for idx in range(score_array.shape[0]):
            for posi in range(score_array.shape[1]):
                if idx + posi == t:
                    score += score_array[idx, posi]
                    count += 1
        score /= count
        scores.append(score)
    return scores
#
def point_AD(as_list, threshold, true_labels, outp, score_indicator):

    pred_labels=[]
    for i in range(len(as_list)):
        if as_list[i]>threshold:
            pred_labels.append(1)
        else:

            pred_labels.append(0)

    true_list=true_labels[:,0].tolist()
    last_list=true_labels[-1,1:].tolist()
    true_list.extend(last_list)
    #
    pred_labels=np.asarray(pred_labels).reshape(len(pred_labels),1)
    # true_labels=true_labels.reshape(len(true_labels),1)
    # true_labels=trans_reallabel(true_labels,1)
    true_labels=np.asarray(true_list).reshape(len(true_list),1)
    pred_labels,true_labels=pred_labels.astype(int),true_labels.astype(int)
    data=np.concatenate([pred_labels,true_labels],axis=1)
    df=pd.DataFrame(data=data,columns=['pred','real'])
    df.to_csv(outp + score_indicator+'_pred_real_alarms.csv')
    TP = len(df.loc[(df['pred'] == 1) & (df['real'] == 1)])
    FP = len(df.loc[(df['pred'] == 1) & (df['real'] == 0)])
    TN = len(df.loc[(df['pred'] == 0) & (df['real'] == 0)])
    FN = len(df.loc[(df['pred'] == 0) & (df['real'] == 1)])
    if TP==0:
        P,R,F1=0,0,0
    else:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P*R / (P + R)  # 
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    if fpr==0:
        rpr='inf'
    else:
        rpr=tpr/fpr
    #
    df_out=pd.DataFrame([{'P':P,"R":R,'F1':F1,'FP rate':fpr,'relative positive ratio':rpr,'threshold':threshold}])
    df_out.to_csv(outp+score_indicator+'_pointlevel_pred_metrics.csv')
    return



def sequencelevel_threshold(valscore_array,val_label,outp,score_indicator):
    val_label = val_label.astype(int)
    # print(type(valscore_array))

    candidates = sorted(valscore_array, reverse=True)  

    index = -1
    fmax = -1
    # score_array=np.asarray(sequence_score).reshape((len(sequence_score), 1))
    score_array = valscore_array
    # val_label=np.asarray(val_label).reshape((len(val_label), 1))


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
    df_out = df_out.append({'index': index, 'threshold': candidates[index], 'total_length': len(score_array)},
                           ignore_index=True)
    df_out.to_csv(outp + score_indicator + '_pointlevel_threshold.csv')
    return candidates[index]

def pointlevel_threshold(average_score_list, val_label, outp,score_indicator):
 
    point_score=np.asarray(average_score_list).reshape(len(average_score_list),1)
    #
    label_list=val_label[:,0].tolist()
    last_list=val_label[-1,1:].tolist()
    label_list.extend(last_list)
    label_array=np.asarray(label_list).reshape(len(label_list),1).astype(int)


    candidates=sorted(point_score, reverse=True)
  
    index=-1
    fmax=-1
   
    score_array=point_score
  
    val_label=label_array.astype(int)

  

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
        if F1>fmax:
            fmax=F1
            index=j
    df_out=pd.DataFrame(columns=['index','threshold','total_length'])
    df_out=df_out.append({'index':index,'threshold':candidates[index],'total_length':len(score_array)}, ignore_index=True)
    df_out.to_csv(outp+score_indicator+'_pointlevel_threshold.csv')
    return candidates[index]

def compute_metric_tau(ts_num,true_labels,pred_labels,outp,indicator,k): #传入real_label和predict_label

    outp=outp+'tau'+str(k)+'/'
    if os.path.exists(outp)==False:
        os.mkdir(outp,0o777)
    pred_labels = np.reshape(pred_labels,newshape=(len(pred_labels), 1))
    true_labels = np.reshape(true_labels,newshape=(len(true_labels), 1))

    pred_labels, true_labels =pred_labels.astype(int),true_labels.astype(int)


    data = np.concatenate([pred_labels, true_labels], axis=1)
    df = pd.DataFrame(data=data, columns=['pred', 'real'])
    df.to_csv(outp +str(ts_num)+ indicator+'pred_real_labels.csv')
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
   
    df_out = pd.DataFrame(
        [{'P': P, "R": R, 'F1': F1, 'FP rate': fpr, 'relative positive ratio': rpr}])
    df_out.to_csv(outp +str(ts_num)+indicator+ 'pred_metrics.csv')
    return

def compute_metric(ts_num,true_labels,pred_labels,outp,indicator): #传入real_label和predict_label

    pred_labels = np.reshape(pred_labels,newshape=(len(pred_labels), 1))
    true_labels = np.reshape(true_labels,newshape=(len(true_labels), 1))

    pred_labels, true_labels =pred_labels.astype(int),true_labels.astype(int)


    data = np.concatenate([pred_labels, true_labels], axis=1)
    df = pd.DataFrame(data=data, columns=['pred', 'real'])
    df.to_csv(outp +str(ts_num)+ indicator+'pred_real_labels.csv')
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
  
    df_out = pd.DataFrame(
        [{'P': P, "R": R, 'F1': F1, 'FP rate': fpr, 'relative positive ratio': rpr}])
    df_out.to_csv(outp +str(ts_num)+indicator+ 'pred_metrics.csv')
    return

def true_pointlabel(true_labels):  #true_labels为2d array形式
    true_list = true_labels[:, 0].tolist()
    last_list = true_labels[-1, 1:].tolist()
    true_list.extend(last_list)
    true_pl=np.asarray(true_list)
    # true_pl=true_pl.reshape(len(true_pl),1) #在compute_oldmetric中处理
    return true_pl

def trans_reallabel(labels,n_threshold):
    result=[]

    for i in range(len(labels)):
        single=labels[i,:]
        if np.sum(single)>=n_threshold:
            result.append(1)
        else:
            result.append(0)
    result=np.asarray(result)
    return result

def point2seq_label(point_labels,pw):#将一维pointlabel化为sequence label
    result=[]
    for i in range(len(point_labels)-pw+1):
        cur_seq=point_labels[i:i+pw]
        if cur_seq.sum()>0:
            result.append(1)
        else:
            result.append(0)
    result=np.asarray(result)
    return result
