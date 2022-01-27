import os
import numpy as np  #原有numpy为1.19.0，auc需要1.17.0
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
    # 化为Int,否则string得到的index无法和label array中位置对应！！
    real_label = real_label.astype(int)
    predict_label = predict_label.astype(int)

    real_string = ''.join(map(str, real_label))
    i = real_string.find('1')

    if i > -1:
        while i < total:
            while i < total and real_label[i] == 0:  # 定位一个真实的sequence anomaly
                i += 1  # 后续迭代中i定位到下一个1
            if i < total:
                j = i + 1
                while j < total and real_label[j] == 1:  # 定位一个真实的sequence anomaly
                    j += 1  # 跳出时，j对应0或结尾
                # 通过slice_end，统一找到0和结尾两种情况，利用slice_end定位predict_label分片结尾，否则当j=total时，定位Index越界！
                if j < total:
                    slice_end = j
                else:
                    slice_end = -1
                current_predictstring = ''.join(map(str, predict_label[i:slice_end]))
                current_firstdetection = current_predictstring.find('1')

                # 进行point_adjust:
                if predict_label[
                   i:slice_end].sum() > 0 and delay_tolerance >= current_firstdetection >= 0:  # 设置下界，防止出现find=-1，未找到的情况
                    predict_label[i:slice_end] = 1
                else:
                    predict_label[i:slice_end] = 0
                i = j  # 此时 i所指的元素为0，故还需定位下一个1
            else:
                break
    return predict_label


#和fast版本相比，vfast采用矩阵相加，将不同时刻预测的score矩阵视为score矩阵的平移。
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
    #返回每时刻列表
    final_score=score[:,0].tolist()+score[-1,1:].tolist()
    return final_score

#根据特点，将元素同其上对角元素相加，为不同时刻的预测值。
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
        num_tt=timecoverage-tt#注意此处局部变量改用tt!
        for j in range(num_tt):

            score_tt+=score_array[-1-j,j+1]
        score_tt/=num_tt
        score[tt]=score_tt
    return list(score)   #和原average_score返回数据类型一致。
# 根据MAD_GAN，将（b,s）的score_array转化为每时刻平均的score_list
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
#和pointsum_AD区别在于，对每点给label，而非sequence给label
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
    #补充pred_labels,true_labels变成sequence label
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
        F1 = 2 * P*R / (P + R)  # 尝试改用Fb(bata,定bata值)
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    if fpr==0:
        rpr='inf'
    else:
        rpr=tpr/fpr
    #用list of dict 初始化df,每个dict是一个记录
    df_out=pd.DataFrame([{'P':P,"R":R,'F1':F1,'FP rate':fpr,'relative positive ratio':rpr,'threshold':threshold}])
    df_out.to_csv(outp+score_indicator+'_pointlevel_pred_metrics.csv')
    return


# def pointlevel_threshold(val_score_array, val_label, outp,score_indicator): #定义函数求threshold      传入VN2和VA的anomaly score(合为一个列表),以及val_y(list,真实0，1label,而非regression值)
#   #原求point_score方法错误，应该用多个时刻的score均值，而非和label取法一致，将首列和末行结果拼接，只取一个时刻的值
#
#     point_score=average_score(val_score_array)
#     point_score=np.asarray(point_score)
#     #label为真实值，取相应时刻值即可
#     label_list=val_label[:,0].tolist()
#     last_list=val_label[-1,1:].tolist()
#     label_list.extend(last_list)
#     label_array=np.asarray(label_list).astype(int)
#
#     fpr,tpr,ths=roc_curve(label_array,point_score)
#     threshold=-1
#
#     fmax=-1
#     # score_array=np.asarray(sequence_score).reshape((len(sequence_score), 1))
#     score_array=point_score
#
#
#     #补充val_y变成sequence_label
#
#     for th in ths:
#         pred_label=np.where(score_array>=th,1,0)
#         #补充pred_label变成sequence label
#         # pred_label=pred_label.astype(int)
#         F1=f1_score(label_array,pred_label)
#         if F1>fmax:
#             fmax=F1
#             threshold=th
#     df_out=pd.DataFrame(columns=['threshold','total_length'])
#     df_out=df_out.append({'threshold':threshold,'total_length':len(val_score_array)}, ignore_index=True)
#     df_out.to_csv(outp+score_indicator+'_pointlevel_threshold.csv')
#     return threshold
#point score 和point label对应
def sequencelevel_threshold(valscore_array,val_label,outp,score_indicator):
    val_label = val_label.astype(int)
    # print(type(valscore_array))

    candidates = sorted(valscore_array, reverse=True)  #sorted函数将array变成list
    # print(type(candidates))

    # for i in range(len(val_score_list) - 1):
    #     candidates.append((val_score_list[i] + val_score_list[i + 1]) / 2)
    index = -1
    fmax = -1
    # score_array=np.asarray(sequence_score).reshape((len(sequence_score), 1))
    score_array = valscore_array
    # val_label=np.asarray(val_label).reshape((len(val_label), 1))


    # 补充val_y变成sequence_label

    for j in range(len(candidates)):
        pred_label = np.where(score_array >= candidates[j], 1, 0)
        # 补充pred_label变成sequence label
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
        if F1 > fmax:
            fmax = F1
            index = j
    df_out = pd.DataFrame(columns=['index', 'threshold', 'total_length'])
    df_out = df_out.append({'index': index, 'threshold': candidates[index], 'total_length': len(score_array)},
                           ignore_index=True)
    df_out.to_csv(outp + score_indicator + '_pointlevel_threshold.csv')
    return candidates[index]

def pointlevel_threshold(average_score_list, val_label, outp,score_indicator): #定义函数求threshold      传入VN2和VA的anomaly score(合为一个列表),以及val_y(list,真实0，1label,而非regression值)
  #原求point_score方法错误，应该用多个时刻的score均值，而非和label取法一致，将首列和末行结果拼接，只取一个时刻的值

    # point_score=average_score(val_score_array)
    point_score=np.asarray(average_score_list).reshape(len(average_score_list),1)
    #label为真实值，取相应时刻值即可
    label_list=val_label[:,0].tolist()
    last_list=val_label[-1,1:].tolist()
    label_list.extend(last_list)
    label_array=np.asarray(label_list).reshape(len(label_list),1).astype(int)


    candidates=sorted(point_score, reverse=True)
    # for i in range(len(val_score_list) - 1):
    #     candidates.append((val_score_list[i] + val_score_list[i + 1]) / 2)
    index=-1
    fmax=-1
    # score_array=np.asarray(sequence_score).reshape((len(sequence_score), 1))
    score_array=point_score
    # val_label=np.asarray(val_label).reshape((len(val_label), 1))
    val_label=label_array.astype(int)

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
    # 用list of dict 初始化df,每个dict是一个记录
    df_out = pd.DataFrame(
        [{'P': P, "R": R, 'F1': F1, 'FP rate': fpr, 'relative positive ratio': rpr}])
    df_out.to_csv(outp +str(ts_num)+indicator+ 'pred_metrics.csv')
    return
#indicator指示是否进行point-adjust
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
    # 用list of dict 初始化df,每个dict是一个记录
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
