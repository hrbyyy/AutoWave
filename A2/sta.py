import os
import numpy as np
import pandas as pd
import pickle
names=['seqMAE'] #'original','point_adjust','originalseq','newsequence',

inpath='/mnt/A/PycharmProject/wavelet_rec/A2/double_inputs/8-4-1/haar/level2/'
#step1 统计每个参数组合下10次运行的pred_metrics统计均值
# for dir in os.listdir(inpath):   #遍历pw
repeats=5
# for level in level_list:
#     newp=inpath+'level'+str(level)


for name in names:
    df_sta = pd.DataFrame(columns=['P', 'R', 'F1', 'fpr', 'per inference time', 'epoch_train_time'])
    for r in range(repeats):   #遍历10次循环
        # newpp=newp+'/'+str(r)+'/'+'pointlevel'+'/'
        newp1=inpath +'/'+ str(r) + '/'

# path=dirp+'/9units/'  #针对GRU和LSTM

        #只有评分方法不同，train time和inference time 相同
        df_accuracy=pd.read_csv(newp1 + name+'_metrics.csv', header=0, index_col=0)
        df_efficiency=pd.read_csv(newp1+'test_losstime.csv',header=0,index_col=0)
        df_complexity=pd.read_csv(newp1+'epoch_train_time.csv',header=0,index_col=0)
        df_sta.loc[r,['P','R','F1','fpr']]=df_accuracy.loc['mean',['P','R','F1','fpr']]
        df_sta.loc[r, 'per inference time']=df_efficiency.loc[0,'per inference time']
        df_sta.loc[r, 'epoch_train_time']=df_complexity.loc['mean','epoch_train_time']




    df_sta2=df_sta        #copy一份，便于增设标准差
    df_sta.loc['mean']=df_sta.apply(lambda x:x.mean())
    df_sta.loc['std']=df_sta2.apply(lambda x: x.std())
    # newdir=newp+'/'+'pointlevel'
    # if os.path.exists(newdir)==False:
    #     os.mkdir(newdir,0o777)

    df_sta.to_csv(inpath+'/'+name+'round_sta.csv')   #统计结果中第一行的值顺序不对，待解决！

