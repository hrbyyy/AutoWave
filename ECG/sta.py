
import os
import numpy as np
import pandas as pd
import pickle

names=['seqMAE']

wavelet_name='haar'
n_level=4


window=320

inpath='/mnt/A/PycharmProject/wavelet_rec/ecg/double_inputs/conv/haar/level4/adam10.0005/epoch100/' 
repeats=5

newp=inpath


for name in names:
    df_sta = pd.DataFrame(columns=['P', 'R', 'F1', 'FP rate', 'per inference time', 'epoch_train_time'])
    for r in range(repeats):  
        # newpp=newp+'/'+str(r)+'/'+'pointlevel'+'/'
        newp1=newp + str(r) + '/'



       
        df_accuracy=pd.read_csv(newp1 + name+'pred_metrics.csv', header=0, index_col=0)
        df_efficiency=pd.read_csv(newp1+'test_losstime.csv',header=0,index_col=0)
        df_complexity=pd.read_csv(newp1+'epoch_train_time.csv',header=0,index_col=0)
        df_sta.loc[r,['P','R','F1','FP rate']]=df_accuracy.loc[0,['P','R','F1','FP rate']]
        df_sta.loc[r, 'per inference time']=df_efficiency.loc[0,'per inference time']
        df_sta.loc[r, 'epoch_train_time']=df_complexity.loc['mean','epoch_train_time']




    df_sta2=df_sta       
    df_sta.loc['mean']=df_sta.apply(lambda x:x.mean())
    df_sta.loc['std']=df_sta2.apply(lambda x: x.std())
  

    df_sta.to_csv(newp+'/'+name+'round_sta.csv')   
