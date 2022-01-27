import os
import numpy as np
import pandas as pd
import pickle
names=['seqMAE'] #

inpath='/mnt/A/PycharmProject/wavelet_rec/A2/double_inputs/8-4-1/haar/level2/'

repeats=5



for name in names:
    df_sta = pd.DataFrame(columns=['P', 'R', 'F1', 'fpr', 'per inference time', 'epoch_train_time'])
    for r in range(repeats):   
        # newpp=newp+'/'+str(r)+'/'+'pointlevel'+'/'
        newp1=inpath +'/'+ str(r) + '/'


        df_accuracy=pd.read_csv(newp1 + name+'_metrics.csv', header=0, index_col=0)
        df_efficiency=pd.read_csv(newp1+'test_losstime.csv',header=0,index_col=0)
        df_complexity=pd.read_csv(newp1+'epoch_train_time.csv',header=0,index_col=0)
        df_sta.loc[r,['P','R','F1','fpr']]=df_accuracy.loc['mean',['P','R','F1','fpr']]
        df_sta.loc[r, 'per inference time']=df_efficiency.loc[0,'per inference time']
        df_sta.loc[r, 'epoch_train_time']=df_complexity.loc['mean','epoch_train_time']




    df_sta2=df_sta       
    df_sta.loc['mean']=df_sta.apply(lambda x:x.mean())
    df_sta.loc['std']=df_sta2.apply(lambda x: x.std())
   
    df_sta.to_csv(inpath+'/'+name+'round_sta.csv')   

