import os
import numpy as np
import pandas as pd
import pickle

names=['seqMAE']



window=60

selected_ts=[16, 26, 27] 

wavelet_name='haar'
n_level=2

inpath='/mnt/A/PycharmProject/wavelet_rec/kpi/double_inputs/conv/window60/haar/level2/RMSprop/learning_rate0.0005/'

repeats=5

for ts_id in selected_ts:
    newp=inpath+str(ts_id)


    for name in names:
        df_sta = pd.DataFrame(columns=['P', 'R', 'F1', 'FP rate', 'per inference time', 'epoch_train_time'])
        for r in range(repeats):  
            # newpp=newp+'/'+str(r)+'/'+'pointlevel'+'/'
            newp1=newp +'/'+ str(r) + '/'

   

          
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



for name in names:
    df1=pd.DataFrame(columns=['P','R',	'F1','FP rate','per inference time','epoch_train_time'])
    for ts_id in selected_ts:
        # if os.path.isdir(inpath+dir) and dir!='ts10':
        dirp=inpath+str(ts_id)
            # dirp=os.path.join(inpath,dir)
        df_in=pd.read_csv(dirp+'/'+name+'round_sta.csv',header=0,index_col=0)

        df1=df1.append(df_in.loc['mean'],ignore_index=True)
        df1.loc[len(df1)-1,'ts_id']=ts_id

    df1_copy=df1


    df1['ts_id']=df1['ts_id'].apply(lambda x:float(x))
    df1.loc['mean']=df1.apply(lambda x:x.mean())
    df1.loc['std']=df1_copy.apply(lambda x:x.std())

    ts_col=df1['ts_id']
    df1=df1.drop('ts_id',axis=1)
    df1.insert(0,'ts_id',ts_col)

    df1.to_csv(inpath+name+'ts_result.csv')



