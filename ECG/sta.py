
import os
import numpy as np
import pandas as pd
import pickle
# names=['originalseq','newsequence']
names=['seqMAE']
# names=['original']  #对于ecg,逐sequence 给label，只有original result(sequence)
# dimzs=[1] # 1,5,10,15
wavelet_name='haar'
n_level=4

# hw=120
window=320
# k=3
# selected_ts=[16, 26, 27] # 21,22,
# ts_id=27
# inpath='/mnt/A/PycharmProject/wavelet_rec/ecg/conv_temporal_freq/haar/level4/'  #adam1e-5/
inpath='/mnt/A/PycharmProject/wavelet_rec/ecg/double_inputs/conv/haar/level4/adam10.0005/epoch100/' #window320/
repeats=5
#
# inpath='/mnt/A/PycharmProject/wavelet_rec/ecg/auto_statistical/haar/level4/adam10.0001/'
newp=inpath
# for ddir in os.listdir(dirp):     #遍历参数组合数
#     newp=os.path.join(dirp,ddir)

for name in names:
    df_sta = pd.DataFrame(columns=['P', 'R', 'F1', 'FP rate', 'per inference time', 'epoch_train_time'])
    for r in range(repeats):   #遍历10次循环
        # newpp=newp+'/'+str(r)+'/'+'pointlevel'+'/'
        newp1=newp + str(r) + '/'

# path=dirp+'/9units/'  #针对GRU和LSTM

        #只有评分方法不同，train time和inference time 相同
        df_accuracy=pd.read_csv(newp1 + name+'pred_metrics.csv', header=0, index_col=0)
        df_efficiency=pd.read_csv(newp1+'test_losstime.csv',header=0,index_col=0)
        df_complexity=pd.read_csv(newp1+'epoch_train_time.csv',header=0,index_col=0)
        df_sta.loc[r,['P','R','F1','FP rate']]=df_accuracy.loc[0,['P','R','F1','FP rate']]
        df_sta.loc[r, 'per inference time']=df_efficiency.loc[0,'per inference time']
        df_sta.loc[r, 'epoch_train_time']=df_complexity.loc['mean','epoch_train_time']




    df_sta2=df_sta        #copy一份，便于增设标准差
    df_sta.loc['mean']=df_sta.apply(lambda x:x.mean())
    df_sta.loc['std']=df_sta2.apply(lambda x: x.std())
    # newdir=newp+'/'+'pointlevel'
    # if os.path.exists(newdir)==False:
    #     os.mkdir(newdir,0o777)

    df_sta.to_csv(newp+'/'+name+'round_sta.csv')   #统计结果中第一行的值顺序不对，待解决！
