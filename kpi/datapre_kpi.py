#--coding:utf-8--
#

import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pickle

import matplotlib.pyplot as plt

import pyedflib
import datetime
import pytz
from scipy.fftpack import fft

inpath='/mnt/A/dataset/kpi/'

stats_outp='/mnt/A/dataset/kpi/alarm_index_stats/original/April/'

newts_outp='/mnt/A/dataset/kpi/normaltrain_ts/original/'
data_inpath='/mnt/A/dataset/kpi/final_for_model/original/April/'
final_outp='/mnt/A/dataset/kpi/final_for_model/original/April/reconstruction/'

if os.path.exists(stats_outp)==False:
    os.mkdir(stats_outp,0o777)
if os.path.exists(newts_outp) == False:
    os.mkdir(newts_outp, 0o777)
if os.path.exists(final_outp) == False:
    os.mkdir(final_outp, 0o777)

# train=pd.read_csv(inpath+'phase2_train.csv',header=0)
#
# columns=train.columns  #['timestamp', 'value', 'label', 'KPI ID']
# length=len(train)
# ids=train['KPI ID'].unique()   
# print(len(ids))
# unique_ids=train['KPI ID'].value_counts()
# print(train['KPI ID'].value_counts()) 
# # length_dict=train.set_index('0')[]
# length_dict=unique_ids.to_dict()
# pickle.dump(length_dict,open(inpath+'id_length.pkl','wb'))




num=1
for kpi in ids:
    print(kpi)
    df_single=train.loc[train['KPI ID']==kpi]
    df_single.reset_index(inplace=True,drop=True)
    df_single.to_csv(inpath+str(num)+'ts.csv')
    num=num+1



def fixp_ratio(df_vt):
    col=df_vt.columns
    a_idx=df_vt[df_vt['label']==1].index.tolist()
    split_p=a_idx[len(a_idx)//2]
    df_v=df_vt.loc[:split_p,col]
    df_te=df_vt.loc[split_p:,col]
    v_a=(df_v['label']==1).sum()
    v_aratio=v_a/len(df_v)
    t_a=(df_te['label']==1).sum()
    t_aratio=t_a/len(df_te)
    print(len(df_v),len(df_te))
    return split_p,v_a,v_aratio,t_a,t_aratio,len(df_v),len(df_te)


def cal_sta():
    vt_split_dict={}
    train_dict={}
   
    df_sta = pd.DataFrame(
        columns=['ts_id', 'total', 'total_a', 'a_ratio', 'ndrop_a', 'val2_test_split','l_val2', 'val2_a', 'val2_aratio',
                 'l_test','test_a', 'test_aratio'])
    for i in range(1,30):

        df_in = pd.read_csv(inpath + str(i) + 'ts.csv', header=0, index_col=0)
        total=len(df_in)
        a_sum=(df_in['label']==1).sum()
        a_ratio=a_sum/total
        df_train=df_in[:total//2]
        train_a=df_train[df_train['label']==1].index.tolist()
        num_dropa=len(train_a)
        df_train=df_train.drop(train_a)
        # df_train.reset_index(inplace=True,drop=False) 
        df_vt=df_in[total//2:]  

        point, v_a, v_ar, t_a, t_ar,l_v,l_te = fixp_ratio(df_vt)
        
        train_dict[i]=len(df_train)
        vt_split=point   
       
        final_df=pd.concat([df_train,df_vt],axis=0)
        final_df.reset_index(inplace=True,drop=False)

      
        final_vtsplit=final_df[final_df['index']==vt_split].index.tolist()[0]
      
        df_sta=df_sta.append({'ts_id':i,'total':total,'total_a':a_sum,'a_ratio':a_ratio,'ndrop_a':num_dropa,'val2_test_split':final_vtsplit,
                              'l_val2':l_v,'val2_a':v_a,'val2_aratio':v_ar,'l_test':l_te,'test_a':t_a,'test_aratio':t_ar},ignore_index=True)
        vt_split_dict[i]=final_vtsplit
        data_df=final_df[['index','value','label']]  #保存index,便于分割train时，non_cross
        data=data_df.values.T
      
        with open(final_outp+str(i)+'.pkl','wb') as f3:
            pickle.dump(data,f3)

      
    return


def normalize(dd,train_points):
    fmin=dd[1,:train_points].min()
    fmax=dd[1,:train_points].max()
    dd[1,:]=-1+2*(dd[1,:]-fmin)/(fmax-fmin)
    return dd


def set_split(dd, train_points, train_ratio, test_split):
    trainp = int(train_points * train_ratio)
    train = dd[:, :trainp]
    val1 = dd[:, trainp:train_points]
    val2 = dd[:,train_points:test_split]
    test = dd[:, test_split:]
    return train, val1, val2, test

def shapetransform_train(dd, window_size):
   
    dx = []
   
    for i in range(dd.shape[1]- window_size+ 1):
        if dd[0,i+window_size-1]!=dd[0,i]+window_size-1:
            i+=window_size-1
            continue

        a = dd[1:, i:i +window_size] 

        dx.append(a)

    dx= np.array(dx)  #（#samples，#features,#timesteps）

    return dx


def shapetransform(dd, window_size):
   
    dx = []
   
    for i in range(dd.shape[1] - window_size + 1): 
        a = dd[1:, i:i + window_size] 
        dx.append(a)
    dx = np.array(dx) #（#samples，#features,#timesteps）

    return dx

def compute_topk_freq(array_in,k):
    ts=array_in[:,0,:]  
    F_x = fft(ts)  
    A_f = np.abs(F_x)  
    sorted_Af=np.sort(A_f) 
    topK_freq=sorted_Af[:,-k:]
    return topK_freq


def dump_data(dd,window_size,  indicator, outp):
    if indicator=='train':
        x=shapetransform_train(dd,window_size)
    else:
        x=shapetransform(dd,window_size)
   
    with open(outp + indicator + 'x.pkl', 'wb') as f1:
        pickle.dump(x, f1)
    with open(outp + indicator + 'y.pkl', 'wb') as f2:
        pickle.dump(y, f2)
    with open(outp + indicator + '_kf.pkl', 'wb') as f3:
        pickle.dump(x_kf, f3)
    return



selected_ts=[ 16,26,27]  
window_size=60

train_ratio=0.8
with open(stats_outp+'train_split_dict.pkl','rb') as f2:
    train_dict=pickle.load(f2)
with open(stats_outp+'fixpvt_split_dict.pkl','rb') as f3:
    val2_test_split_dict=pickle.load(f3)


outp1=final_outp+'window'+str(window_size)+'/'
if os.path.exists(outp1)==False:
    os.mkdir(outp1,0o777)
for i in selected_ts:
    outp=outp1+str(i)+'/'
    if os.path.exists(outp)==False:
        os.mkdir(outp,0o777)
    train_point=train_dict[i]
    vt_split_point=val2_test_split_dict[i]
    with open(data_inpath+str(i)+'.pkl','rb') as f1:
        ts_data=pickle.load(f1) 
    ts_n=normalize(ts_data,train_point)
    train, val1, val2, test = set_split(ts_n,train_point,train_ratio,vt_split_point)
    dump_data(train,window_size,'train',outp)
    dump_data(val1, window_size, 'val1', outp)
    dump_data(val2, window_size, 'val2', outp)
    dump_data(test, window_size, 'test', outp)
#


