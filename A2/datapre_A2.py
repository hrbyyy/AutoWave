
import sys
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR
import pickle
import matplotlib.pyplot as plt
from functools import reduce
from torch.nn import BatchNorm1d
import time
import datetime
import pytz
import pickle
from scipy.fftpack import fft

inpath='/mnt/A/dataset/Yahoo A1A2/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/'
sta_outp=inpath+'sta/'
if os.path.exists(sta_outp)==False:
    os.mkdir(sta_outp,0o777)
# sta_outp='/mnt/A/dataset/Yahoo A1A2/raw_sta/'


seq_ts=[]
raw_aratio_dict={}
seq_firsta_dict={}  
df_out=pd.DataFrame(columns=['ts_id','aratio','total','ano'])
df_seq=pd.DataFrame(columns=['ts_id','aratio','total','ano',"first_alarm"])
for file in os.listdir(inpath):
    if file.endswith('.csv'):

        ts_id=file.split('.')[0][10:]
        df_in=pd.read_csv(inpath+file,header=0)  
        df_in['time'] = df_in['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(int(x), pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'))

        raw_ano=df_in['is_anomaly'].sum()
        total = len(df_in)
        raw_aratio = raw_ano / total
        raw_aratio_dict[ts_id] = raw_aratio
        if raw_ano>1:
            seq_ts.append(file)
            first_alarm=df_in[df_in['is_anomaly']==1].index.tolist()[0]
            seq_firsta_dict[ts_id]=first_alarm
            df_seq = df_seq.append({'ts_id': ts_id, 'aratio': raw_aratio, 'total': total, 'ano': raw_ano,'first_alarm':first_alarm},
                                   ignore_index=True)

        df_out=df_out.append({'ts_id':ts_id,'aratio':raw_aratio,'total':total,'ano':raw_ano},ignore_index=True)


with open(sta_outp+'seqa_ts.pkl','wb') as f3:
    pickle.dump(seq_ts,f3)



def normalize(df_raw,train_points):
    fmin=df_raw.loc[:train_points,'value'].min()
    fmax=df_raw.loc[:train_points,'value'].max()
    df_raw['value']=-1+2*(df_raw['value']-fmin)/(fmax-fmin)
    return df_raw

def detrend(df_raw):
    x=list(range(len(df_raw)))
    y=df_raw['value'].values.tolist()
    x=np.asarray(x).reshape(len(x),1)
    y=np.asarray(y).reshape(len(y),1)
    model=LinearRegression()
    model.fit(x,y)
    y_hat=model.predict(x)
    detrend=y-y_hat
    label=np.asarray(df_raw['is_anomaly'].values.tolist()).reshape(len(df_raw),1)
    data=np.concatenate([detrend,label],axis=1)
    data=data.T
    return data



def set_split(dd, train_points, train_ratio, test_split):
    trainp = int(train_points * train_ratio)
    train = dd[:, :trainp]
    val1 = dd[:, trainp:train_points]
    val2 = dd[:,train_points:test_split]
    test = dd[:, test_split:]
    return train, val1, val2, test

def shapetransform(dd, window):
   

    dx, dy = [], []
  
    for i in range(dd.shape[1] - window  + 1):  
        a = dd[:, i:i + window] 
        b = a
        dx.append(a)
        dy.append(b)
    dx, dy = np.array(dx), np.array(dy)  #（#samples，#features,#timesteps）

    return dx,dy




def accu_set(inp, outp, indicator, window):
    x_list,y_list=[],[]
    for file in os.listdir(inp):
        with open(inp+file,'rb') as f:
            data_in=pickle.load(f)
        x,y=shapetransform(data_in, window)
        # if indicator=='val2' or indicator=='test':
        print(indicator+' '+str(x.shape[0]))

        x_list.append(x)
        y_list.append(y)

    X=np.concatenate(x_list,axis=0)
    Y=np.concatenate(y_list,axis=0)

    with open(outp+indicator+'x.pkl','wb') as f1:
        pickle.dump(X,f1)
    with open(outp+indicator+'y.pkl','wb') as f2:
        pickle.dump(Y,f2)
    return

trainpoints=600
train_ratio=0.8
test_split=700

window=4
data_outp=inpath+'window'+str(window)+'/'
if os.path.exists(data_outp)==False:
    os.mkdir(data_outp,0o777)
train_path=data_outp+'train/'
val1_path=data_outp+'val1/'
val2_path=data_outp+'val2/'
test_path=data_outp+'test/'
# test_path='/mnt/A/dataset/Yahoo A1A2/test/'

if os.path.exists(train_path)==False:
    os.mkdir(train_path,0o777)
if os.path.exists(test_path)==False:
    os.mkdir(test_path,0o777)
if os.path.exists(val1_path) == False:
    os.mkdir(val1_path, 0o777)
if os.path.exists(val2_path) == False:
    os.mkdir(val2_path, 0o777)

for f in seq_ts:
    ts_id = f.split('.')[0][10:]
    df_in=pd.read_csv(inpath+f,header=0)
    df_raw=normalize(df_in,trainpoints)
    data=detrend(df_raw)
    train,val1,val2,test=set_split(data,trainpoints,train_ratio,test_split)
    with open(train_path+ts_id+'.pkl','wb') as f1:
        pickle.dump(train,f1)
    with open(val1_path+ts_id+'.pkl','wb') as f2:
        pickle.dump(val1,f2)
    with open(val2_path + ts_id + '.pkl','wb') as f3:
        pickle.dump(val2, f3)
    with open(test_path + ts_id + '.pkl','wb') as f4:
        pickle.dump(test, f4)

#
accu_set(train_path,data_outp,'train',window)
accu_set(val1_path, data_outp, 'val1', window)
accu_set(val2_path, data_outp, 'val2', window)
accu_set(test_path, data_outp, 'test', window)



hw, train_ratio, k = 24, 0.8, 3
train_path=train_path+'hw'+str(hw)+'/'
test_path=test_path+'hw'+str(hw)+'/'
if os.path.exists(train_path)==False:
    os.mkdir(train_path,0o777)
if os.path.exists(test_path)==False:
    os.mkdir(test_path,0o777)

pw_list = [2, 4, 6]
for pw in pw_list:
    outp=train_path+'pw'+str(pw)+'/'
    outp_test=test_path+'pw'+str(pw)+'/'
    if os.path.exists(outp)==False:
        os.mkdir(outp,0o777)
    if os.path.exists(outp_test)==False:
        os.mkdir(outp_test,0o777)
    accu_train(inpath,outp,train_path,train_ts)
    accu_test(inpath,outp_test,val2_ts,'val2',hw,pw,k)
    accu_test(inpath,outp_test,test_ts,'test',hw,pw,k)
