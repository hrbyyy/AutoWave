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
# downsampled_path=inpath+'downsampled/'
stats_outp='/mnt/A/dataset/kpi/alarm_index_stats/original/April/'
#注：newts_outp和final_outp中ts的 training已经无异常（根据split_dict,其前位置已无异常）
newts_outp='/mnt/A/dataset/kpi/normaltrain_ts/original/'
data_inpath='/mnt/A/dataset/kpi/final_for_model/original/April/'
final_outp='/mnt/A/dataset/kpi/final_for_model/original/April/reconstruction/'
# if os.path.exists(downsampled_path) == False:
#     os.mkdir(downsampled_path, 0o777)
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
# ids=train['KPI ID'].unique()    #29个ID
# print(len(ids))
# unique_ids=train['KPI ID'].value_counts()
# print(train['KPI ID'].value_counts())  #value_counts返回Series形式
# # length_dict=train.set_index('0')[]
# length_dict=unique_ids.to_dict()
# pickle.dump(length_dict,open(inpath+'id_length.pkl','wb'))



#step1:将原始train按KPI ID化为每个ID对应一个ts
# num=1
# for kpi in ids:
#     print(kpi)
#     df_single=train.loc[train['KPI ID']==kpi]
#     df_single.reset_index(inplace=True,drop=True)
#     df_single.to_csv(inpath+str(num)+'ts.csv')
#     num=num+1

#ts=14,ts=15人工调整 若范围限制在+-200,则ts9,10也需要人工调整。
#确定val2,test分割点，使得a_ratio相差不超过0.1
def tune_set(df_vt):

    cols=df_vt.columns
    total=len(df_vt)
    n_a=(df_vt['label']==1).sum()
    a_ratio=n_a/total
    point=-1
    v_a,v_aratio,t_a,t_aratio=-1,-1,-1,-1
    for i in range(df_vt.index[0]+100,df_vt.index[-1],100):   #range(len(df_vt)//2-500,len(df_vt)//2+500,100)
        df_v=df_vt.loc[df_vt.index[0]:i-1,cols]  #df.loc中分片和list不同，包含截止点
        df_t=df_vt.loc[i-1:,cols]
        print(len(df_v),len(df_t))
        va=(df_v['label']==1).sum()
        varatio=v_a/len(df_v)
        ta=(df_t['label']==1).sum()
        taratio=t_a/len(df_t)
        # diff=np.abs(v_aratio-t_aratio)
        # print(df_v.head(5))
        # print(df_t.head(5))
        # print(diff)
        if np.abs(v_aratio-t_aratio)<1e-4 and np.abs(v_aratio-a_ratio)<1e-4 and  np.abs(t_aratio-a_ratio)<1e-4 :
            point=i #直接返回原始df中的index
            v_a=va
            v_aratio=varatio
            t_a=ta
            t_aratio=taratio
            # print(len(df_vt),point,diff)
            break
    return point,v_a,v_aratio,t_a,t_aratio
#最终用此结果，结合人工筛选合适的ts
#方法二：确定val2,test 1/2异常对应的位置，反求val2,test的异常比例
def fixp_ratio(df_vt):
    col=df_vt.columns
    a_idx=df_vt[df_vt['label']==1].index.tolist()
    split_p=a_idx[len(a_idx)//2] #half anomaly的位置
    df_v=df_vt.loc[:split_p,col]
    df_te=df_vt.loc[split_p:,col]
    v_a=(df_v['label']==1).sum()
    v_aratio=v_a/len(df_v)
    t_a=(df_te['label']==1).sum()
    t_aratio=t_a/len(df_te)
    print(len(df_v),len(df_te))
    return split_p,v_a,v_aratio,t_a,t_aratio,len(df_v),len(df_te)

#方法三：确定val2+test总异常比例，调整val2大小，接近总比例。
def tune_val():
    df_sta=pd.DataFrame(columns=['ts','l_vt','vt_a','vt_aratio','a_idx'])
    for i in range(1,30):

        df_in = pd.read_csv(inpath + str(i) + 'ts.csv', header=0, index_col=0)
        total=len(df_in)
        df_vt=df_in[total//2:]
        # cols = df_vt.columns
        vt_total = len(df_vt)
        n_a = (df_vt['label'] == 1).sum()
        a_ratio = n_a / vt_total

        a_idx=df_vt[df_vt['label']==1].index.tolist()
        df_sta=df_sta.append({'ts':i,'l_vt':vt_total,'vt_a':n_a,'vt_aratio':a_ratio,'a_idx':a_idx},ignore_index=True)
    df_sta.to_csv(stats_outp+'vt_sta.csv')
    return
# tune_val()

#对原始数据，固定将前1/2作为train(含val1),后1/2划分val2和test,调整分割点，使得val2和test异常比例一致。
def cal_sta():
    vt_split_dict={}
    train_dict={}
    # df_sta = pd.DataFrame(
    #     columns=['ts_id', 'total', 'total_a', 'a_ratio', 'ndrop_a', 'val2_test_split', 'val2_a', 'val2_aratio',
    #              'test_a', 'test_aratio'])
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
        # df_train.reset_index(inplace=True,drop=False) #保存原始index,一是便于shape_transform non_cross,二是便于利用后方确定的val2_test split point
        df_vt=df_in[total//2:]  #注意，截取后半段后，df的index仍为原总df中的index！需reset。

        # point,v_a,v_ar,t_a,t_ar=tune_set(df_vt)
        point, v_a, v_ar, t_a, t_ar,l_v,l_te = fixp_ratio(df_vt)
        # print(i,point)
        train_dict[i]=len(df_train)
        vt_split=point   #由于一直未改变index,此处的index即为原df中的index
        # vt_split=total//2+point  #(在整个原ts中定位分割点)
        final_df=pd.concat([df_train,df_vt],axis=0)
        final_df.reset_index(inplace=True,drop=False)

        #final_vtsplit保存的是删除train中异常后，新的ts的val2_test分割点
        final_vtsplit=final_df[final_df['index']==vt_split].index.tolist()[0]
        # df_sta = df_sta.append({'ts_id': i, 'total': total, 'total_a': a_sum, 'a_ratio': a_ratio, 'ndrop_a': num_dropa,
        #                         'val2_test_split': final_vtsplit,
        #                          'val2_a': v_a, 'val2_aratio': v_ar,  'test_a': t_a,
        #                         'test_aratio': t_ar}, ignore_index=True)
        df_sta=df_sta.append({'ts_id':i,'total':total,'total_a':a_sum,'a_ratio':a_ratio,'ndrop_a':num_dropa,'val2_test_split':final_vtsplit,
                              'l_val2':l_v,'val2_a':v_a,'val2_aratio':v_ar,'l_test':l_te,'test_a':t_a,'test_aratio':t_ar},ignore_index=True)
        vt_split_dict[i]=final_vtsplit
        data_df=final_df[['index','value','label']]  #保存index,便于分割train时，non_cross
        data=data_df.values.T
        #保存的ts只进行了剔除train中anomaly，未进行归一化
        with open(final_outp+str(i)+'.pkl','wb') as f3:
            pickle.dump(data,f3)

        # print(final_vtsplit)

    #
    # df_sta.to_csv(stats_outp+'fixp_sta.csv')
    # with open(stats_outp+'fixpvt_split_dict.pkl','wb') as f:
    #     pickle.dump(vt_split_dict,f)
    # with open(stats_outp+'train_split_dict.pkl','wb') as f2:
    #     pickle.dump(train_dict,f2)
    return
# cal_sta()
#step1:每个ts min_max归一化至[-1,1]  #dd中含有原始index,故value位于1
def normalize(dd,train_points):
    fmin=dd[1,:train_points].min()
    fmax=dd[1,:train_points].max()
    dd[1,:]=-1+2*(dd[1,:]-fmin)/(fmax-fmin)
    return dd

# test_split为test和val2在原ts中对应的分界点。
def set_split(dd, train_points, train_ratio, test_split):
    trainp = int(train_points * train_ratio)
    train = dd[:, :trainp]
    val1 = dd[:, trainp:train_points]
    val2 = dd[:,train_points:test_split]
    test = dd[:, test_split:]
    return train, val1, val2, test

def shapetransform_train(dd, window_size):
    #dd=pd.DataFrame(data=df_in.values.T,columns=df_in.index,index=df_in.columns)#df_in转置,化为（#features,#timesteps形式）
    #需要的是数组，直接转置即可
    # dd=df_in.values.T
    dx = []
    # totalslot = dd.shape[1] - look_back
    # ltrain = int(totalslot * train_ratio)
    # lval = int(totalslot * val_ratio)
    #以对后方注意内容调整。注意：此处range范围只适用于look_forward=1的情况，后续通用需要调整
    # for i in range(len(df_in) - look_back - look_forward + 1):  # 遍历每个样本，按时间段划分train_val_test,处理成regression变量形式
        # if df_in.iloc[i+look_back+look_forward-1,0]!=df_in.iloc[i,0]+look_back+look_forward-1:
    for i in range(dd.shape[1]- window_size+ 1):
        if dd[0,i+window_size-1]!=dd[0,i]+window_size-1:
            i+=window_size-1
            continue

        a = dd[1:, i:i +window_size]  #后两行为p和alarm，转置

        dx.append(a)

    dx= np.array(dx)  #（#samples，#features,#timesteps）

    return dx


def shapetransform(dd, window_size):
    #dd=pd.DataFrame(data=df_in.values.T,columns=df_in.index,index=df_in.columns)#df_in转置,化为（#features,#timesteps形式）
    #需要的是数组，直接转置即可

    dx = []
    # totalslot = dd.shape[1] - look_back
    # ltrain = int(totalslot * train_ratio)
    # lval = int(totalslot * val_ratio)
    #以对后方注意内容调整。注意：此处range范围只适用于look_forward=1的情况，后续通用需要调整
    for i in range(dd.shape[1] - window_size + 1):  # 遍历每个样本，按时间段划分train_val_test,处理成regression变量形式
        a = dd[1:, i:i + window_size]  #后两行为p和alarm，转置
        dx.append(a)
    dx = np.array(dx) #（#samples，#features,#timesteps）

    return dx

def compute_topk_freq(array_in,k):
    ts=array_in[:,0,:]  #只取uts维度(b,t)
    F_x = fft(ts)  # 默认在最后一维做FFT
    A_f = np.abs(F_x)   #（b,freq）其中freq=t
    sorted_Af=np.sort(A_f) #按行排序 #升序排列，取后k个
    topK_freq=sorted_Af[:,-k:]
    return topK_freq


def dump_data(dd,window_size,  indicator, outp):
    if indicator=='train':
        x=shapetransform_train(dd,window_size)
    else:
        x=shapetransform(dd,window_size)
    # x_kf=compute_topk_freq(x,k)
    with open(outp + indicator + 'x.pkl', 'wb') as f1:
        pickle.dump(x, f1)
    # with open(outp + indicator + 'y.pkl', 'wb') as f2:
    #     pickle.dump(y, f2)
    # with open(outp + indicator + '_kf.pkl', 'wb') as f3:
    #     pickle.dump(x_kf, f3)
    return


# selected_ts=[1,16,21,22,26,27]
selected_ts=[ 16,26,27,28,12,29,15,17,19,25]  #后续和16,26,27汇总 ,28,12,29,15,17,19,25
window_size=10

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
        ts_data=pickle.load(f1) #含index,value,label,（f,t）形式
    ts_n=normalize(ts_data,train_point)
    train, val1, val2, test = set_split(ts_n,train_point,train_ratio,vt_split_point)
    dump_data(train,window_size,'train',outp)
    dump_data(val1, window_size, 'val1', outp)
    dump_data(val2, window_size, 'val2', outp)
    dump_data(test, window_size, 'test', outp)
#


