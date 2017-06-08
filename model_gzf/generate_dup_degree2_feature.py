
# coding: utf-8

# In[157]:

import pandas as pd
import numpy as np
from tqdm import *


# In[158]:

train=pd.read_csv('data/train.csv')


# In[159]:

test=pd.read_csv('data/test.csv')


# In[160]:

data=pd.concat([train[['question1','question2']],test[['question1','question2']]]).reset_index()


# In[161]:

data.drop('index',axis=1,inplace=True)


# In[162]:

q_list={}
dd=data.values
#for i in tqdm(np.arange(data.values.shape[0])):
for i in np.arange(dd.shape[0]):
    q1,q2=dd[i]
    if q_list.setdefault(q1,[i])!=[i]:
        q_list[q1].append(i)
    if q_list.setdefault(q2,[i])!=[i]:
        q_list[q2].append(i)


# In[163]:

data['question1_link']=data.question1.map(q_list)
data['question2_link']=data.question2.map(q_list)


# In[164]:

link=data[['question1_link','question2_link']].values


# In[219]:

#link_d2_cnt=[]
link1_d2=[]                #degree2 sum q1-q1_dups
link2_d2=[]                #degree2 sum q2-q2_dups
link1_2_d2=[]              #degree2 sum q1-q2_dups
link2_1_d2=[]               # degree2 sum q2_q1_dups
link_d2=[]                  # degrees sum all
#for i in tqdm(range(link.shape[0])):
for i in range(link.shape[0]):
    if i%100000==0:
        print i
    ############
    ctx=[]
    node1=[]
    ctx.extend(link[link[i,0],0])
    for items in ctx:
        node1.extend(items)
    node1=set(node1)
    link1_d2.append(len(node1))
    #########
    ctx=[]
    node2=[]
    ctx.extend(link[link[i,1],1])
    for items in ctx:
        node2.extend(items)
    node2=set(node2)
    link2_d2.append(len(node2))
    ##################
    ctx=[]
    node1_2=[]
    ctx.extend(link[link[i,0],1])
    for items in ctx:
        node1_2.extend(items)
    node1_2=set(node1_2)
    link1_2_d2.append(len(node1_2))
    ################
    ctx=[]
    node2_1=[]
    ctx.extend(link[link[i,1],0])
    for items in ctx:
        node2_1.extend(items)
    node2_1=set(node2_1)
    link2_1_d2.append(len(node2_1))
    
    node=node1.union(node2).union(node1_2).union(node2_1)
    link_d2.append(len(node))
    #link_d2.append(node)
    #link_d2_cnt.append(len(node))


# In[225]:

link2_feature=np.vstack([link1_d2,link2_d2,link1_2_d2,link2_1_d2,link_d2]).T


# In[228]:

link2_feature=pd.DataFrame(link2_feature,columns=['q1_q1','q2_q2','q1_q2','q2_q1','all_link'])


# In[237]:

#link2_feature['min_d2']=link2_feature.apply(lambda x:min(x['q1_q1'],x['q2_q2'],x['q1_q2'],x['q2_q1']),axis=1)
#link2_feature['max_d2']=link2_feature.apply(lambda x:max(x['q1_q1'],x['q2_q2'],x['q1_q2'],x['q2_q1']),axis=1)
link2_feature['min_d2']=link2_feature.values[:,:-1].min(axis=1)
link2_feature['max_d2']=link2_feature.values[:,:-1].max(axis=1)


# In[238]:

link2_feature['ratio_min_max']=link2_feature['min_d2']*1.0/link2_feature['max_d2']


# In[239]:

train_len=train.shape[0]


# In[240]:

pd.to_pickle(link2_feature[:train_len],'data/train_link_d2.pkl')
pd.to_pickle(link2_feature[train_len:],'data/test_link_d2.pkl')

