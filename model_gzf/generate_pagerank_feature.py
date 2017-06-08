
# coding: utf-8

# In[2]:

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
#import matplotlib.pyplot as plt


# In[3]:
path='../data/'
train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')
y_train=train.is_duplicate.values
data=pd.concat([train[['question1','question2']],test[['question1','question2']]]).reset_index()
data.drop('index',axis=1,inplace=True)
q_list={}
dd=data.values
for i in tqdm(np.arange(data.values.shape[0])):
#for i in np.arange(dd.shape[0]):
    q1,q2=dd[i]
    if q_list.setdefault(q1,[i])!=[i]:
        q_list[q1].append(i)
    if q_list.setdefault(q2,[i])!=[i]:
        q_list[q2].append(i)
data['question1_link']=data.question1.map(q_list)
data['question2_link']=data.question2.map(q_list)


# In[7]:

data['adj_node']=data.apply(lambda x:x['question1_link']+x['question2_link'],axis=1)


# In[22]:


print('train')

G=nx.Graph()
cnt=0
for i,adj_list in tqdm(enumerate(data.adj_node.values)):
    edges=[(i,item) for item in adj_list if item !=i]
    if edges==[]:
        if cnt%100000==0:
            print("ignore cnts %d" %cnt)
            cnt+=1
    else:
        G.add_edges_from(edges)  


# In[ ]:

pr=nx.pagerank(G)
pd.to_pickle(pr,path+'page_rank.pkl')

