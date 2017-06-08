
# coding: utf-8

# In[10]:

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as sps
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[2]:

train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')
y_train=train.is_duplicate.values


# In[3]:

pr=pd.read_pickle('data/page_rank.pkl')                                         ###node as index
hashed_pr=pd.read_pickle('data/pr.pkl')                                         ###edge as index
clique=pd.read_pickle('data/max_clique.pkl')                                    ###node as index
hashed_clique=pd.read_pickle('data/max_clique_edges.pkl')                       ###edge as index


# In[4]:

train_hashed=pd.read_csv('data/train_hashed.csv')
test_hashed=pd.read_csv('data/test_hashed.csv')


# In[5]:

for fe in train_hashed.columns:
    train_hashed[fe+'_pr']=train_hashed[fe].astype(str).map(hashed_pr)
    test_hashed[fe+'_pr']=test_hashed[fe].astype(str).map(hashed_pr)
    train_hashed[fe+'_clique']=train_hashed[fe].astype(str).map(hashed_clique)
    test_hashed[fe+'_clique']=test_hashed[fe].astype(str).map(hashed_clique)


# In[6]:

train_hashed['pr_max']=train_hashed[['question1_hash_pr','question2_hash_pr']].values.max(axis=1)
test_hashed['pr_max']=test_hashed[['question1_hash_pr','question2_hash_pr']].values.max(axis=1)
train_hashed['pr_min']=train_hashed[['question1_hash_pr','question2_hash_pr']].values.min(axis=1)
test_hashed['pr_min']=test_hashed[['question1_hash_pr','question2_hash_pr']].values.min(axis=1)


# In[7]:

train_hashed['pr_max_ratio_pr_min']=train_hashed.pr_max/train_hashed.pr_min
train_hashed['pr_dis']=train_hashed.pr_max-train_hashed.pr_min
train_hashed['pr_sum']=train_hashed.pr_max+train_hashed.pr_min

test_hashed['pr_max_ratio_pr_min']=test_hashed.pr_max/test_hashed.pr_min
test_hashed['pr_dis']=test_hashed.pr_max-test_hashed.pr_min
test_hashed['pr_sum']=test_hashed.pr_max+test_hashed.pr_min


# In[9]:

train_hashed['clique_max']=train_hashed[['question1_hash_clique','question2_hash_clique']].values.max(axis=1)
test_hashed['clique_max']=test_hashed[['question1_hash_clique','question2_hash_clique']].values.max(axis=1)
train_hashed['clique_min']=train_hashed[['question1_hash_clique','question2_hash_clique']].values.min(axis=1)
test_hashed['clique_min']=test_hashed[['question1_hash_clique','question2_hash_clique']].values.min(axis=1)
train_hashed['clique_max_ratio_clique_min']=train_hashed.clique_max/train_hashed.clique_min
train_hashed['clique_dis']=train_hashed.clique_max-train_hashed.clique_min
train_hashed['clique_sum']=train_hashed.clique_max+train_hashed.clique_min

test_hashed['clique_max_ratio_clique_min']=test_hashed.clique_max/test_hashed.clique_min
test_hashed['clique_dis']=test_hashed.clique_max-test_hashed.clique_min
test_hashed['clique_sum']=test_hashed.clique_max+test_hashed.clique_min


# In[11]:

for fe in train_hashed.columns:
    print fe,sps.spearmanr(train_hashed[fe],y_train)


# In[13]:

sel_hashed_cols=['question1_hash_pr',
       'question1_hash_clique', 'question2_hash_pr',
       'question2_hash_clique', 'pr_max', 'pr_min', 'pr_max_ratio_pr_min',
       'pr_dis', 'pr_sum', 'clique_max', 'clique_min',
       'clique_max_ratio_clique_min', 'clique_dis', 'clique_sum']


# In[16]:

pd.to_pickle(train_hashed[sel_hashed_cols],'data/train_edges_features.pkl')
pd.to_pickle(test_hashed[sel_hashed_cols],'data/test_edges_features.pkl')


# In[17]:

pr_features=np.zeros(train.shape[0]+test.shape[0])
for key,val in pr.items():
    pr_features[key]=val


# In[18]:

pd.to_pickle(pr_features[:train.shape[0]],'data/train_pr_node_feature.pkl')
pd.to_pickle(pr_features[train.shape[0]:],'data/test_pr_node_feature.pkl')


# In[ ]:



