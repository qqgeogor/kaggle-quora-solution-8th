# coding: utf-8

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
#import matplotlib.pyplot as plt


path='../input/'
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



data['adj_node']=data.apply(lambda x:x['question1_link']+x['question2_link'],axis=1)



G=nx.Graph()
cnt=0
for i,adj_list in tqdm(enumerate(data.adj_node.values)):
#for i,adj_list in enumerate(data.adj_node.values):
    edges=[(i,item) for item in adj_list if item !=i]
    if edges==[]:
#        if cnt%100000==0:
#            print("ignore cnts %d" %cnt)
        cnt+=1
    else:
        G.add_edges_from(edges)  
print("ignore cnts %d" %cnt)

print('begin train')
cnt=0
max_clique=np.zeros(data.shape[0])
for clique in tqdm(nx.find_cliques(G)):
#for clique in nx.enumerate_all_cliques(G):
    if cnt%100000==0:
        print("deal cnts %d" %cnt)
    len_clique=len(clique)
    for item in clique:
        max_clique[item]=max(max_clique[item],len_clique)
    cnt+=1


print("totally max_clique %d" %cnt)

pd.to_pickle(max_clique,path+'max_clique.pkl')

