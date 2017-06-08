
# coding: utf-8

# In[13]:

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from csv import DictReader
#import matplotlib.pyplot as plt

from config import path


# In[15]:

def prepare_graph(paths):
    G = nx.Graph()

    idf_dict = dict()
    for path in paths:
        print(path)
        c = 0
        start = datetime.now()

        for t, row in enumerate(DictReader(open(path), delimiter=',')):
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            G.add_edge(q1,q2)
            c+=1
    end = datetime.now()
    print('times:',end-start)
    return G
    
G = prepare_graph([path+'train_hashed.csv',path+'test_hashed.csv'])


# In[13]:

cnt=0
max_clique=dict()
for clique in nx.find_cliques(G):
    if cnt%100000==0:
        print("deal cnts %d" %cnt)
    len_clique=len(clique)
    for item in clique:
        c = max_clique.get(item,0)
        c+=1
        max_clique[item]=c  
    cnt+=1


# In[ ]:

pd.to_pickle(max_clique,path+'max_clique_edges_belong.pkl')

max_clique = pd.read_pickle(path+'max_clique_edges_belong.pkl')

def prepare_hash_clique_belong(path,out,idf_dict):

    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_hash_clique_belong,question2_hash_clique_belong\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            # q1 = hash(q1)
            # q2 = hash(q2)
            
            q1_idf = idf_dict.get(q1,0)
            q2_idf = idf_dict.get(q2,0)
            
            outfile.write('%s,%s\n' % (q1_idf, q2_idf))
            
            c+=1
            end = datetime.now()


    print 'times:',end-start


prepare_hash_clique_belong(path+'train_hashed.csv',path+'train_hashed_clique_belong.csv',max_clique)

prepare_hash_clique_belong(path+'test_hashed.csv',path+'test_hashed_clique_belong.csv',max_clique)
