
# coding: utf-8

# In[13]:

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from csv import DictReader
#import matplotlib.pyplot as plt

#from config import path
path='data/'

# In[15]:

def prepare_graph(paths):
    G = nx.DiGraph()

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

max_clique=nx.pagerank(G,alpha=0.9)

def prepare_hash_clique_stats(path,out,idf_dict):

    print path
    c = 0
    start = datetime.now()
    header = [                
        'min_pr',
        'max_pr',
        ]
    header = ','.join(header)
    with open(out, 'w') as outfile:
        outfile.write('%s\n'%header)
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            # q1 = hash(q1)
            # q2 = hash(q2)
            
            q1_idf = idf_dict.get(q1,0.0)
            q2_idf = idf_dict.get(q2,0.0)
            
            max_q1 = max((q1_idf,q2_idf))
            min_q1 = min((q1_idf,q2_idf))


            outfile.write('%s,%s\n' % (
                max_q1,
                min_q1,
                ))
            
            c+=1
            end = datetime.now()


    print 'times:',end-start


prepare_hash_clique_stats(path+'train_hashed.csv',path+'train_pagerank_directed.csv',max_clique)

prepare_hash_clique_stats(path+'test_hashed.csv',path+'test_pagerank_directed.csv',max_clique)
