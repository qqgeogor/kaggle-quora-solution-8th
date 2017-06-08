
# coding: utf-8

# In[13]:

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from csv import DictReader
import itertools
#import matplotlib.pyplot as plt

from config import path



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




cnt=0
max_clique=dict()
for clique in nx.find_cliques(G):
    if cnt%100000==0:
        print("deal cnts %d" %cnt)
    len_clique=len(clique)

    tc = 0

    for item1,item2 in itertools.combinations(clique,2):
        if tc>=9999:
            break
        if item2 in G[item1]:
            tc+=1
    
    for item in clique:
        tc= max(max_clique.get(item,0),tc)
        max_clique[item]=tc
    cnt+=1


pd.to_pickle(max_clique,path+'/max_clique_total.pkl')

max_clique = pd.read_pickle(path+'max_clique_total.pkl')

def prepare_hash_clique_total(path,out,idf_dict):

    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('q_min,q_max,q_mean,q_diff\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            # q1 = hash(q1)
            # q2 = hash(q2)
            
            q1_idf = idf_dict.get(q1,0)
            q2_idf = idf_dict.get(q2,0)
            
            q_min = min(q1_idf,q2_idf)
            q_max = max(q1_idf,q2_idf)
            q_mean = 0.5*(q1_idf+q2_idf)
            q_diff = q_max-q_min
            
            
            outfile.write('%s,%s,%s,%s\n' % (
                q_min,
                q_max,
                q_mean,
                q_diff,   
                ))
            
            c+=1
            end = datetime.now()


    print 'times:',end-start


prepare_hash_clique_total(path+'train_hashed.csv',path+'train_hashed_clique_total.csv',max_clique)

prepare_hash_clique_total(path+'test_hashed.csv',path+'test_hashed_clique_total.csv',max_clique)
