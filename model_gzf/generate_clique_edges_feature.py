# coding: utf-8
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from csv import DictReader
#import matplotlib.pyplot as plt


path='data/'



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
    for item in clique:
        max_clique[item]=max(max_clique.get(item,0),len_clique)   
    cnt+=1


pd.to_pickle(max_clique,'data/max_clique_edges.pkl')

