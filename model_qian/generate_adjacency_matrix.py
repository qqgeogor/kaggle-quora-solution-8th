# coding: utf-8

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from csv import DictReader
from sklearn.decomposition import NMF
from config import path
seed=1024
np.random.seed(seed)

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

A = nx.adjacency_matrix(G)
nmf = NMF(n_components=2,random_state=seed)
print(A.shape)
A = nmf.fit_transform(A)
print(A.shape)
pd.to_pickle(A,path+'adjacency_matrix_nmf.pkl')
del A
A = nx.incidence_matrix(G)
nmf = NMF(n_components=2,random_state=seed)
print(A.shape)
A = nmf.fit_transform(A)
print(A.shape)
pd.to_pickle(A,path+'incidence_matrix_nmf.pkl')
del A

nodes = G.nodes()
adjacency_matrix_nmf = pd.read_pickle(path+'adjacency_matrix_nmf.pkl')
incidence_matrix_nmf = pd.read_pickle(path+'incidence_matrix_nmf.pkl')
d = dict()
for n,a,i in zip(nodes,adjacency_matrix_nmf,incidence_matrix_nmf):
    # print(n,a,i)
    d[n] = np.concatenate([a,i])

pd.to_pickle(d,path+'graph_decom.pkl')

train = pd.read_csv(path+'train_hashed.csv').astype(str)
test = pd.read_csv(path+'test_hashed.csv').astype(str)
train_q1_decom = np.vstack(train['question1_hash'].apply(lambda x:d[x]).values.tolist())
print(train_q1_decom.shape)
train_q2_decom = np.vstack(train['question2_hash'].apply(lambda x:d[x]).values.tolist())
test_q1_decom = np.vstack(test['question1_hash'].apply(lambda x:d[x]).values.tolist())
test_q2_decom = np.vstack(test['question2_hash'].apply(lambda x:d[x]).values.tolist())

pd.to_pickle(train_q1_decom,path+'train_q1_decom.pkl')
pd.to_pickle(train_q2_decom,path+'train_q2_decom.pkl')
pd.to_pickle(test_q1_decom,path+'test_q1_decom.pkl')
pd.to_pickle(test_q2_decom,path+'test_q2_decom.pkl')
