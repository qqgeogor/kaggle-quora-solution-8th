from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
from ngram import getUnigram
import string
import random
from config import path
import networkx as nx
seed =1024
random.seed(seed)


def prepare_graph(paths):
    G = nx.Graph()

    pr_dict = dict()
    for path in paths:
        print(path)
        c = 0
        start = datetime.now()

        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print('finished',c)
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            G.add_edge(q1,q2)
            c+=1
    end = datetime.now()
    print('times:',end-start)
    return G



def prepare_hash_pr(path,out,pr):

    print(path)
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_hash_pr,question2_hash_pr\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print('finished',c)
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            # q1 = hash(q1)
            # q2 = hash(q2)
            
            q1_pr = pr.get(q1,0)
            q2_pr = pr.get(q2,0)
            # print(q1_pr,q2_pr)
            outfile.write('%s,%s\n' % (q1_pr, q2_pr))
            
            c+=1
            end = datetime.now()


    print('times:',end-start)

G = prepare_graph([path+'train_hashed.csv',path+'test_hashed.csv'])
count=0
d_nodes= dict()
d_edges = dict()
for c in  nx.connected_component_subgraphs(G):
    l_nodes = len(c.nodes())
    l_edges = len(c.edges())
    for cc in c.nodes():
        d_nodes[cc] = l_nodes
        d_edges[cc] = l_edges
    count+=1
print('count of subgraph',count)

def prepare_hash_subgraph(path,out,idf_dict):

    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question_subgraph_max,question_subgraph_min,question_subgraph_diff,question_subgraph_mean\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            # q1 = hash(q1)
            # q2 = hash(q2)
            
            q1_idf = idf_dict.get(q1,0)
            q2_idf = idf_dict.get(q2,0)
            qmax = max(q1_idf,q2_idf)
            qmin = min(q1_idf,q2_idf)
            qdiff = qmax-qmin
            qmean = 0.5*(q1_idf+q2_idf)

            outfile.write('%s,%s,%s,%s\n' % (qmax,qmin,qdiff,qmean))
            
            c+=1
            end = datetime.now()


    print 'times:',end-start


prepare_hash_subgraph(path+'train_hashed.csv',path+'train_hashed_subgraph_d_nodes.csv',d_nodes)
prepare_hash_subgraph(path+'test_hashed.csv',path+'test_hashed_subgraph_d_nodes.csv',d_nodes)

prepare_hash_subgraph(path+'train_hashed.csv',path+'train_hashed_subgraph_d_edges.csv',d_edges)
prepare_hash_subgraph(path+'test_hashed.csv',path+'test_hashed_subgraph_d_edges.csv',d_edges)
