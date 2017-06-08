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
            q1 = str(row['question1_unigram'])
            q2 = str(row['question2_unigram'])
            G.add_edge(q1,q2)
            c+=1
    end = datetime.now()
    print('times:',end-start)
    return G



def prepared_unigram_pr(path,out,pr):

    print(path)
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_unigram_pr,question2_unigram_pr\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print('finished',c)
            q1 = str(row['question1_unigram'])
            q2 = str(row['question2_unigram'])
            # q1 = hash(q1)
            # q2 = hash(q2)
            
            q1_pr = pr.get(q1,0)
            q2_pr = pr.get(q2,0)
            # print(q1_pr,q2_pr)
            outfile.write('%s,%s\n' % (q1_pr, q2_pr))
            
            c+=1
            end = datetime.now()


    print('times:',end-start)

def prepared_unigram_subgraph(path,out,keywords):

    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('clique_topic\n')
        clique_topic = "default"
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_unigram'])
            q2 = str(row['question2_unigram'])


            q1 = q1.split(' ')
            q2 = q2.split(' ')
            common = set(q1).intersection(set(q2))
            for ele in keywords:
                if len(ele)>len(common) or len(ele)==0:
                    continue
                tmp = ele.intersection(common)
                if len(tmp)==len(ele):
                    clique_topic = '_'.join(list(ele))
                    # print clique_topic
                    break

            

            outfile.write('%s\n' % clique_topic)
            
            c+=1
            end = datetime.now()


    print 'times:',end-start


G = prepare_graph([
    path+'train_unigram.csv',
    path+'test_unigram.csv',
    ])

count=0
d_nodes= dict()
d_edges = dict()
keywords= []
for c in  nx.find_cliques(G):
    l_nodes = len(c)
    # l_edges = len(c.edges())
    if l_nodes<=2:
        continue
    start = True
    key = []
    for node in c:
        if start:
            key = set(node.split(' '))
            start=False
        else:
            key = set(node.split(' ')).intersection(key)
        if len(key)==0:
            break
    # print(key)
    if key not in keywords:
        keywords.append(key)
        count+=1
print('count of subgraph',count)
# pickle.dump(keywords,open(path+'keywords_clique.pkl','wb'))

# keywords = pickle.load(open(path+'keywords_clique.pkl','rb'))

prepared_unigram_subgraph(path+'train_unigram.csv',path+'train_unigram_clique_topic.csv',keywords)
prepared_unigram_subgraph(path+'test_unigram.csv',path+'test_unigram_clique_topic.csv',keywords)
