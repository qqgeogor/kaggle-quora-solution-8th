from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
import string
import numpy as np


from config import path

import string


string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

from gensim.models import Word2Vec
# model = Word2Vec.load_word2vec_format(path+'GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
# print model.vocab

def read_emb(path):
    count=0
    f = open(path,'r')
    emb_dict = dict()
    for line in f:
        # if count==0:
        #     count+=1
        #     continue
        line = line.strip().split(' ')
        id = line[0]
        
        weights = line[1:]
        weights = np.array([float(i) for i in weights])
        count+=1
        emb_dict[id] = weights
    return emb_dict
# data_all = pd.concat([train,test])
# model = Word2Vec.load_word2vec_format(path+'glove.840B.300d.txt', binary=False)  # C binary format
model = read_emb(path+'glove.840B.300d.txt')


def remove_punctuation(x):
    new_line = [ w for w in list(x) if w not in string.punctuation]
    new_line = ''.join(new_line)
    return new_line

def distinct_terms(lst1, lst2):
    lst1 = lst1.split(" ")
    lst2 = lst2.split(" ")
    common = set(lst1).intersection(set(lst2))
    new_lst1 = ' '.join([w for w in lst1 if w not in common])
    new_lst2 = ' '.join([w for w in lst2 if w not in common])
    
    return (new_lst1,new_lst2)

from  scipy.spatial.distance import cosine
def prepare_distinct(path,out,embedder):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        columns = [
            'w2v_sim_mean',
            'w2v_sim_max',
            'w2v_sim_min',
            'w2v_sim_std',
            'w2v_dist_mean',
            'w2v_dist_max',
            'w2v_dist_min',
            'w2v_dist_std',
        ]
        columns = ','.join(columns)
        outfile.write(columns+'\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = remove_punctuation(str(row['question1']).lower())
            q2 = remove_punctuation(str(row['question2']).lower())
            # print q1,q2
            q1,q2 = distinct_terms(q1,q2)
            # print q1,"_______",q2
            a2 = [x for x in q1.split(' ') if x in embedder]
            b2 = [x for x in q2.split(' ') if x in embedder]
            # print a2,b2

            sims = []
            dists = []
            if len(a2)==0 or len(b2)==0:
                sims=[0.0]
                dists=[0.0]
            else:
                for i in range(len(a2)):
                    for j in range(len(b2)):
                        try:
                            worda = a2[i]
                            wordb = b2[j]
                            if worda=="" or wordb=="":
                                continue
                            # sim = embedder.n_similarity(worda, wordb)
                            sim = cosine(embedder[worda],embedder[wordb])
                            vector_diff = embedder[worda]-embedder[wordb]
                            dist = np.sqrt(np.sum(vector_diff**2))

                            sims.append(sim)
                            dists.append(dist)
                        except Exception,e:
                            # print e
                            continue
            if len(sims)==0 or len(dists)==0:
                sims=[0.0]
                dists=[0.0]

            w2v_sim_mean = np.mean(sims)
            w2v_sim_max = np.max(sims)
            w2v_sim_min = np.min(sims)
            w2v_sim_std = np.std(sims)

            w2v_dist_mean = np.mean(dists)
            w2v_dist_max = np.max(dists)
            w2v_dist_min = np.min(dists)
            w2v_dist_std = np.std(dists)
            features = (
                w2v_sim_mean,
                w2v_sim_max,
                w2v_sim_min,
                w2v_sim_std,
                w2v_dist_mean,
                w2v_dist_max,
                w2v_dist_min,
                w2v_dist_std,
            )
            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % features)
            c+=1
        end = datetime.now()
    print 'times:',end-start

prepare_distinct(path+'train.csv',path+'train_distinct_word_stats_pretrained_glove.csv',model)
prepare_distinct(path+'test.csv',path+'test_distinct_word_stats_pretrained_glove.csv',model)
