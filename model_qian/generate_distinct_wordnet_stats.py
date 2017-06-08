from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
import string
import numpy as np
from sematch.semantic.similarity import WordNetSimilarity
from config import path
wns = WordNetSimilarity()

import string


string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

# from gensim.models import Word2Vec
# model = Word2Vec.load_word2vec_format(path+'GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
# print model.vocab
model = None
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
            # 'w2v_dist_mean',
            # 'w2v_dist_max',
            # 'w2v_dist_min',
            # 'w2v_dist_std',
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
            a2 = [x for x in q1.split(' ')]
            b2 = [x for x in q2.split(' ')]
            # print a2,b2

            sims = []
            dists = []
            if len(a2)==0 or len(b2)==0:
                sims=[0.0]
            else:
                for i in range(len(a2)):
                    for j in range(len(b2)):
                        try:
                            try:
                                worda = unicode(a2[i])
                            except:
                                worda = u"unknowa"

                            try:
                                wordb = unicode(b2[i])
                            except:
                                wordb = u"unknowb"
                            if worda=="" or wordb=="":
                                continue
                            # sim = embedder.n_similarity(worda, wordb)
                            # vector_diff = embedder[worda]-embedder[wordb]
                            # dist = np.sqrt(np.sum(vector_diff**2))
                            sim = wns.word_similarity(worda, wordb, 'li')
                            sims.append(sim)
                            # dists.append(dist)
                        except Exception,e:
                            print e
                            continue
            if len(sims)==0:
                sims=[0.0]

            w2v_sim_mean = np.mean(sims)
            w2v_sim_max = np.max(sims)
            w2v_sim_min = np.min(sims)
            w2v_sim_std = np.std(sims)

            # w2v_dist_mean = np.mean(dists)
            # w2v_dist_max = np.max(dists)
            # w2v_dist_min = np.min(dists)
            # w2v_dist_std = np.std(dists)
            features = (
                w2v_sim_mean,
                w2v_sim_max,
                w2v_sim_min,
                w2v_sim_std,
                # w2v_dist_mean,
                # w2v_dist_max,
                # w2v_dist_min,
                # w2v_dist_std,
            )
            outfile.write('%s,%s,%s,%s\n' % features)
            c+=1
        end = datetime.now()
    print 'times:',end-start

prepare_distinct(path+'train.csv',path+'train_distinct_wordnet_stats.csv',model)
prepare_distinct(path+'test.csv',path+'test_distinct_wordnet_stats.csv',model)
