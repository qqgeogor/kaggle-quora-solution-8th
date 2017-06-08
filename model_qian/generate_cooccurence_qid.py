from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
import string



from config import path


def cooccurrence_terms(lst1, lst2, join_str="__"):
    lst1 = lst1.split(" ")
    lst2 = lst2.split(" ")
    terms = [""] * len(lst1) * len(lst2)
    cnt =  0
    for item1 in lst1:
        for item2 in lst2:
            terms[cnt] = item1 + join_str + item2
            cnt += 1
    res = " ".join(terms)
    return res

def prepare_cooccurrence(path1,path2,out):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_unigram_question2_hash,question1_hash_question2_unigram\n')
        for row1, row2 in zip(DictReader(open(path1), delimiter=','),DictReader(open(path2), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1_unigram = str(row1['question1_unigram'])
            q2_unigram = str(row1['question2_unigram'])

            q1_hash = str(row2['question1_hash'])
            q2_hash = str(row2['question2_hash'])

            coo_terms1 = cooccurrence_terms(q1_unigram,q2_hash)
            coo_terms2 = cooccurrence_terms(q2_unigram,q1_hash)
            outfile.write('%s,%s\n' % (coo_terms1,coo_terms2))
            c+=1
        end = datetime.now()
    print 'times:',end-start

prepare_cooccurrence(path+'train_unigram.csv',path+'train_hashed.csv',path+'train_cooccurrence_qid.csv')
prepare_cooccurrence(path+'test_unigram.csv',path+'test_hashed.csv',path+'test_cooccurrence_qid.csv')
