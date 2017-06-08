from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
import string


path = '../input/'


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

def prepare_cooccurrence(path,out):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_distinct_bigram_question2_distinct_bigram\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_distinct_bigram'])
            q2 = str(row['question2_distinct_bigram'])
            coo_terms = cooccurrence_terms(q1,q2)
            outfile.write('%s\n' % coo_terms)
            c+=1
        end = datetime.now()
    print 'times:',end-start

prepare_cooccurrence(path+'train_distinct_bigram.csv',path+'train_cooccurrence_distinct_bigram.csv')
prepare_cooccurrence(path+'test_distinct_bigram.csv',path+'test_cooccurrence_distinct_bigram.csv')
