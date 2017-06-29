from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
from ngram import getUnigram
import string
import random
seed =1024
random.seed(seed)


path = "../input/"


def prepare_hash(path,out):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_hash,question2_hash\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1']).lower()
            q2 = str(row['question2']).lower()
            q1 = hash(q1)
            q2 = hash(q2)
            outfile.write('%s,%s\n' % (q1, q2))
            if c!=t:
                print c,t
            c+=1
            
        end = datetime.now()
        
    print(c)
    print('times:',end-start)

prepare_hash(path+'train.csv',path+'train_hashed.csv')
prepare_hash(path+'test.csv',path+'test_hashed.csv')
