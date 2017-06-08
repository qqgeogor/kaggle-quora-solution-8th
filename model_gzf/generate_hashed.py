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


path = "../data/"


string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

def remove_punctuation(x):
    new_line = [ w for w in list(x) if w not in string.punctuation]
    new_line = ''.join(new_line)
    return new_line

def prepare_hash(path,out):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_hash,question2_hash\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = remove_punctuation(str(row['question1_porter']).lower())#.split(' ')
            q2 = remove_punctuation(str(row['question2_porter']).lower()).lower()#.split(' ')
            q1 = str(hash(q1))
            q2 = str(hash(q2))
            outfile.write('%s,%s\n' % (q1, q2))


            c+=1
        end = datetime.now()


    print 'times:',end-start

prepare_hash(path+'train_porter.csv',path+'train_hashed.csv')
prepare_hash(path+'test_porter.csv',path+'test_hashed.csv')
