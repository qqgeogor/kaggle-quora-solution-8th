from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
from ngram import getBigram
import string


path = '../input/'


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

def prepare_bigram(path,out):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_bigram,question2_bigram\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = remove_punctuation(str(row['question1_porter']).lower()).split(' ')
            q2 = remove_punctuation(str(row['question2_porter']).lower()).lower().split(' ')
            q1_bigram = getBigram(q1)
            q2_bigram = getBigram(q2)
            q1_bigram = ' '.join(q1_bigram)
            q2_bigram = ' '.join(q2_bigram)
            outfile.write('%s,%s\n' % (q1_bigram, q2_bigram))


            c+=1
        end = datetime.now()


    print 'times:',end-start

prepare_bigram(path+'train_porter.csv',path+'train_bigram.csv')

prepare_bigram(path+'test_porter.csv',path+'test_bigram.csv')
