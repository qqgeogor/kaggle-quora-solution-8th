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

path = "F:\\Quora\\"


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

def prepare_idf_dict(paths,smooth=1.0):
    idf_dict = dict()
    for path in paths:
        print path
        c = 0
        start = datetime.now()

        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = remove_punctuation(str(row['question1_hash']).lower())#.split(' ')
            q2 = remove_punctuation(str(row['question2_hash']).lower()).lower()#.split(' ')
            q1 = str(hash(q1))
            q2 = str(hash(q2))
            for key in [q1,q2]:
                df = idf_dict.get(key,0)
                df+=1
                idf_dict[key]=df
            c+=1
        end = datetime.now()

    n = c*2
    for key in idf_dict:
        idf_dict[key] = get_idf(idf_dict[key] ,n,smooth=smooth)
    idf_dict["default_idf"] = get_idf(0 ,n,smooth=smooth)

    print 'times:',end-start
    return idf_dict

def get_idf(df,n,smooth=1):
    idf = log((smooth + n) / (smooth + df))
    return idf

def prepare_hash_idf(path,out,idf_dict):

    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_hash_count,question2_hash_count\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = remove_punctuation(str(row['question1_hash']).lower())#.split(' ')
            q2 = remove_punctuation(str(row['question2_hash']).lower()).lower()#.split(' ')
            q1 = str(hash(q1))
            q2 = str(hash(q2))

            q1_idf = idf_dict.get(q1,idf_dict['default_idf'])
            q2_idf = idf_dict.get(q2,idf_dict['default_idf'])

            outfile.write('%s,%s\n' % (q1_idf, q2_idf))

            c+=1
            end = datetime.now()


    print 'times:',end-start



idf_dict = prepare_idf_dict([path+'train_hashed.csv',path+'test_hashed.csv'])
prepare_hash_idf(path+'train_hashed.csv',path+'train_hashed_idf.csv',idf_dict)
prepare_hash_idf(path+'test_hashed.csv',path+'test_hashed_idf.csv',idf_dict)
