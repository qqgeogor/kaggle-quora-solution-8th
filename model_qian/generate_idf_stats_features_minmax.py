#############################################################################################################
# Created by qqgeogor
# https://www.kaggle.com/qqgeogor
#############################################################################################################

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
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

print("Generate question idf stats features")
idf_dict = pickle.load(open(path+'idf_dict.pkl','rb'))

def mean(x):
    return sum(x)/float(len(x))

def median(x):
    len_2 = len(x)/2
    return x[len_2]

def std(x):
    mean_x = mean(x)
    s = 0.0
    for xx in x:
        s+=(xx-mean_x)**2
    s/=len(x)
    s = sqrt(s)
    return s


def create_idf_stats_features(path,idf_dict,out):
    K_dict = dict()
    print path
    c = 0
    start = datetime.now()
    sentences = []
    with open(out, 'w') as outfile:
        outfile.write('min_q1_idfs,max_q1_idfs,mean_q1_idfs,median_q1_idfs,std_q1_idfs,min_q2_idfs,max_q2_idfs,mean_q2_idfs,median_q2_idfs,std_q2_idfs\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = remove_punctuation(str(row['question1'])).lower()
            q2 = remove_punctuation(str(row['question2'])).lower()
            
            q1_idfs = [idf_dict.get(key,idf_dict['default_idf']) for key in q1.split(" ")]
            q2_idfs = [idf_dict.get(key,idf_dict['default_idf']) for key in q2.split(" ")]

            min_q1_idfs = min(q1_idfs)
            max_q1_idfs = max(q1_idfs)
            mean_q1_idfs = mean(q1_idfs)
            median_q1_idfs = median(q1_idfs)
            std_q1_idfs = std(q1_idfs)

            min_q2_idfs = min(q2_idfs)
            max_q2_idfs = max(q2_idfs)
            mean_q2_idfs = mean(q2_idfs)
            median_q2_idfs = median(q2_idfs)
            std_q2_idfs = std(q2_idfs)

            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                min_q1_idfs,
                max_q1_idfs,
                mean_q1_idfs,
                median_q1_idfs,
                std_q1_idfs,
                min_q2_idfs,
                max_q2_idfs,
                mean_q2_idfs,
                median_q2_idfs,
                std_q2_idfs
                ))
            c+=1
        end = datetime.now()
        print 'times:',end-start
create_idf_stats_features(path+'train.csv',idf_dict,path+'train_idf_stats_minmax_features.csv')
create_idf_stats_features(path+'test.csv',idf_dict,path+'test_idf_stats_minmax_features.csv')

print("End")

