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
            tmp = sorted([item1,item2])
            terms[cnt] = tmp[0] + join_str + tmp[1]
            cnt += 1
    return terms

# def common_terms(lst1, lst2):
#     lst1 = lst1.split(" ")
#     lst2 = lst2.split(" ")
#     common = set(lst1).intersection(set(lst2))
#     return common

def H(terms,count_dict,n):
    s = 0.0
    for word in terms:
        s+=-(count_dict[word]/n)*log(count_dict[word]/n)
    return s


def create_df_dict(path,smooth=1.0,inverse=False,ngram='unigram'):
    K_dict = dict()
    print path
    c = 0
    start = datetime.now()
    sentences = []
    
    for t, row in enumerate(DictReader(open(path), delimiter=',')): 
        if c%100000==0:
            print 'finished',c
        q1 = (str(row['question1_%s'%ngram])).lower()
        q2 = (str(row['question2_%s'%ngram])).lower()
        
        for sentence in [q1,q2]:
            for key in sentence.split(" "):
                df = K_dict.get(key,0)
                K_dict[key] = df+1
        c+=1

    K_dict["default_idf"] = 1
    end = datetime.now()
    print 'times:',end-start
    return K_dict

print("Generate question df dict")
df_dict_unigram = create_df_dict(path+"train_unigram.csv",inverse=False,ngram='unigram')
df_dict_bigram = create_df_dict(path+"train_bigram.csv",inverse=False,ngram='bigram')

def entropy(seq1,seq2,df_dict):
    n = (404302.0)*2
    HA = 0.0
    for w in seq1:
        df = df_dict.get(w,df_dict['default_idf'])
        # print df
        HA+=-(df/n)*log(df/n)

    HB = 0.0
    for w in seq2:
        df = df_dict.get(w,df_dict['default_idf'])
        # print df
        HB+=-(df/n)*log(df/n)

    return HA,HB


def prepare_entropy(path,out,ngram='unigram'):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('max_entropy,min_entropy\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_%s'%ngram]).split(' ')
            q2 = str(row['question2_%s'%ngram]).split(' ')
            if ngram=='unigram':
                df_dict=df_dict_unigram
            elif ngram=='bigram':
                df_dict = df_dict_bigram
            HA,HB = entropy(q1,q2,df_dict)
            max_H = max(HA,HB)
            min_H = min(HA,HB)

            outfile.write('%s,%s\n' % (max_H,min_H))
            c+=1
        end = datetime.now()
    print 'times:',end-start



prepare_entropy(path+'train_unigram.csv',path+'train_entropy_unigram.csv','unigram')
prepare_entropy(path+'test_unigram.csv',path+'test_entropy_unigram.csv','unigram')

prepare_entropy(path+'train_bigram.csv',path+'train_entropy_bigram.csv','bigram')
prepare_entropy(path+'test_bigram.csv',path+'test_entropy_bigram.csv','bigram')
