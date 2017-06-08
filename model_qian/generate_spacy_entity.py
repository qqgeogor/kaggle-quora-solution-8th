from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
import string
import numpy as np
import spacy
from config import path
from feat_utils import get_jaccard
nlp = spacy.load("en_core_web_md")
# path = '../input/'

import string


string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

# print model.vocab
def remove_punctuation(x):
    new_line = [ w for w in list(x) if w not in string.punctuation]
    tmp = []
    for u in new_line:
        try:
            tmp.append(unicode(u))
        except:
            tmp.append("unicode_%s"%hash(u))
            
    new_line = ''.join(tmp)
    return new_line

def distinct_terms(lst1, lst2):
    lst1 = lst1.split(" ")
    lst2 = lst2.split(" ")
    common = set(lst1).intersection(set(lst2))
    new_lst1 = ' '.join([w for w in lst1 if w not in common])
    new_lst2 = ' '.join([w for w in lst2 if w not in common])
    
    return (new_lst1,new_lst2)


def prepare_distinct(path,out,nlp):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        columns = [
            'ent_max',
            'ent_min',
            'ent_diff',
            'ent_jaccard',
        ]
        columns = ','.join(columns)
        outfile.write(columns+'\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = unicode(remove_punctuation(str(row['question1']).lower()))
            q2 = unicode(remove_punctuation(str(row['question2']).lower()))
            # spacy_sim = nlp(q1).similarity(nlp(q2))
            # print q1,q2
            # q1,q2 = distinct_terms(q1,q2)
            # spacy_sim_distinct = nlp(unicode(q1)).similarity(nlp(unicode(q2)))
            q1 = nlp(q1)
            q2 = nlp(q2)
            # q1_ent = [ent.label_ for ent in q1.ents]
            # q2_ent = [ent.label_ for ent in q2.ents]

            q1_ent = [ent.text for ent in q1.ents]
            q2_ent = [ent.text for ent in q2.ents]
            
            q1_len = len(q1_ent)
            q2_len = len(q2_ent)
            ent_max = max(q1_len,q2_len)
            ent_min = min(q1_len,q2_len)
            ent_diff = ent_max-ent_min
            # print(q1,q2)
            # print(q1_ent,q2_ent)
            try:
                ent_jaccard = get_jaccard(q1_ent,q2_ent)
            except:
                ent_jaccard = -1
            
            features = (
                ent_max,
                ent_min,
                ent_diff,
                ent_jaccard,
            )
            outfile.write('%s,%s,%s,%s\n' % features)
            c+=1
        end = datetime.now()
    print 'times:',end-start

prepare_distinct(path+'train.csv',path+'train_spacy_entity.csv',nlp)
prepare_distinct(path+'test.csv',path+'test_spacy_entity.csv',nlp)
