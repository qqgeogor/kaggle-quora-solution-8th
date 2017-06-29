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
from config import path


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

def get_idf(df,n,smooth=1):
    idf = log((smooth + n) / (smooth + df))
    return idf



def create_idf_dict(path,smooth=1.0,inverse=False):
    K_dict = dict()
    print path
    c = 0
    start = datetime.now()
    sentences = []
    
    for t, row in enumerate(DictReader(open(path), delimiter=',')): 
        if c%100000==0:
            print 'finished',c
        q1 = remove_punctuation(str(row['question1'])).lower()
        q2 = remove_punctuation(str(row['question2'])).lower()
        
        for sentence in [q1,q2]:
            for key in sentence.split(" "):
                df = K_dict.get(key,0)
                K_dict[key] = df+1
        c+=1
    n = c*2
    for key in K_dict:
        K_dict[key] = get_idf(K_dict[key] ,n,smooth=smooth)
    K_dict["default_idf"] = get_idf(0 ,n,smooth=smooth)
    end = datetime.now()
    print 'times:',end-start
    return K_dict

print("Generate question idf dict")
idf_dict = create_idf_dict(path+"train.csv",inverse=False)

print("Dumping")
pickle.dump(idf_dict,open(path+'idf_dict.pkl','wb'))
print("End")

