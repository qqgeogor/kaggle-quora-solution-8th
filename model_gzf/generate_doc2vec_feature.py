import pandas as pd
import numpy as np
import os
import string
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

f1 = open('doc2vec/sentence_vectors_q1.txt','r')
f2 = open('doc2vec/sentence_vectors_q2.txt','r')
_cosine=[]
_cityblock=[]
_jaccard=[]
_canberra=[]
_euclidean=[]
_minkowski=[]
_braycurtis=[]
for q1 in f1:
    q2 = f2.readline()
    v1=np.array(q1.strip().split()[1:]).astype(float)
    v2=np.array(q2.strip().split()[1:]).astype(float)
    #cos=np.dot(v1,v2)/np.sqrt((np.dot(v1,v1)*np.dot(v2,v2)))
    #abs_dis=np.sum(np.abs(v1-v2))
    _cosine.append(cosine(v1,v2))
    _cityblock.append(cityblock(v1,v2))
    _jaccard.append(jaccard(v1,v2))
    _canberra.append(canberra(v1,v2))
    _euclidean.append(euclidean(v1,v2))
    _minkowski.append(minkowski(v1,v2,3))
    _braycurtis.append(braycurtis(v1,v2))
f1.close()
f2.close()

y=pd.read_csv('data/train.csv')['is_duplicate'].values

doc_sim=np.vstack([_cosine,
           _cityblock,
           _jaccard,
           _canberra,
           _euclidean,
           _minkowski,
           _braycurtis]).T

pd.to_pickle(doc_sim[:y.shape[0]],'data/train_doc2vec_sim.pkl')
pd.to_pickle(doc_sim[y.shape[0]:],'data/test_doc2vec_sim.pkl')

