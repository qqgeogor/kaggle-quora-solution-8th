import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime,timedelta
import jieba
from gensim import corpora, models, similarities
seed = 1024
np.random.seed(seed)
from config import path

features = ['question1','question2']
train = pd.read_csv(path+'train_porter.csv')[features].astype(str)
test = pd.read_csv(path+'test_porter.csv')[features].astype(str)
data_all = pd.concat([train,test])

docs = data_all['question1'].values.tolist()+data_all['question2'].values.tolist()


docs = [d.split(' ') for d in docs]
dic = corpora.Dictionary(docs)
# pd.to_pickle(dic,path+'dict.pkl')
# dic = pd.read_pickle(path+'dict.pkl')

corpus_search = [dic.doc2bow(doc.split(' ')) for doc in data_all['question1'].values.tolist()]
corpus_title = [dic.doc2bow(doc.split(' ')) for doc in data_all['question2'].values.tolist()]

tfidf = models.TfidfModel(corpus_search+corpus_title)
# pd.to_pickle(tfidf,path+'tfidf.pkl')
# tfidf = pd.read_pickle(path+'tfidf.pkl')
vec_len = len(dic.keys())

from datetime import datetime
start = datetime.now()
sims = []
print 'start tfidf'
count = 0
for cq,ct in zip(corpus_search,corpus_title):

    if count%10000==0:
        print 'count: ',count
    index_t = similarities.SparseMatrixSimilarity([ct], num_features=vec_len)
    sim_qt = index_t[tfidf[cq]]
    sims.append(sim_qt[0])
    count+=1

end = datetime.now()
print 'total time: ',end-start

sims = np.array(sims)

train_gensim_tfidf_sim = sims[:train.shape[0]]
test_gensim_tfidf_sim = sims[train.shape[0]:]

pd.to_pickle(train_gensim_tfidf_sim,path+'train_gensim_tfidf_sim.pkl')
pd.to_pickle(test_gensim_tfidf_sim,path+'test_gensim_tfidf_sim.pkl')
