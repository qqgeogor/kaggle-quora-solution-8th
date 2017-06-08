# coding: utf-8

import pandas as pd
import numpy as np
import gc
from MulticoreTSNE import MulticoreTSNE as TSNE

path='../input/'

def toTsne(train,test,n_component=2,file_name='tf_idf',path='data/'):
    tsne=TSNE(n_components=n_component,random_state=1123,njobs=-1)
    lentrain=train.shape[0]
    X=np.vstack([train,test])
    tsne.fit(X)
    res=tsne.embedding_
    #print res
    pd.to_pickle(res[:lentrain],path+'train_svd_20_tsne_'+str(n_component)+'_'+file_name+'.pkl')
    pd.to_pickle(res[lentrain:],path+'test_svd_20_tsne_'+str(n_component)+'_'+file_name+'.pkl')
    return 'Success'



svd100_tfidf_co_train=pd.read_pickle(path+'train_svd_20_question1_unigram_question2_unigram_tfidf.pkl')
svd100_tfidf_co_test=pd.read_pickle(path+'test_svd_20_question1_unigram_question2_unigram_tfidf.pkl')

toTsne(svd100_tfidf_co_train,svd100_tfidf_co_test,file_name='question1_unigram_question2_unigram')
del svd100_tfidf_co_train,svd100_tfidf_co_test
gc.collect()

svd100_dis_tfidf_co_train=pd.read_pickle(path+'train_svd_20_distinct_question1_unigram_question2_unigram_tfidf.pkl')
svd100_dis_tfidf_co_test=pd.read_pickle(path+'test_svd_20_distinct_question1_unigram_question2_unigram_tfidf.pkl')

toTsne(svd100_dis_tfidf_co_train,svd100_dis_tfidf_co_test,file_name='distinct_question1_unigram_question2_unigram')
del svd100_dis_tfidf_co_train,svd100_dis_tfidf_co_test
gc.collect()
print('success 1....')

############################################################
svd100_tfidf_q1_unigram_train=pd.read_pickle(path+'train_svd_20_question1_unigram_tfidf.pkl')
svd100_tfidf_q1_unigram_test=pd.read_pickle(path+'test_svd_20_question1_unigram_tfidf.pkl')

toTsne(svd100_tfidf_q1_unigram_train,svd100_tfidf_q1_unigram_test,file_name='question1_unigram')
del svd100_tfidf_q1_unigram_train,svd100_tfidf_q1_unigram_test
gc.collect()

svd100_tfidf_q2_unigram_train=pd.read_pickle(path+'train_svd_20_question2_unigram_tfidf.pkl')
svd100_tfidf_q2_unigram_test=pd.read_pickle(path+'test_svd_20_question2_unigram_tfidf.pkl')

toTsne(svd100_tfidf_q1_unigram_train,svd100_tfidf_q1_unigram_test,file_name='question1_unigram')
toTsne(svd100_tfidf_q2_unigram_train,svd100_tfidf_q2_unigram_test,file_name='question2_unigram')
del svd100_tfidf_q2_unigram_train,svd100_tfidf_q2_unigram_test
gc.collect()
################################################################################
print('success 2....')

svd100_tfidf_q1_bigram_train=pd.read_pickle(path+'train_svd_100_question1_bigram_tfidf.pkl')
svd100_tfidf_q1_bigram_test=pd.read_pickle(path+'test_svd_100_question1_bigram_tfidf.pkl')

toTsne(svd100_tfidf_q1_bigram_train,svd100_tfidf_q1_bigram_test,file_name='question1_bigram')
del svd100_tfidf_q1_bigram_train,svd100_tfidf_q1_bigram_test
gc.collect()

svd100_tfidf_q2_bigram_train=pd.read_pickle(path+'train_svd_100_question2_bigram_tfidf.pkl')
svd100_tfidf_q2_bigram_test=pd.read_pickle(path+'test_svd_100_question2_bigram_tfidf.pkl')

toTsne(svd100_tfidf_q1_bigram_train,svd100_tfidf_q1_bigram_test,file_name='question1_bigram')
toTsne(svd100_tfidf_q2_bigram_train,svd100_tfidf_q2_bigram_test,file_name='question2_bigram')
del svd100_tfidf_q2_bigram_train,svd100_tfidf_q2_bigram_test
gc.collect()
print('success 3....')

