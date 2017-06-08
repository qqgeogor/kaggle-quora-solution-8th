
# coding: utf-8

# In[14]:

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD,NMF


# In[15]:

def svd(train,test,dims=6,it=15,file_name='tf_idf',path='data/'):
    svd=NMF(random_state=1123,n_components=dims)
    svd.fit(train)
    #print svd.transform(train).shape
    pd.to_pickle(svd.transform(train),path+'train_NMF_'+str(dims)+'_'+file_name+'.pkl')
    pd.to_pickle(svd.transform(test),path+'test_NMF_'+str(dims)+'_'+file_name+'.pkl')
    return 'Success'


# In[16]:

tf_idf_co_train=pd.read_pickle('data/train_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_dis_co_train=pd.read_pickle('data/train_distinct_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_co_test=pd.read_pickle('data/test_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_dis_co_test=pd.read_pickle('data/test_distinct_question1_unigram_question2_unigram_tfidf.pkl')


# In[17]:

svd(tf_idf_co_train,tf_idf_co_test,file_name='question1_unigram_question2_unigram_tfidf')
svd(tf_idf_dis_co_train,tf_idf_dis_co_test,file_name='distinct_question1_unigram_question2_unigram_tfidf')


# In[18]:

tf_idf_q1_unigram_train=pd.read_pickle('data/train_question1_unigram_tfidf.pkl')
tf_idf_q2_unigram_train=pd.read_pickle('data/train_question2_unigram_tfidf.pkl')
tf_idf_q1_unigram_test=pd.read_pickle('data/test_question1_unigram_tfidf.pkl')
tf_idf_q2_unigram_test=pd.read_pickle('data/test_question2_unigram_tfidf.pkl')


# In[19]:

svd(tf_idf_q1_unigram_train,tf_idf_q1_unigram_test,file_name='question1_unigram_tfidf')
svd(tf_idf_q2_unigram_train,tf_idf_q2_unigram_test,file_name='question2_unigram_tfidf')


# In[20]:

tf_idf_q1_bigram_train=pd.read_pickle('data/train_question1_bigram_tfidf.pkl')
tf_idf_q2_bigram_train=pd.read_pickle('data/train_question2_bigram_tfidf.pkl')
tf_idf_q1_bigram_test=pd.read_pickle('data/test_question1_bigram_tfidf.pkl')
tf_idf_q2_bigram_test=pd.read_pickle('data/test_question2_bigram_tfidf.pkl')


# In[21]:

svd(tf_idf_q1_bigram_train,tf_idf_q1_bigram_test,file_name='question1_bigram_tfidf')
svd(tf_idf_q2_bigram_train,tf_idf_q2_bigram_test,file_name='question2_bigram_tfidf')

