
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
path='../data/'


# In[16]:

def get_distinct_q1(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1_porter']).lower().split():
        q1words[word] = 1
    for word in str(row['question2_porter']).lower().split():
        q2words[word] = 1
    shared_words_in_q1 = [w for w in q1words.keys() if w not in q2words]
    #shared_words_in_q2 = [w for w in q2words.keys() if w not in q1words]
    if len(shared_words_in_q1)==0:
        shared_words_in_q1=['#']
    #if len(shared_words_in_q2)==0:
    #    shared_words_in_q2=['#']
    #R = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))
    return ' '.join(shared_words_in_q1)

def get_distinct_q2(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1_porter']).lower().split():
        q1words[word] = 1
    for word in str(row['question2_porter']).lower().split():
        q2words[word] = 1
    #shared_words_in_q1 = [w for w in q1words.keys() if w not in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w not in q1words]
    #if len(shared_words_in_q1)==0:
    #    shared_words_in_q1=['#']
    if len(shared_words_in_q2)==0:
        shared_words_in_q2=['#']
    #R = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))
    return ' '.join(shared_words_in_q2)


# In[19]:

train=pd.read_csv(path+'train_porter.csv',index_col=0)
test=pd.read_csv(path+'test_porter.csv',index_col=0)


# In[20]:

train['distinct_porter_q1']=train.apply(get_distinct_q1,axis=1)
train['distinct_porter_q2']=train.apply(get_distinct_q2,axis=1)
test['distinct_porter_q1']=test.apply(get_distinct_q1,axis=1)
test['distinct_porter_q2']=test.apply(get_distinct_q2,axis=1)


# In[21]:

train[['distinct_porter_q1','distinct_porter_q2']].to_csv(path+'train_distinct.csv',index=False)
test[['distinct_porter_q1','distinct_porter_q2']].to_csv(path+'test_distinct.csv',index=False)


# In[22]:

corpus=[]
for fe in ['distinct_porter_q1','distinct_porter_q2']:
    corpus+=train[fe].astype(str).values.tolist()
    #corpus+=test[fe].astype(str).values.tolist()


# In[27]:

from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
from sklearn.preprocessing import OneHotEncoder


# In[29]:
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 3
tfidf=TfidfVectorizer(ngram_range=(1,1),stop_words=stopwords, min_df=tfidf__min_df,max_df=tfidf__max_df,norm=tfidf__norm)
tfidf.fit(corpus)


# In[32]:

pd.to_pickle(tfidf.transform(train.distinct_porter_q1),path+'train_distinct_porter_q1_tfidf.pkl')
pd.to_pickle(tfidf.transform(train.distinct_porter_q2),path+'train_distinct_porter_q2_tfidf.pkl')
pd.to_pickle(tfidf.transform(test.distinct_porter_q1),path+'test_distinct_porter_q1_tfidf.pkl')
pd.to_pickle(tfidf.transform(test.distinct_porter_q2),path+'test_distinct_porter_q2_tfidf.pkl')





