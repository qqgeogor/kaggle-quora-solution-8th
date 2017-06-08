import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 1024
np.random.seed(seed)

from config import path

train = pd.read_csv(path+"train_hashed.csv")
test = pd.read_csv(path+"test_hashed.csv")
neighbour = pd.read_csv(path+"neighbour.csv")

len_train = train.shape[0]

max_features = None
ngram_range = (1,1)
min_df = 1
print('Generate tfidf')

vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)
feats= ['question1_hash','question2_hash']
corpus = []
for f in feats:
    train[f+'_ids'] = pd.merge(train,neighbour,left_on= f,right_on='question')['ids']
    test[f+'_ids'] = pd.merge(test,neighbour,left_on= f,right_on='question')['ids']
    corpus+=train[f+'_ids'].values.tolist()
    corpus+=test[f+'_ids'].values.tolist()


vect_orig.fit(
    corpus
    )
train_tfidf = []
test_tfidf = []
for f in feats:
    train_tfidf.append(vect_orig.transform(train[f+'_ids'].values.tolist()))
    test_tfidf.append(vect_orig.transform(test[f+'_ids'].values.tolist()))

train_tfidf = train_tfidf[0]+train_tfidf[1]
test_tfidf = test_tfidf[0]+test_tfidf[1]

print train_tfidf.shape,test_tfidf.shape
pd.to_pickle(train_tfidf,path+'train_question_hash_tfidf.pkl')
pd.to_pickle(test_tfidf,path+'test_question_hash_tfidf.pkl')
