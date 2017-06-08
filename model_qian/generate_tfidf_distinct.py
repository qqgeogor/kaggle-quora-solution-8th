import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 1024
np.random.seed(seed)
from config import path

ft = ['question1_distinct_unigram','question2_distinct_unigram']
train = pd.read_csv(path+"train_distinct_unigram.csv")[ft]

test = pd.read_csv(path+"test_distinct_unigram.csv")[ft]

len_train = train.shape[0]

max_features = None
ngram_range = (1,1)
min_df = 3
print('Generate tfidf')
feats= ['question1_distinct_unigram','question2_distinct_unigram']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus = []
for f in feats:
    train[f] = train[f].astype(str)
    test[f] = test[f].astype(str)
    corpus+=train[f].values.tolist()

vect_orig.fit(
    corpus
    )

for f in feats:
    train_tfidf = vect_orig.transform(train[f].values.tolist())
    test_tfidf = vect_orig.transform(test[f].values.tolist())

    pd.to_pickle(train_tfidf,path+'train_%s_tfidf.pkl'%f)
    pd.to_pickle(test_tfidf,path+'test_%s_tfidf.pkl'%f)

