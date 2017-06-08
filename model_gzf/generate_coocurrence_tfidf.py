import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords
seed = 1024
np.random.seed(seed)
path = "..//data/"
ft = ['question1','question2']
train = pd.read_csv(path+"train_porter.csv")[ft]
test = pd.read_csv(path+"test_porter.csv")[ft]

# test['is_duplicated']=[-1]*test.shape[0]
########################

## Cooccurrence terms ##

########################

def cooccurrence_terms(lst1, lst2, join_str):
    lst1 = str(lst1).translate(None, string.punctuation).lower().strip().split()
    lst2 = str(lst2).translate(None, string.punctuation).lower().strip().split()
    terms = [""] * len(lst1) * len(lst2)
    cnt = 0
    for item1 in lst1:
        for item2 in lst2:
            terms[cnt] = item1 + join_str + item2
            cnt += 1
    res = " ".join(terms)
    return res
print 'generate train cooccurrence..'
train['cooccurrence_terms'] = train.apply(lambda x:cooccurrence_terms(x['question1'],x['question2'],"_"),axis=1)
print 'generate test cooccurrence..'
test['cooccurrence_terms'] = test.apply(lambda x:cooccurrence_terms(x['question1'],x['question2'],"_"),axis=1)

corpus = train['cooccurrence_terms'].values.tolist()
tfidf = TfidfVectorizer(max_features=None,ngram_range=(1,1))
tfidf.fit(corpus)
#print tfidf.transform(train.ix[0]['cooccurrence_terms']).values.tolist()
print 'save...'
train_cooccurrence_tfidf = tfidf.transform(train['cooccurrence_terms'].values.tolist())
pd.to_pickle(train_cooccurrence_tfidf,path+"train_cooccurrence_tfidf.pkl")
del train_cooccurrence_tfidf

test_cooccurrence_tfidf = tfidf.transform(test['cooccurrence_terms'].values.tolist())
pd.to_pickle(test_cooccurrence_tfidf,path+"test_cooccurrence_tfidf.pkl")
del test_cooccurrence_tfidf