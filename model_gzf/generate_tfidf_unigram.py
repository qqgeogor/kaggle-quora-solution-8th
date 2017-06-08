import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
seed = 1024
np.random.seed(seed)
path = "../input/"

ft = ['question1_unigram','question2_unigram']
train = pd.read_csv(path+"train_unigram.csv")[ft]
test = pd.read_csv(path+"test_unigram.csv")[ft]

len_train = train.shape[0]

print('Generate tfidf')
feats= ['question1_unigram','question2_unigram']
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 3
vect_orig=TfidfVectorizer(ngram_range=(1,1),stop_words=stopwords,
                          min_df=tfidf__min_df,max_df=tfidf__max_df,norm=tfidf__norm)


corpus = []
for f in feats:
    train[f] = train[f].astype(str)
    test[f]=test[f].astype(str)
    corpus+=train[f].values.tolist()

vect_orig.fit(
    corpus
    )

data_all=pd.concat([train,test],axis=0)

for f in feats:
    tfidfs = vect_orig.transform(data_all[f].values.tolist())
    train_tfidf = tfidfs[:train.shape[0]]
    test_tfidf = tfidfs[train.shape[0]:]
    pd.to_pickle(train_tfidf,path+'train_%s_tfidf.pkl'%f)
    pd.to_pickle(test_tfidf,path+'test_%s_tfidf.pkl'%f)
