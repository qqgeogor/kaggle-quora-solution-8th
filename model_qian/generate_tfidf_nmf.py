import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from sklearn.decomposition import NMF
seed = 1024
np.random.seed(seed)
from config import path


feats= ['question1_unigram','question2_unigram','question1_bigram','question2_bigram','question1_distinct_unigram','question2_distinct_unigram','question1_distinct_bigram','question2_distinct_bigram','question1_unigram_question2_unigram','question1_distinct_unigram_question2_distinct_unigram']
for f in feats:
    print f
    X = pd.read_pickle(path+'train_%s_tfidf.pkl'%f)
    nmf = NMF(n_components=4,random_state=seed)
    X = nmf.fit_transform(X)
    pd.to_pickle(X,path+'train_%s_tfidf_nmf.pkl'%f)
    del X
    X_t = pd.read_pickle(path+'test_%s_tfidf.pkl'%f)
    X_t = nmf.transform(X_t)
    pd.to_pickle(X_t,path+'test_%s_tfidf_nmf.pkl'%f)
    del X_t
