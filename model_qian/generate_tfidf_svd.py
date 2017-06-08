import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from sklearn.decomposition import NMF,TruncatedSVD
# from tsne import bh_sne
seed = 1024
np.random.seed(seed)
from config import path

feats= ['question1_unigram','question2_unigram','question1_bigram','question2_bigram','question1_distinct_unigram','question2_distinct_unigram','question1_distinct_bigram','question2_distinct_bigram','question1_unigram_question2_unigram','question1_distinct_unigram_question2_distinct_unigram']
for f in feats:
    print f
    X = pd.read_pickle(path+'train_%s_tfidf.pkl'%f)
    len_train = X.shape[0]
    X_t = pd.read_pickle(path+'test_%s_tfidf.pkl'%f)
    data_all = ssp.vstack([X,X_t])
    data_all = TruncatedSVD(n_components=16,random_state=seed).fit_transform(data_all)
    del X
    del X_t
    # data_all = bh_sne(data_all)
    X = data_all[:len_train]
    X_t = data_all[len_train:]
    del data_all
    pd.to_pickle(X,path+'train_%s_tfidf_svd.pkl'%f)
    pd.to_pickle(X_t,path+'test_%s_tfidf_svd.pkl'%f)
    