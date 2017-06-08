import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.decomposition import PCA,NMF,SparsePCA,TruncatedSVD
from .utils import split_data
import scipy.stats as sps


np.random.rand(1024)

path = '../X_v2/'

q1_indicator = pd.read_pickle(path+'uni_gram_q1.pkl')
q2_indicator = pd.read_pickle(path+'uni_gram_q2.pkl')
train = pd.read_csv('../data/train.csv')

indicator_all = scipy.sparse.hstack([q1_indicator,q2_indicator]).T

pca = TruncatedSVD(n_components=12,random_state=1123)#(samples,features)
pca.fit(indicator_all)
pca_fea = pca.components_.T


#nmf
nmf = NMF(n_components=12,random_state=1123)
nmf.fit(indicator_all)
nmf_fea = nmf.components_.T

train_pca = pca_fea[:train.shape[0]]
test_pca = pca_fea[train.shape[0]:]
train_nmf = nmf_fea[:train.shape[0]]
test_nmf = nmf_fea[train.shape[0]:]

train_fea = np.hstack([train_pca,train_nmf])
test_fea = np.hstack([test_pca,test_nmf])
test_x = split_data.split_test(test_fea)

pd.to_pickle(train_fea,'../X_v2/train_uni_ind.pkl')
for i in range(6):
    pd.to_pickle(test_x[i],'../X_v2/test_uni_ind{0}.pkl'.format(i))

