import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.decomposition import PCA,NMF,SparsePCA,TruncatedSVD
from .utils import split_data
import scipy.stats as sps


np.random.rand(1024)

path = '../X_v2/'

q1_indicator = pd.read_pickle(path+'uni_gram_q1_pos.pkl')
q2_indicator = pd.read_pickle(path+'uni_gram_q2_pos.pkl')
train = pd.read_csv('../data/train.csv')



# sps.spearmanr(q1_indicator.todense()[:train.shape[0]].sum(axis=1),train['is_duplicate'])[0]
#
q1_stats = np.hstack([q1_indicator.max(axis=1).todense(),q1_indicator.mean(axis=1),
                      q1_indicator.todense().std(axis=1)])

q2_stats = np.hstack([q2_indicator.max(axis=1).todense(),q2_indicator.mean(axis=1),
                    q2_indicator.todense().std(axis=1)])

stats = np.hstack([q1_stats,q2_stats])
train_ = stats[:train.shape[0]]
test_ = stats[train.shape[0]:]
pd.to_pickle(train_,'../X_v2/train_uni_pos_stats.pkl')
test_x = split_data.split_data(test_)
for i in range(6):
    pd.to_pickle(test_x[i],'../X_v2/test_uni_ind_pos{0}.pkl'.format(i))
indicator_all = np.hstack([q1_indicator,q2_indicator])

# pca = TruncatedSVD(n_components=12,random_state=1123)#(samples,features)
# pca.fit(indicator_all)
# pca_fea = pca.components_.T
#

#nmf
nmf = NMF(n_components=12,random_state=1123)
nmf.fit(indicator_all)
nmf_fea = nmf.components_.T

train_nmf = nmf_fea[:train.shape[0]]
test_nmf = nmf_fea[train.shape[0]:]

train_fea = np.hstack([train_nmf])
test_fea = np.hstack([test_nmf])
test_x = split_data.split_test(test_fea)

pd.to_pickle(train_fea,'../X_v2/train_uni_ind_pos.pkl')
for i in range(6):
    pd.to_pickle(test_x[i],'../X_v2/test_uni_ind_pos{0}.pkl'.format(i))
