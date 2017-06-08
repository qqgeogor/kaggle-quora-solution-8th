import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.decomposition import PCA,NMF,SparsePCA,TruncatedSVD
from .utils import split_data
import scipy.stats as sps

path = '../X_v3/'

train_vec = pd.read_pickle(path + 'train_glove_vec.pkl')
test_vec= []
for i in range(6):
    test_vec.append(pd.read_pickle(path+'test_glove_vec{0}.pkl'.format(i)))
test_vec = np.vstack(test_vec)

q1_vec_train = train_vec[:,:100]
q2_vec_train = train_vec[:,100:200]
dif_vec_train = train_vec[:,200:]
q1_vec_test = test_vec[:,:100]
q2_vec_test = test_vec[:,100:200]
dif_vec_test = test_vec[:,200:]
y_train = pd.read_csv('../data/train.csv')['is_duplicate']

#aggregate and separate
def generate_stats_feature(q1_vec,q2_vec):
    std_stats = np.vstack([q1_vec.std(axis=1),q2_vec.std(axis=1)]).T
    std_feature = np.vstack([std_stats.mean(axis=1),std_stats.std(axis=1)]).T
    mean_stats = np.vstack([q1_vec.mean(axis=1),q2_vec.mean(axis=1)]).T
    mean_feature = np.vstack([mean_stats.max(axis=1),mean_stats.std(axis=1),mean_stats.mean(axis=1)]).T
    return np.hstack([std_feature,mean_feature])

stats_train = generate_stats_feature(q1_vec_train,q2_vec_train)
stats_test = generate_stats_feature(q1_vec_test,q2_vec_test)
#sps.spearmanr(std_stats.sum(axis=1),y_train)[0]
pd.to_pickle(stats_train,'../X_v2/glove_stats_train.pkl')
pd.to_pickle(stats_test,'../X_v2/glove_stats_test.pkl')

#pca
q1_vec_train = np.hstack([q1_vec_train,q2_vec_train])
q1_vec_test = np.hstack([q1_vec_test,q2_vec_test])

path = '../X_v2/'

q1_vec = np.vstack([q1_vec_train,q1_vec_test]).T
pca=TruncatedSVD(n_components=80,random_state=1123)#(samples,features)
pca.fit(q1_vec)
# pd.to_pickle(pca.components_.T,path+'pca_20_glove_q1.pkl')
q1_pca = pca.components_.T

# diff vec

# pca=TruncatedSVD(n_components=30,random_state=1123)#(features,samples)
# pca.fit(diff_vec)
# diff_pca = pca.components_.T

train_q = q1_pca[:train_vec.shape[0]]
test_q = q1_pca[train_vec.shape[0]:]

train_q = np.hstack([train_q,dif_vec_train])
test_q = np.hstack([test_q,dif_vec_test])


dif_stats_train = dif_vec_train.std(axis=1)
dif_stats_test = dif_vec_test.std(axis=1)
stats_train = pd.read_pickle('../X_v2/glove_stats_train.pkl')
stats_test = pd.read_pickle('../X_v2/glove_stats_test.pkl')
train_all = np.hstack([train_q,dif_stats_train.reshape(-1,1),stats_train])
test_all = np.hstack([test_q,dif_stats_test.reshape(-1,1),stats_test])

pd.to_pickle(train_all,path+'train_glove.pkl')
test_x = split_data.split_test(test_all)
for i in range(6):
    pd.to_pickle(test_x[i],path+'test_glove{0}.pkl'.format(i))



#select the import feature
# sps.spearmanr(train_all,y_train)[0][-1,:]
# train_glove_vec = train_all
#
# col_list = []
# for i in range(train_all.shape[1]):
#     if abs(sps.spearmanr(train_all[:,i],y_train)[0])>0.05:
#         col_list.append(i)