import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.decomposition import PCA,NMF,SparsePCA,TruncatedSVD
from .utils import split_data

np.random.rand(1024)

path = '../X_v2/'

train_char = pd.read_pickle(path+'train/train_char_dis.pkl')
test_x = []
for i in range(6):
    test_x.append(pd.read_pickle(path+'test/test_char_dis{0}.pkl'.format(i)))
test_char = np.vstack(test_x)

q1_train_char = train_char[:,:36]
q2_train_char = train_char[:,36:72]
q1_test_char = test_char[:,:36]
q2_test_char = test_char[:,36:72]

import scipy.stats as sps

y_train = pd.read_csv('../data/train.csv')['is_duplicate']

# stats
std_stats_train = np.hstack([q1_train_char.std(axis=1).reshape(-1,1),q2_train_char.std(axis=1).reshape(-1,1)])
std_fea_train = np.vstack([std_stats_train.max(axis=1),std_stats_train.std(axis=1)]).T

std_stats_test= np.hstack([q1_test_char.std(axis=1).reshape(-1,1),q2_test_char.std(axis=1).reshape(-1,1)])
std_fea_test = np.vstack([std_stats_test.max(axis=1),std_stats_test.std(axis=1)]).T


#dif
q1_q2_rest = np.vstack([train_char[:,72:],test_char[:,72:]])
# select_index = []
# for i in range(q1_q2_rest.shape[1]):
#     s = sps.spearmanr(q1_q2_rest[:y_train.shape[0], i], y_train)[0]
#     if abs(s)>0.01:
#         select_index.append(s)
# q1_q2_rest = q1_q2_rest[:,select_index]
# def drop_feature(data):
#     drop_list = []
#     for i in range(data.shape[1]):
#         for j in range(i,data.shape[1]):
#             s = sps.spearmanr(data[:,i],data[:,j])[0]
#             if abs(s)>0.8:
#                 drop_list.append(j)
#     drop_list = set(drop_list)
#     return  drop_list

#pca
q1_q2_train = train_char[:,:72]
q1_q2_test = test_char[:,:72]
q1_q2_all = np.vstack([q1_q2_train,q1_q2_test]).T

pca = TruncatedSVD(n_components=20,random_state=1123)#(samples,features)
pca.fit(q1_q2_all)
q1_q2_all = pca.components_.T

q1_q2_all = np.hstack([q1_q2_all,q1_q2_rest])

train_char = q1_q2_all[:train_char.shape[0]]
test_char = q1_q2_all[train_char.shape[0]:]
train_char = np.hstack([train_char,std_fea_train])
test_char = np.hstack([test_char,std_fea_test])

pd.to_pickle(train_char,path+'train_char.pkl')
test_x = split_data.split_test(test_char)
for i in range(6):
    pd.to_pickle(test_x[i],path+'test_char{0}.pkl'.format(i))

