import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
import xgboost as xgb
from config import path
seed = 1024
np.random.seed(seed)
# path = '../input/'


feats= ['question1_unigram','question2_unigram','question1_bigram','question2_bigram','question1_distinct_unigram','question2_distinct_unigram','question1_distinct_bigram','question2_distinct_bigram','question1_unigram_question2_unigram','question1_distinct_unigram_question2_distinct_unigram']
X = []
for f in feats:
    X.append(pd.read_pickle(path+'train_%s_tfidf.pkl'%f))
    # X.append(pd.read_pickle(path+'train_%s_tfidf.pkl'%f))
# X.append(np.loadtxt(path+'train_spacy_diff_pretrained.txt'))

# feats2= [
# 'question1',
# 'question2',
# 'question1_porter',
# 'question2_porter',
# ]
# for f in feats2:
#     X.append(pd.read_pickle(path+'train_%s_nmf.pkl'%f))
#     X.append(pd.read_pickle(path+'train_%s_svd.pkl'%f))

X = ssp.hstack(X).tocsr()
len_train = X.shape[0]

train_unigram_features = pd.read_csv(path+'train_unigram_minmax_features.csv').values
train_bigram_features = pd.read_csv(path+'train_bigram_minmax_features.csv').values
train_distinct_unigram_features = pd.read_csv(path+'train_distinct_unigram_minmax_features.csv').values
train_distinct_bigram_features = pd.read_csv(path+'train_distinct_bigram_minmax_features.csv').values
train_porter_stop_features = pd.read_csv(path+'train_porter_stop_features.csv').values

train_position_index = pd.read_csv(path+'train_position_index.csv').values
train_position_normalized_index = pd.read_csv(path+'train_position_normalized_index.csv').values
train_idf_stats_features = pd.read_csv(path+'train_idf_stats_features.csv').values



train_len =pd.read_pickle(path+'train_len.pkl')
train_selftrained_w2v_sim_dist =pd.read_pickle(path+'train_selftrained_w2v_sim_dist.pkl')
train_pretrained_w2v_sim_dist =pd.read_pickle(path+'train_pretrained_w2v_sim_dist.pkl')
train_selftrained_glove_sim_dist =pd.read_pickle(path+'train_selftrained_glove_sim_dist.pkl')
train_gensim_tfidf_sim = pd.read_pickle(path+'train_gensim_tfidf_sim.pkl')[:].reshape(-1,1)

train_hashed_idf = pd.read_csv(path+'train_hashed_idf.csv')
train_hashed_idf['hash_count_same'] = (train_hashed_idf['question1_hash_count']==train_hashed_idf['question2_hash_count']).astype(int)
train_hashed_idf['dup_max']=train_hashed_idf.apply(lambda x:max(x['question1_hash_count'],x['question2_hash_count']),axis=1)
train_hashed_idf['dup_min']=train_hashed_idf.apply(lambda x:min(x['question1_hash_count'],x['question2_hash_count']),axis=1)
train_hashed_idf['dup_dis']=train_hashed_idf['dup_max']-train_hashed_idf['dup_min']
weight = train_hashed_idf['dup_max'].values
f = [
'hash_count_same',
'dup_max',
'dup_min',
'dup_dis',
]

train_hashed_idf = train_hashed_idf[f].values

train_distinct_word_stats = pd.read_csv(path+'train_distinct_word_stats.csv').values
train_distinct_word_stats_pretrained = pd.read_csv(path+'train_distinct_word_stats_pretrained.csv').values

train_spacy_sim_pretrained = pd.read_csv(path+'train_spacy_sim_pretrained.csv')[['spacy_sim']].values
print weight.shape
train_pattern = pd.read_pickle(path+'train.pattern.pkl').reshape((X.shape[0],3))
print train_pattern.shape
# train_tfidf_sim = pd.read_pickle(path+'train_tfidf_sim.pkl').reshape(-1,1)

train_hashed_pr = pd.read_csv(path+'train_hashed_pr.csv').values[:,0].reshape(-1,1)
pr = pd.read_pickle(path+'page_rank.pkl')
train_page_rank = np.array([ pr.get(i,0) for i in range(len_train)]).reshape(-1,1)
train_max_clique = pd.read_pickle(path+'max_clique.pkl')[:X.shape[0]].reshape(-1,1)

train_question1_deepwalk = pd.read_pickle(path+'train_question1_deepwalk.pkl')
train_question2_deepwalk = pd.read_pickle(path+'train_question2_deepwalk.pkl')
train_deepwalk_dist = []
for q1,q2 in zip(train_question1_deepwalk,train_question2_deepwalk):
        vector_diff = q1-q2
        dist = np.sqrt(np.sum(vector_diff**2))
        train_deepwalk_dist.append(dist)
train_deepwalk_dist = np.array(train_deepwalk_dist).reshape(-1,1)

train_edges_features = pd.read_pickle(path+'train_edges_features.pkl')

train_cooccurence_distinct_encoding_by_label = pd.read_pickle(path+'train_cooccurence_distinct_encoding_by_label.pkl')
train_cooccurence_distinct_bigram_encoding_by_label= pd.read_pickle(path+'train_cooccurence_distinct_bigram_encoding_by_label.pkl')

train_hashed_clique_belong = pd.read_csv(path+'train_hashed_clique_belong.csv')
train_hashed_clique_belong['hashed_clique_belong_min'] = train_hashed_clique_belong.apply(lambda x: min(x.values[0],x.values[1]),axis=1)
train_hashed_clique_belong['hashed_clique_belong_max'] = train_hashed_clique_belong.apply(lambda x: max(x.values[0],x.values[1]),axis=1)
f = ['hashed_clique_belong_min','hashed_clique_belong_max']
train_hashed_clique_belong = train_hashed_clique_belong[f].values
train_hashed_clique_total = pd.read_csv(path+'train_hashed_clique_total.csv').values

import os
train = pd.read_csv(path+'train.csv')
train_collins_duffy = []
for p in os.listdir(path+'collins_duffy_train/'):
    tmp = pd.read_csv(path+'collins_duffy_train/'+p)
    tmp = pd.merge(train,tmp,on='id',how='left').fillna(0)
    train_collins_duffy.append(tmp['cdNorm_st'].values.reshape(-1,1))
del train
train_collins_duffy = np.hstack(train_collins_duffy)
print train_collins_duffy.shape

train_basic_features = np.hstack([
    train_unigram_features,
    train_bigram_features,
    # train_porter_stop_features,
    train_distinct_unigram_features,
    # train_distinct_bigram_features,
    # train_position_index,
    # train_position_normalized_index,
    train_idf_stats_features,
    train_len,
    train_selftrained_w2v_sim_dist,
    train_pretrained_w2v_sim_dist,
    train_selftrained_glove_sim_dist,
    train_gensim_tfidf_sim,
    train_hashed_idf,
    # train_distinct_word_stats,
    # train_distinct_word_stats_pretrained,
    train_spacy_sim_pretrained,
    # train_pattern,
    train_hashed_pr,
    train_page_rank,
    train_max_clique,
    # train_question1_deepwalk,
    # train_question2_deepwalk,
    train_edges_features,
    train_cooccurence_distinct_encoding_by_label,
    train_cooccurence_distinct_bigram_encoding_by_label,
    train_hashed_clique_belong,
    # train_hashed_clique_total,
    train_deepwalk_dist,
    # train_collins_duffy,
    ])

weight = train_edges_features['clique_min'].values
# weight = 1.0/(np.log(1+weight))
# for i in range(train_basic_features.shape[1]):
#     train_basic_features[:,i] = train_basic_features[:,i]*weight

y = pd.read_csv(path+"train.csv")['is_duplicate'].values

scaler = MinMaxScaler()
scaler.fit(train_basic_features)
train_basic_features = scaler.transform(train_basic_features)

X = ssp.hstack([X,train_basic_features]).tocsr()


del train_unigram_features
del train_bigram_features
del train_porter_stop_features
del train_distinct_unigram_features
del train_distinct_bigram_features
del train_position_index
del train_position_normalized_index
del train_idf_stats_features
del train_len
del train_selftrained_w2v_sim_dist
del train_pretrained_w2v_sim_dist
del train_selftrained_glove_sim_dist
del train_gensim_tfidf_sim
del train_hashed_idf
del train_distinct_word_stats
del train_distinct_word_stats_pretrained
del train_spacy_sim_pretrained
del train_pattern
del train_hashed_pr
del train_page_rank
del train_max_clique
del train_question1_deepwalk
del train_question2_deepwalk
del train_edges_features
del train_cooccurence_distinct_encoding_by_label
del train_cooccurence_distinct_bigram_encoding_by_label
del train_hashed_clique_belong
del train_deepwalk_dist
del train_hashed_clique_total
del train_collins_duffy
del train_basic_features
print X.shape

X_t = []
for f in feats:
    X_t.append(pd.read_pickle(path+'test_%s_tfidf.pkl'%f))
    # X_t.append(pd.read_pickle(path+'test_%s_tfidf_svd.pkl'%f))

# feats2= [
# 'question1',
# 'question2',
# 'question1_porter',
# 'question2_porter',
# ]
# for f in feats2:
#     X_t.append(pd.read_pickle(path+'test_%s_nmf.pkl'%f))
#     X_t.append(pd.read_pickle(path+'test_%s_svd.pkl'%f))

X_t = ssp.hstack(X_t)
len_test = X_t.shape[0]

test_unigram_features = pd.read_csv(path+'test_unigram_minmax_features.csv').values
test_bigram_features = pd.read_csv(path+'test_bigram_minmax_features.csv').values
test_porter_stop_features = pd.read_csv(path+'test_porter_stop_features.csv').values
test_distinct_unigram_features = pd.read_csv(path+'test_distinct_unigram_minmax_features.csv').values
test_distinct_bigram_features = pd.read_csv(path+'test_distinct_bigram_minmax_features.csv').values
test_position_index = pd.read_csv(path+'test_position_index.csv').values
test_position_normalized_index = pd.read_csv(path+'test_position_normalized_index.csv').values
test_idf_stats_features = pd.read_csv(path+'test_idf_stats_features.csv').values


test_len =pd.read_pickle(path+'test_len.pkl')

test_selftrained_w2v_sim_dist =pd.read_pickle(path+'test_selftrained_w2v_sim_dist.pkl')
test_pretrained_w2v_sim_dist =pd.read_pickle(path+'test_pretrained_w2v_sim_dist.pkl')
test_selftrained_glove_sim_dist =pd.read_pickle(path+'test_selftrained_glove_sim_dist.pkl')
test_gensim_tfidf_sim = pd.read_pickle(path+'test_gensim_tfidf_sim.pkl')[:].reshape(-1,1)

test_hashed_idf = pd.read_csv(path+'test_hashed_idf.csv')
test_hashed_idf['hash_count_same'] = (test_hashed_idf['question1_hash_count']==test_hashed_idf['question2_hash_count']).astype(int)
test_hashed_idf['dup_max']=test_hashed_idf.apply(lambda x:max(x['question1_hash_count'],x['question2_hash_count']),axis=1)
test_hashed_idf['dup_min']=test_hashed_idf.apply(lambda x:min(x['question1_hash_count'],x['question2_hash_count']),axis=1)
test_hashed_idf['dup_dis']=test_hashed_idf['dup_max']-test_hashed_idf['dup_min']
weight = test_hashed_idf['dup_max'].values
f = [
'hash_count_same',
'dup_max',
'dup_min',
'dup_dis',
]

test_hashed_idf = test_hashed_idf[f].values

test_distinct_word_stats = pd.read_csv(path+'test_distinct_word_stats.csv').values
test_distinct_word_stats_pretrained = pd.read_csv(path+'test_distinct_word_stats_pretrained.csv').values

test_spacy_sim_pretrained = pd.read_csv(path+'test_spacy_sim_pretrained.csv')[['spacy_sim']].values
test_pattern = pd.read_pickle(path+'test.pattern.pkl').reshape((X_t.shape[0],3))
# test_tfidf_sim = pd.read_pickle(path+'test_tfidf_sim.pkl').reshape(-1,1)
test_hashed_pr = pd.read_csv(path+'test_hashed_pr.csv').values[:,0].reshape(-1,1)

test_page_rank = np.array([ pr.get(i,0) for i in range(len_train,len_train+len_test)]).reshape(-1,1)

test_max_clique = pd.read_pickle(path+'max_clique.pkl')[len_train:].reshape(-1,1)

test_question1_deepwalk = pd.read_pickle(path+'test_question1_deepwalk.pkl')
test_question2_deepwalk = pd.read_pickle(path+'test_question2_deepwalk.pkl')
test_deepwalk_dist = []
for q1,q2 in zip(test_question1_deepwalk,test_question2_deepwalk):
        vector_diff = q1-q2
        dist = np.sqrt(np.sum(vector_diff**2))
        test_deepwalk_dist.append(dist)
test_deepwalk_dist = np.array(test_deepwalk_dist).reshape(-1,1)



test_edges_features = pd.read_pickle(path+'test_edges_features.pkl')

test_cooccurence_distinct_encoding_by_label = pd.read_pickle(path+'test_cooccurence_distinct_encoding_by_label.pkl')
test_cooccurence_distinct_bigram_encoding_by_label= pd.read_pickle(path+'test_cooccurence_distinct_bigram_encoding_by_label.pkl')

test_hashed_clique_belong = pd.read_csv(path+'test_hashed_clique_belong.csv')
test_hashed_clique_belong['hashed_clique_belong_min'] = test_hashed_clique_belong.apply(lambda x: min(x.values[0],x.values[1]),axis=1)
test_hashed_clique_belong['hashed_clique_belong_max'] = test_hashed_clique_belong.apply(lambda x: max(x.values[0],x.values[1]),axis=1)
f = ['hashed_clique_belong_min','hashed_clique_belong_max']
test_hashed_clique_belong = test_hashed_clique_belong[f].values
test_hashed_clique_total = pd.read_csv(path+'test_hashed_clique_total.csv').values

import os
test = pd.read_csv(path+'test.csv')
test_collins_duffy = []
for p in os.listdir(path+'collins_duffy_test/'):
    tmp = pd.read_csv(path+'collins_duffy_test/'+p)
    tmp = pd.merge(test,tmp,left_on='test_id',right_on='id',how='left').fillna(0)
    test_collins_duffy.append(tmp['cdNorm_st'].values.reshape(-1,1))
del test
test_collins_duffy = np.hstack(test_collins_duffy)
print test_collins_duffy.shape


test_basic_features = np.hstack([
    test_unigram_features,
    test_bigram_features,
    # test_porter_stop_features,
    test_distinct_unigram_features,
    # test_distinct_bigram_features,
    # test_position_index,
    # test_position_normalized_index,
    test_idf_stats_features,
    test_len,
    test_selftrained_w2v_sim_dist,
    test_pretrained_w2v_sim_dist,
    test_selftrained_glove_sim_dist,
    test_gensim_tfidf_sim,
    test_hashed_idf,
    # test_distinct_word_stats,
    # test_distinct_word_stats_pretrained,
    test_spacy_sim_pretrained,
    # test_pattern,
    test_hashed_pr,
    test_page_rank,
    test_max_clique,
    # test_question1_deepwalk,
    # test_question2_deepwalk,
    test_edges_features,
    test_cooccurence_distinct_encoding_by_label,
    test_cooccurence_distinct_bigram_encoding_by_label,
    test_hashed_clique_belong,
    # test_hashed_clique_total,
    test_deepwalk_dist,
    # test_collins_duffy,
    ])

weight = test_edges_features['clique_min'].values
weight = 1.0/(np.log(1+weight))
# for i in range(test_basic_features.shape[1]):
#     test_basic_features[:,i] = test_basic_features[:,i]*weight
# test_basic_features = scaler.transform(test_basic_features)
X_t = ssp.hstack([X_t,test_basic_features]).tocsr()

del test_unigram_features,
del test_bigram_features
del test_porter_stop_features
del test_distinct_unigram_features
del test_distinct_bigram_features
del test_position_index
del test_position_normalized_index
del test_idf_stats_features
del test_len
del test_selftrained_w2v_sim_dist
del test_pretrained_w2v_sim_dist
del test_selftrained_glove_sim_dist
del test_gensim_tfidf_sim
del test_hashed_idf
del test_distinct_word_stats
del test_distinct_word_stats_pretrained
del test_spacy_sim_pretrained
del test_pattern
del test_hashed_pr
del test_page_rank
del test_max_clique
del test_question1_deepwalk
del test_question2_deepwalk
del test_edges_features
del test_cooccurence_distinct_encoding_by_label
del test_cooccurence_distinct_bigram_encoding_by_label
del test_hashed_clique_belong
del test_hashed_clique_total
del test_deepwalk_dist
del test_basic_features
# test_collins_duffy,


from utils import make_mf_classification,make_mf_lsvc_classification

from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint,Callback
from keras.layers.noise import GaussianNoise
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Nadam
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
from keras.regularizers import l2, activity_l2
from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from sklearn.base import BaseEstimator

# LinearSVC.predict_proba = LinearSVC.decision_function
# clf = LinearSVC(C=1,dual=True,random_state=seed)
# make_mf_lsvc_classification(X ,y, clf, X_t, n_folds=5,seed=seed,nb_epoch=1,max_features=1.0,name='lsvc_tfidf',path=path)


# clf = LogisticRegression(C=2,dual=True,random_state=seed)
# make_mf_classification(X ,y, clf, X_t, n_folds=5,seed=seed,nb_epoch=1,max_features=1.0,name='lr_tfidf',path=path)

# def build_model(maxlen=238,hidden=512):
#     inputs = []
#     inputs_q1 = Input(shape=(maxlen,),name='input_q1',sparse=True)
#     inputs.append(inputs_q1)

#     fc1 = Dense(hidden)(inputs_q1)
#     fc1 = PReLU()(fc1)
#     fc1 = Dropout(0.85)(fc1)

#     outputs = Dense(1,activation='sigmoid')(fc1)
#     model = Model(input=inputs, output=outputs)

#     model.compile(
#                 optimizer='adam',
#                 loss = 'binary_crossentropy',
#               )
#     return model

# class NN(BaseEstimator):
#     def __init__(self,input_shape,hidden=32):
#         self.input_shape = input_shape
#         self.hidden = hidden
#     def fit(self,X,y):
#         model = build_model(maxlen=self.input_shape,hidden=self.hidden)
#         model.fit(X,y,nb_epoch=8,batch_size=128,shuffle=True,verbose=1)
#         self.model=model
#     def predict_proba(self,X):
#         y_p = self.model.predict(X,batch_size=1024).ravel()
#         y_p_inv= np.ones(y_p.shape[0])-y_p
#         y_p = np.hstack([y_p_inv.reshape(-1,1),y_p.reshape(-1,1)])
#         return y_p

# clf = NN(X.shape[1],32)
# make_mf_classification(X ,y, clf, X_t, n_folds=5,seed=seed,nb_epoch=1,max_features=1.0,name='nn_tfidf',path=path)

