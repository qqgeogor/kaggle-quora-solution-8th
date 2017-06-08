# coding: utf-8

import xgboost as xgb
import scipy.stats as sps
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import warnings
import itertools
import lightgbm as lgb
from pre_data import get_tfidf_train,get_other_train,\
		get_tfidf_test,get_other_test,\
		get_extra_train,get_extra_test,\
		get_mf_train,get_mf_test
warnings.filterwarnings(action='ignore')
#####################################
###  7042  0.168659 
###  9042  0.166343
###  9546  0.165927



X_tfidf=get_tfidf_train()
X_other=get_other_train()
X_test_tfidf=get_tfidf_test()
X_test_other=get_other_test()

X_extra=get_extra_train()
X_test_extra=get_extra_test()

X_train=np.hstack([X_tfidf,X_other,X_extra])
X_test=np.hstack([X_test_tfidf,X_test_other,X_test_extra])

train_mf=get_mf_train()
test_mf=get_mf_test()

print X_train.shape
print X_test.shape
y=pd.read_csv('data/train.csv')['is_duplicate'].values

train_coo_bigram=pd.read_pickle('data/train_cooccurence_distinct_bigram_encoding_by_label.pkl')
train_coo_unigram=pd.read_pickle('data/train_cooccurence_distinct_encoding_by_label.pkl')
train_neigh_pos_v2=pd.read_pickle('data/train_neigh_pos_v2.pkl')
train_hashed_clique_stats_sep=pd.read_csv('data/train_hashed_clique_stats_sep.csv')
test_coo_bigram=pd.read_pickle('data/test_cooccurence_distinct_bigram_encoding_by_label.pkl')
test_coo_unigram=pd.read_pickle('data/test_cooccurence_distinct_encoding_by_label.pkl')
test_neigh_pos_v2=pd.read_pickle('data/test_neigh_pos_v2.pkl')
test_hashed_clique_stats_sep=pd.read_csv('data/test_hashed_clique_stats_sep.csv')


X_train=np.hstack([X_train,
		train_coo_bigram,
		train_coo_unigram,
		train_neigh_pos_v2,
                train_hashed_clique_stats_sep,
		train_mf])


print X_train.shape

X_test=np.hstack([X_test,
                  test_coo_bigram,
                  test_coo_unigram,
                  test_neigh_pos_v2,
                  test_hashed_clique_stats_sep,
                  test_mf,])

print X_test.shape

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 31,
    'learning_rate': 0.003,
    'feature_fraction': 0.55,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'verbose': 0,
    'max_depth':8,
    'min_gain_to_split':0.3,
    'nthread':16
}

te_pred=np.zeros(X_train.shape[0])
global best_it
best_it=8888
cnt=0
skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=1024)
for ind_tr, ind_te in skf:
    break
    global best_it
    train = X_train[ind_tr]
    test = X_train[ind_te]
    train_y = y[ind_tr]
    test_y = y[ind_te]
    dtrain = lgb.Dataset(train, train_y)
    dtest = lgb.Dataset(test, test_y, reference=dtrain)
    clf=lgb.train(  params,
                dtrain,
                num_boost_round=170000,
                valid_sets=dtest,
                early_stopping_rounds=500,
    	        verbose_eval=500
		)

    #print clf.best_iteration
    best_it=clf.best_iteration
    print best_it
    break
    te_pred[ind_te]=clf.predict(test)
    print('end fold:{}'.format(cnt))
    cnt+=1

#pd.to_pickle(te_pred,'stack/lgb_model_1.train')
print best_it
dall=lgb.Dataset(X_train,y)
clf=lgb.train(params,dall,num_boost_round=best_it)
#pd.to_pickle(clf,'model/lgb_5_23.model')
#clf = pd.read_pickle('model/lgb_5_23.model')



ans=clf.predict(X_test)
res=pd.DataFrame()
res['test_id']=range(len(ans))
res['is_duplicate']=ans


res.to_csv('res/lgb_6_7_meta_v1.csv',index=False)



def adj(x,te=0.173,tr=0.369): 
    a=te/tr 
    b=(1-te)/(1-tr) 
    return a*x/(a*x+b*(1-x))



res.is_duplicate=res.is_duplicate.apply(adj)
res.to_csv('res/lgb_6_7_meta_v1_adj.csv',index=False)

