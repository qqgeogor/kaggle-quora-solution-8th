# coding: utf-8
import xgboost as xgb
import scipy.stats as sps
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import warnings
import itertools
from pre_data import *
warnings.filterwarnings(action='ignore')


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



print X_train.shape,X_test.shape
y=pd.read_csv('data/train.csv')['is_duplicate'].values
params={
    'max_depth':8,
    'nthread':14,
    'eta':0.003,
    'eval_metric':'logloss',
    #'eval_metric':['logloss','error'],
    'objective':'binary:logistic',
    'subsample':0.7,
    'colsample_bytree':0.5,
    'silent':1,
    'seed':1123,
    'gamma':0.3,
    'min_child_weight':10
    #'scale_pos_weight':0.3
}

te_pred=np.zeros(X_train.shape[0])
cnt=0
best_it=50000
skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=1024)
for ind_tr, ind_te in skf:
    global best_it
    train = X_train[ind_tr]
    test = X_train[ind_te]
    train_y = y[ind_tr]
    test_y = y[ind_te]
    dtrain=xgb.DMatrix(train,train_y)
    dtest=xgb.DMatrix(test,test_y)


    clf=xgb.train(params,dtrain,
              num_boost_round=best_it,
              evals=[(dtrain,'Train'),(dtest,'Test')],
              early_stopping_rounds=500,
              verbose_eval=500
		)
    if cnt==0:
	best_it=clf.best_ntree_limit
    break
    print best_it,clf.best_ntree_limit
    te_pred[ind_te]=clf.predict(dtest)
    print('end fold:{}'.format(cnt))
    cnt+=1
print best_it
dall=xgb.DMatrix(X_train,y)
clf=xgb.train(params,dall,num_boost_round=best_it)
dtest=xgb.DMatrix(X_test)
ans=clf.predict(dtest)
res=pd.DataFrame()
res['test_id']=range(len(ans))
res['is_duplicate']=ans


res.to_csv('res/xgb_6_7_meta_v1.csv',index=False)



def adj(x,te=0.173,tr=0.369):
    a=te/tr
    b=(1-te)/(1-tr)
    return a*x/(a*x+b*(1-x))



res.is_duplicate=res.is_duplicate.apply(adj)
res.to_csv('res/xgb_6_7_meta_v1_adj.csv',index=False)


