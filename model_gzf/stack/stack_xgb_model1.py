# coding: utf-8
import xgboost as xgb
import scipy.stats as sps
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import warnings
import itertools
warnings.filterwarnings(action='ignore')
from pre_data import get_tfidf_train,get_other_train,get_tfidf_test,get_other_test

X_tfidf=get_tfidf_train()
X_other=get_other_train()
X_test_tfidf=get_tfidf_test()
X_test_other=get_other_test()

X_train=np.hstack([X_tfidf,X_other])
X_test=np.hstack([X_test_tfidf,X_test_other])

print X_train.shape,X_test.shape
y=pd.read_csv('data/train.csv')['is_duplicate'].values
params={
    'max_depth':8,
    'nthread':14,
    'eta':0.01,
    'eval_metric':['error','logloss'],
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
    #break
    train = X_train[ind_tr]
    test = X_train[ind_te]
    train_y = y[ind_tr]
    test_y = y[ind_te]
    dtrain=xgb.DMatrix(train,train_y)
    dtest=xgb.DMatrix(test,test_y)


    clf=xgb.train(params,dtrain,
              num_boost_round=best_it,
              evals=[(dtrain,'Train'),(dtest,'Test')],
              early_stopping_rounds=200,
              verbose_eval=500
		)
    if cnt==0:
	best_it=clf.best_ntree_limit
        print best_it,clf.best_ntree_limit
    te_pred[ind_te]=clf.predict(dtest)
    print('end fold:{}'.format(cnt))
    cnt+=1

pd.to_pickle(te_pred,'stack/mf_3/xgb_model1.train')
dall=xgb.DMatrix(X_train,y)
clf=xgb.train(params,dall,num_boost_round=best_it)
pd.to_pickle(clf,'model/xgboost_5_30.model')
dtest=xgb.DMatrix(X_test)
ans=clf.predict(dtest)
pd.to_pickle(ans,'stack/mf_3/xgb_model1.test')



