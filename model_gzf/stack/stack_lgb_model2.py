# coding: utf-8
import scipy.stats as sps
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import warnings
import itertools
import lightgbm as lgb
from pre_data import get_tfidf_train,get_other_train,get_tfidf_test,get_other_test,get_extra_train,get_extra_test

warnings.filterwarnings(action='ignore')

############################


X_tfidf=get_tfidf_train()
X_other=get_other_train()
X_test_tfidf=get_tfidf_test()
X_test_other=get_other_test()
X_extra=get_extra_train()
X_test_extra=get_extra_test()


X_train=np.hstack([X_tfidf,X_other,X_extra])
X_test=np.hstack([X_test_tfidf,X_test_other,X_test_extra])



print X_train.shape
print X_test.shape
y=pd.read_csv('data/train.csv')['is_duplicate'].values
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 128,
    'learning_rate': 0.01,
    'feature_fraction': 0.55,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'verbose': 0,
    #'max_depth':8,
    #'min_gain_to_split':0.3,
    'nthread':14
}

te_pred=np.zeros(X_train.shape[0])
best_it=50000
cnt=0
skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=1024)
for ind_tr, ind_te in skf:
#    break
    global best_it
    train = X_train[ind_tr]
    test = X_train[ind_te]
    train_y = y[ind_tr]
    test_y = y[ind_te]
    dtrain = lgb.Dataset(train, train_y)
    dtest = lgb.Dataset(test, test_y, reference=dtrain)
    clf=lgb.train(  params,
                dtrain,
                num_boost_round=best_it,
                valid_sets=dtest,
                early_stopping_rounds=1000,
                 verbose_eval=500
                )

    #print clf.best_iteration
    if cnt==0:
        best_it=clf.best_iteration
    te_pred[ind_te]=clf.predict(test)
    print('end fold:{}'.format(cnt))
    cnt+=1

pd.to_pickle(te_pred,'stack/mf_3/lgb_model2.train')
clf=xgb.train(params,dall,num_boost_round=best_it)
ans=clf.predict(X_test)
pd.to_pickle(ans,'stack/mf_3/lgb_model2.test')



