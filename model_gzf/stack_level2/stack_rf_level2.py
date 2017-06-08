import os
import sys
import tqdm
import h5py 
import numpy as np
import pandas as pd
import scipy.sparse as ssp
from sklearn.cross_validation import StratifiedKFold

np.random.seed(1123)
reload(sys)
sys.setdefaultencoding('utf-8')
from pre_data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


X_tfidf=get_tfidf_train()
X_other=get_other_train()

X_test_tfidf=get_tfidf_test()
X_test_other=get_other_test()

X_extra=get_extra_train()
X_test_extra=get_extra_test()


train_mf=get_mf_train()
test_mf=get_mf_test()

y=pd.read_csv('data/train.csv')['is_duplicate'].values

train_coo_bigram=pd.read_pickle('data/train_cooccurence_distinct_bigram_encoding_by_label.pkl')
train_coo_unigram=pd.read_pickle('data/train_cooccurence_distinct_encoding_by_label.pkl')
train_neigh_pos_v2=pd.read_pickle('data/train_neigh_pos_v2.pkl')
train_hashed_clique_stats_sep=pd.read_csv('data/train_hashed_clique_stats_sep.csv')

test_coo_bigram=pd.read_pickle('data/test_cooccurence_distinct_bigram_encoding_by_label.pkl')
test_coo_unigram=pd.read_pickle('data/test_cooccurence_distinct_encoding_by_label.pkl')
test_neigh_pos_v2=pd.read_pickle('data/test_neigh_pos_v2.pkl')
test_hashed_clique_stats_sep=pd.read_csv('data/test_hashed_clique_stats_sep.csv')


X_train=np.hstack([X_other,X_extra,
                train_coo_bigram,
                train_coo_unigram,
                train_neigh_pos_v2,
                train_hashed_clique_stats_sep])


print X_train.shape

X_test=np.hstack([X_test_other,X_test_extra,
                  test_coo_bigram,
                  test_coo_unigram,
                  test_neigh_pos_v2,
                  test_hashed_clique_stats_sep])

print X_test.shape

from sklearn.preprocessing import StandardScaler,MinMaxScaler
X_train=pd.DataFrame(X_train).fillna(0.0).values

mm=MinMaxScaler()
mm.fit(X_train)
X_train=mm.transform(X_train)

X_train=np.hstack([X_train,
		       X_tfidf,
                       train_mf])
print X_train.shape

LEN_RAW_INPUT=X_train.shape[1]
X_train=X_train.astype('float32')

X_test=pd.DataFrame(X_test).fillna(0.0).values
X_test=mm.transform(X_test)

X_test=np.hstack([X_test,
                  X_test_tfidf,
		test_mf])
print X_test.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
fold=0
te_pred=np.zeros(X_train.shape[0])
test_pred=np.zeros(X_test.shape[0])
skf = StratifiedKFold(y,n_folds=5, shuffle=True, random_state=1024)
for ind_tr, ind_te in skf:
    #break
    clf=RandomForestClassifier(n_estimators=500,max_depth=None,random_state=1123,
                              n_jobs=13,min_weight_fraction_leaf=0,criterion='gini',
                              min_samples_leaf=5)
    X_tr_raw=X_train[ind_tr]
    y_tr=y[ind_tr]
    X_te_raw=X_train[ind_te]
    y_te=y[ind_te]
    print X_tr_raw.shape,X_te_raw.shape,y_tr.mean(),y_te.mean()
    clf.fit(X_tr_raw ,y_tr)
    print log_loss(y_te,clf.predict_proba(X_te_raw)[:,1]),log_loss(y_tr,clf.predict_proba(X_tr_raw)[:,1])
    #te_pred[ind_te]=model.predict(X_te_raw)
    test_pred+=clf.predict_proba(X_test)[:,1]
    print('end fold:{}'.format(fold))    
    fold+=1

test_pred/=5
           

res=pd.DataFrame()
res['test_id']=range(len(test_pred))
res['is_duplicate']=test_pred


res.to_csv('res/rf_6_7_meta_v1.csv',index=False)



def adj(x,te=0.173,tr=0.369):
    a=te/tr
    b=(1-te)/(1-tr)
    return a*x/(a*x+b*(1-x))



res.is_duplicate=res.is_duplicate.apply(adj)
res.to_csv('res/rf_6_7_meta_v1_adj.csv',index=False)



