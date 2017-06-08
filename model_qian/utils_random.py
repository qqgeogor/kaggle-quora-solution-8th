import numpy as np
from scipy import sparse as ssp
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def make_mf_classification(X ,y, clf, X_test, n_folds=5,seed=1024,nb_epoch=50,max_features=0.75,name='xgb',path=''):
    n = X.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- classifier
    '''
    print clf
    np.random.seed(seed)
    feature_index = np.arange(X.shape[1])
    for epoch in range(nb_epoch):
        print "Start epoch:",epoch
        mf_tr = np.zeros((X.shape[0],len(np.unique(y))))
        mf_te = np.zeros((X_test.shape[0],len(np.unique(y))))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X,y)

        np.random.shuffle(feature_index)
        new_index = feature_index[:int(max_features*len(feature_index))]

        for ind_tr, ind_te in skf:
            if ssp.issparse(X):
                X_tr = X[ind_tr].tocsc()[:,new_index]
                X_te = X[ind_te].tocsc()[:,new_index]
            else:
                X_tr = X[ind_tr][:,new_index]
                X_te = X[ind_te][:,new_index]
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            
            clf.fit(X_tr, y_tr)
            mf_tr[ind_te] += clf.predict_proba(X_te)
            mf_te += clf.predict_proba(X_test[:,new_index])
            score = log_loss(y_te, mf_tr[ind_te])
            print '\tpred[{}] score:{}'.format(epoch, score)
        mf_te/=n_folds
        pd.to_pickle(mf_tr,path+'X_mf_%s_%s_random_r.pkl'%(name,epoch))
        pd.to_pickle(mf_te,path+'X_t_mf_%s_%s_random_r.pkl'%(name,epoch))
