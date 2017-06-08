import numpy as np
from scipy import sparse as ssp
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,accuracy_score


def make_mf_classification(X ,y, clf, X_test, n_folds=5,seed=1024,nb_epoch=50,max_features=0.75,name='xgb',path=''):
    n = X.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- classifier
    '''
    print clf
    for epoch in range(nb_epoch):
        print "Start epoch:",epoch
        mf_tr = np.zeros((X.shape[0],len(np.unique(y))))
        mf_te = np.zeros((X_test.shape[0],len(np.unique(y))))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X,y)

        
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]

            if ssp.issparse(X):
                clf.fit(X_tr.tocsc(), y_tr)    
                mf_tr[ind_te] += clf.predict_proba(X_te.tocsc())
            else:
                clf.fit(X_tr, y_tr)    
                mf_tr[ind_te] += clf.predict_proba(X_te)
            del X_tr
            del X_te

            l = 600000
            y_pred = []
            for batch in range(4):
                if batch!=3:
                    X_tmp = X_test[l*batch:l*(batch+1)]
                else:
                    X_tmp = X_test[l*batch:]
                if ssp.issparse(X):
                    y_pred.append(clf.predict_proba(X_tmp.tocsc()))
                else:
                    y_pred.append(clf.predict_proba(X_tmp))    
            y_pred = np.vstack(y_pred)
            mf_te += y_pred
            score = log_loss(y_te, mf_tr[ind_te])
            print '\tpred[{}] score:{}'.format(epoch, score)
        mf_te/=n_folds
        pd.to_pickle(mf_tr,path+'X_mf_%s_%s_random.pkl'%(name,epoch))
        pd.to_pickle(mf_te,path+'X_t_mf_%s_%s_random.pkl'%(name,epoch))


def make_mf_lsvc_classification(X ,y, clf, X_test, n_folds=5,seed=1024,nb_epoch=50,max_features=0.75,name='xgb',path=''):
    n = X.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- classifier
    '''
    print clf
    for epoch in range(nb_epoch):
        print "Start epoch:",epoch
        mf_tr = np.zeros(X.shape[0])
        mf_te = np.zeros(X_test.shape[0])
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X,y)

        
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            clf.fit(X_tr, y_tr)
            mf_tr[ind_te] += clf.predict_proba(X_te).ravel()
            score = accuracy_score(y_te, clf.predict(X_te).ravel())
            del X_tr
            del X_te
            
            mf_te += clf.predict_proba(X_test).ravel()

            print '\tpred[{}] score:{}'.format(epoch, score)
        mf_te/=n_folds
        pd.to_pickle(mf_tr.reshape(-1,1),path+'X_mf_%s_%s_random.pkl'%(name,epoch))
        pd.to_pickle(mf_te.reshape(-1,1),path+'X_t_mf_%s_%s_random.pkl'%(name,epoch))


def make_mf_regression(X ,y, clf, X_test, n_folds=5,seed=1024,nb_epoch=50,max_features=0.75,name='xgb',path=''):
    n = X.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- classifier
    '''
    print clf
    for epoch in range(nb_epoch):
        print "Start epoch:",epoch
        mf_tr = np.zeros(X.shape[0])
        mf_te = np.zeros(X_test.shape[0])
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X,y)

        
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            clf.fit(X_tr, y_tr)
            mf_tr[ind_te] += clf.predict(X_te)
            del X_tr
            del X_te

            l = 600000
            y_pred = []
            for batch in range(4):
                X_tmp = X_test[l*batch:l*(batch+1)]
                y_pred.append(clf.predict(X_tmp))
            y_pred = np.concatenate(y_pred)
            mf_te += y_pred
            score = log_loss(y_te, mf_tr[ind_te])
            print '\tpred[{}] score:{}'.format(epoch, score)
        mf_te/=n_folds
        pd.to_pickle(mf_tr,path+'X_mf_%s_%s_random.pkl'%(name,epoch))
        pd.to_pickle(mf_te,path+'X_t_mf_%s_%s_random.pkl'%(name,epoch))
