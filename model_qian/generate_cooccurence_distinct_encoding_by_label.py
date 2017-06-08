from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from config import path
seed = 1024
np.random.seed(seed)

train = pd.read_csv(path+'train_cooccurrence_distinct.csv')
test = pd.read_csv(path+'test_cooccurrence_distinct.csv')
train['is_duplicate'] = pd.read_csv(path+'train.csv')['is_duplicate']
y = train['is_duplicate'].values
feature=['question1_distinct_unigram_question2_distinct_unigram']

def prepare(x,y,d):
    # print(x)
    x = x[0].split(' ')
    for w in x:
        cnt = d.get(w,0)
        cnt+=y
        d[w]=cnt
    return d


def fillin(x,d,length):
    l = np.array([d.get(w,0)/float(length) for w in x[0].split(' ')])
    l_mean = np.mean(l)
    l_min = np.min(l)
    l_max = np.max(l)
    l_std = np.std(l)
    return np.array([l_min,l_max,l_mean,l_std])

X = train[feature].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X,y)

X_mf = np.zeros((X.shape[0],4))

for ind_tr, ind_te in skf:
    d = dict()
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    
    for xx,yy in zip(X_train.tolist(),y_train):
        d = prepare(xx,yy,d)
    length = X_train.shape[0]
    yp = []
    for xx in X_test.tolist():
        yp.append(fillin(xx,d,length))
    X_mf[ind_te]+=np.array(yp)


X_t = test[feature].values
X_t_mf = np.zeros((X_t.shape[0],4))
d = dict()
X_train = X
X_test = X_t
length = X_train.shape[0]
y_train = y
for xx,yy in zip(X_train.tolist(),y_train):
    d = prepare(xx,yy,d)
yp = []
for xx in X_test.tolist():
    yp.append(fillin(xx,d,length))
X_t_mf+=np.array(yp)

pd.to_pickle(X_mf,path+'train_cooccurence_distinct_encoding_by_label.pkl')
pd.to_pickle(X_t_mf,path+'test_cooccurence_distinct_encoding_by_label.pkl')


train = pd.read_csv(path+'train_cooccurrence_distinct_bigram.csv')
test = pd.read_csv(path+'test_cooccurrence_distinct_bigram.csv')
train['is_duplicate'] = pd.read_csv(path+'train.csv')['is_duplicate']
y = train['is_duplicate'].values
feature=['question1_distinct_bigram_question2_distinct_bigram']

def prepare(x,y,d):
    # print(x)
    x = x[0].split(' ')
    for w in x:
        cnt = d.get(w,0)
        cnt+=y
        d[w]=cnt
    return d


def fillin(x,d,length):
    l = np.array([d.get(w,0)/float(length) for w in x[0].split(' ')])
    l_mean = np.mean(l)
    l_min = np.min(l)
    l_max = np.max(l)
    l_std = np.std(l)
    return np.array([l_min,l_max,l_mean,l_std])

X = train[feature].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X,y)

X_mf = np.zeros((X.shape[0],4))

for ind_tr, ind_te in skf:
    d = dict()
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    
    for xx,yy in zip(X_train.tolist(),y_train):
        d = prepare(xx,yy,d)
    length = X_train.shape[0]
    yp = []
    for xx in X_test.tolist():
        yp.append(fillin(xx,d,length))
    X_mf[ind_te]+=np.array(yp)


X_t = test[feature].values
X_t_mf = np.zeros((X_t.shape[0],4))
d = dict()
X_train = X
X_test = X_t
length = X_train.shape[0]
y_train = y
for xx,yy in zip(X_train.tolist(),y_train):
    d = prepare(xx,yy,d)
yp = []
for xx in X_test.tolist():
    yp.append(fillin(xx,d,length))
X_t_mf+=np.array(yp)

pd.to_pickle(X_mf,path+'train_cooccurence_distinct_bigram_encoding_by_label.pkl')
pd.to_pickle(X_t_mf,path+'test_cooccurence_distinct_bigram_encoding_by_label.pkl')
