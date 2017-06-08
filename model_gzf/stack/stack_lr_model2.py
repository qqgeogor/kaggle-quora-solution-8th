
# coding: utf-8
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
from scipy import sparse as ssp
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

y_train=pd.read_csv('data/train.csv')['is_duplicate'].values
train_hhy=pd.read_pickle('data/extra_hhy/train_hhy.pkl')
train_stats=pd.read_pickle('data/train_stats_features.pkl')
train_dup=pd.read_pickle('data/train_dup_stats.pkl')
#train_dup_link2=pd.read_pickle('data/train_link_d2.pkl')
train_stats_weight=train_stats*np.log(train_dup.dup_min.values.reshape(-1,1)+1.0)
train_pr_edges=pd.read_pickle('data/train_edges_features.pkl')
train_pr_node=pd.read_pickle('data/train_pr_node_feature.pkl').reshape(-1,1)
train_clique=pd.read_pickle('data/max_clique.pkl').reshape(-1,1)[:y_train.shape[0]]
train_hhy=np.array(train_hhy)
train_spacy=pd.read_pickle('data/spacy/train_spacy.pkl')
train_pattern_selected=pd.read_pickle('data/train_pattern_selected.pkl')
train_node2vec=pd.read_pickle('data/train_node2vec.pkl')
train_coo_bigram=pd.read_pickle('data/train_cooccurence_distinct_bigram_encoding_by_label.pkl')
train_coo_unigram=pd.read_pickle('data/train_cooccurence_distinct_encoding_by_label.pkl')


train_basic=np.hstack([
			train_hhy,
#			train_stats,
			train_dup,
			train_stats_weight,
#			train_dup_link2,
			train_pr_edges,
			train_pr_node,
			train_clique,
			train_spacy,
			train_pattern_selected,
			train_node2vec,
			train_coo_bigram,
			train_coo_unigram
			])


train_basic=pd.DataFrame(train_basic).fillna(0.0).values



mm=MinMaxScaler()
#mm=StandardScaler()
mm.fit(train_basic)
train_basic=mm.transform(train_basic)

print train_basic.shape

tf_idf_co_train=pd.read_pickle('data/train_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_dis_co_train=pd.read_pickle('data/train_distinct_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_q1_unigram_train=pd.read_pickle('data/train_question1_unigram_tfidf.pkl')
tf_idf_q2_unigram_train=pd.read_pickle('data/train_question2_unigram_tfidf.pkl')
tf_idf_q1_bigram_train=pd.read_pickle('data/train_question1_bigram_tfidf.pkl')
tf_idf_q2_bigram_train=pd.read_pickle('data/train_question2_bigram_tfidf.pkl')
train_pattern=pd.read_pickle('data/bowen/train.pattern.onehot.pkl')


print('load test data')

fdir='data/extra_hhy/test_hhy'
for i in range(6):
    fname=fdir+str(i)+'.pkl'
    if i==0:
        test_hhy=pd.read_pickle(fname)
    else:
        test_hhy=np.vstack([test_hhy,pd.read_pickle(fname)])
print 'test_hhy',test_hhy.shape
test_hhy=np.array(test_hhy)

fdir='data/spacy/test_spacy'
for i in range(6):
    fname=fdir+str(i)+'.pkl'
    if i==0:
        test_spacy=pd.read_pickle(fname)
    else:
        test_spacy=np.vstack([test_spacy,pd.read_pickle(fname)])
print 'test_spacy',test_spacy.shape

test_dup=pd.read_pickle('data/test_dup_stats.pkl')
#test_dup_link2=pd.read_pickle('data/test_link_d2.pkl')
test_pr_edges=pd.read_pickle('data/test_edges_features.pkl')
test_pr_node=pd.read_pickle('data/test_pr_node_feature.pkl').reshape(-1,1)
test_clique=pd.read_pickle('data/max_clique.pkl').reshape(-1,1)[y_train.shape[0]:]
test_pattern_selected=pd.read_pickle('data/test_pattern_selected.pkl')
test_node2vec=pd.read_pickle('data/test_node2vec.pkl')
test_coo_bigram=pd.read_pickle('data/test_cooccurence_distinct_bigram_encoding_by_label.pkl')
test_coo_unigram=pd.read_pickle('data/test_cooccurence_distinct_encoding_by_label.pkl')

test_stats=pd.read_pickle('data/test_stats_features.pkl')
test_dup_link2=pd.read_pickle('data/test_link_d2.pkl')
test_stats_weight=test_stats*np.log(test_dup.dup_min.values.reshape(-1,1)+1.0)

test_basic=np.hstack([
                        test_hhy,
#                        test_stats,
                        test_dup,
                        test_stats_weight,
#                        test_dup_link2,
                        test_pr_edges,
                        test_pr_node,
                        test_clique,
                        test_spacy,
                        test_pattern_selected,
                        test_node2vec,
                        test_coo_bigram,
                        test_coo_unigram
                        ])

print test_basic.shape

test_basic=pd.DataFrame(test_basic).fillna(0.0).values
test_basic=mm.transform(test_basic)

tf_idf_co_test=pd.read_pickle('data/test_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_dis_co_test=pd.read_pickle('data/test_distinct_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_q1_unigram_test=pd.read_pickle('data/test_question1_unigram_tfidf.pkl')
tf_idf_q2_unigram_test=pd.read_pickle('data/test_question2_unigram_tfidf.pkl')
tf_idf_q1_bigram_test=pd.read_pickle('data/test_question1_bigram_tfidf.pkl')
tf_idf_q2_bigram_test=pd.read_pickle('data/test_question2_bigram_tfidf.pkl')
test_pattern=pd.read_pickle('data/bowen/test.pattern.onehot.pkl')



#basic=np.vstack([train_basic,test_basic])
#ss=StandardScaler()
#mm=MinMaxScaler()
#basic=mm.fit_transform(ss.fit_transform(basic))
#len_train=train_basic.shape[0]
#train_basic=basic[:len_train]
#test_basic=basic[len_train:]

#print train_basic.mean(),test_basic.mean(),train_basic.max(),test_basic.max()

X_train=ssp.hstack([
		    train_basic,
                   # train_tf_idf,
                   train_pattern,
                   tf_idf_co_train,
                   tf_idf_dis_co_train,
                   tf_idf_q1_unigram_train,
                   tf_idf_q2_unigram_train,
                   tf_idf_q1_bigram_train,
                   tf_idf_q2_bigram_train]).tocsr()


X_test=ssp.hstack([
		    test_basic,
                   # test_tf_idf,
                   test_pattern,
                   tf_idf_co_test,
                   tf_idf_dis_co_test,
                   tf_idf_q1_unigram_test,
                   tf_idf_q2_unigram_test,
                   tf_idf_q1_bigram_test,
                   tf_idf_q2_bigram_test]).tocsr()


# In[17]:
#print X_train.shape
#print X_train.shape,X_test.shape

#X_train=train_basic
#X_test=test_basic
# In[40]:

#y_train=pd.read_csv('data/train.csv')['is_duplicate'].values


# In[ ]:

from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.linear_model import LogisticRegression,RidgeClassifier,Ridge
from sklearn.metrics import log_loss

##########for validation##########
print('begin train....')
te_pred=np.zeros(X_train.shape[0])
cnt=0
skf=StratifiedKFold(y_train,n_folds=5,random_state=1024,shuffle=True)
for tr_ind,te_ind in skf:
    print('begin fold:{}'.format(cnt))
    train=X_train[tr_ind]
    train_y=y_train[tr_ind]
    test=X_train[te_ind]
    test_y=y_train[te_ind]
    print train.shape,test.shape
    print train_y.mean()
    lr=LogisticRegression(n_jobs=16,random_state=1024,C=2.0,dual=True)
    lr.fit(train,train_y)
    print log_loss(test_y,lr.predict_proba(test)[:,1]),log_loss(train_y,lr.predict_proba(train)[:,1])
    #tmp=lr.predict_proba(X_test)[:,1]
    te_pred[te_ind]=lr.predict_proba(test)[:,1]
    #print tmp.mean()
    #res+=tmp
    #print res.mean(),res.shape
    print('end fold:{}'.format(cnt))
    cnt+=1

pd.to_pickle(te_pred,'stack/lr_model2.train')

#res=res/5.0

lr=LogisticRegression(n_jobs=16,random_state=1123,C=2.0,dual=True)
lr.fit(X_train,y_train)
res=lr.predict_proba(X_test)[:,1]

pd.to_pickle(res,'stack/lr_model2.test')

sub = pd.DataFrame()
sub['test_id'] = range(res.shape[0])
sub['is_duplicate'] = res



def adj(x,te=0.173,tr=0.369): 
    a=te/tr 
    b=(1-te)/(1-tr) 
    return a*x/(a*x+b*(1-x))


sub.to_csv('res/stack_lr_model2.csv',index=False)

sub.is_duplicate=sub.is_duplicate.apply(adj)

sub.to_csv('res/stack_lr_model2_adj.csv',index=False)

