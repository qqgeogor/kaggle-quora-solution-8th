
#coding: utf-8
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
from scipy import sparse as ssp
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler



adj_base_train=pd.read_pickle('data/adj_base_feature_train.pkl')
base_bigram_train=pd.read_csv('data/train_bigram_features.csv')
base_unigram_train=pd.read_csv('data/train_unigram_features.csv')
num_digit_train=pd.read_pickle('data/train_number_diff.pkl')
Ab_feature_train=pd.read_csv('data/train_features.csv')


# In[22]:

cols_base_unigram=['count_q1_in_q2', 'ratio_q1_in_q2',
       'ratio_of_unique_question1', 'ratio_of_unique_question2']
base_unigram_train=base_unigram_train[cols_base_unigram]


# In[23]:

cols_base_bigram=['jaccard', 'sorensen', 'count_q1_in_q2', 'ratio_q1_in_q2',
       'count_of_question1', 'count_of_question2',
       'count_of_unique_question1', 'count_of_unique_question2',
       'ratio_of_unique_question1', 'ratio_of_unique_question2']
base_bigram_train=base_bigram_train[cols_base_bigram]


# In[24]:

cols_Ab=['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'wmd', 'norm_wmd',
       'cosine_distance', 'cityblock_distance', 'jaccard_distance',
       'canberra_distance', 'euclidean_distance', 'minkowski_distance',
       'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec',
       'kur_q2vec']
Ab_feature_train=Ab_feature_train[cols_Ab].astype(float).fillna(0.0)
Ab_feature_train[Ab_feature_train==np.Inf]=0.0


# In[25]:

train_basic=np.hstack([
           Ab_feature_train.values,
           adj_base_train.values,                      ##########basic statistic features
           base_bigram_train.values,
           base_unigram_train.values,
           num_digit_train])


y=pd.read_csv('data/train.csv')['is_duplicate'].values
#train_hhy=pd.read_pickle('data/extra_hhy/train_extra_features.pkl')
train_hhy=pd.read_pickle('data/extra_hhy/data_train.pkl')
train_hhy=np.array(train_hhy)
train_position=pd.read_pickle('data/position/train_1_pos.pkl')
train_lda_q1=pd.read_pickle('data/train_question1bow_lda10.pkl')
train_lda_q2=pd.read_pickle('data/train_question2bow_lda10.pkl')
train_spacy=pd.read_pickle('data/spacy/train_spacy.pkl')
train_dup=pd.read_pickle('data/train_dup_stats.pkl')
train_dup_link2=pd.read_pickle('data/train_link_d2.pkl')
train_pr_edges=pd.read_pickle('data/train_edges_features.pkl')
train_pr_node=pd.read_pickle('data/train_pr_node_feature.pkl').reshape(-1,1)
train_clique=pd.read_pickle('data/max_clique.pkl').reshape(-1,1)[:y.shape[0]]
train_pattern_selected=pd.read_pickle('data/train_pattern_selected.pkl')
train_node2vec=pd.read_pickle('data/train_node2vec.pkl')
train_coo_bigram=pd.read_pickle('data/train_cooccurence_distinct_bigram_encoding_by_label.pkl')
train_coo_unigram=pd.read_pickle('data/train_cooccurence_distinct_encoding_by_label.pkl')
train_doc2vec_sim=pd.read_pickle('data/train_doc2vec_sim.pkl')
#train_doc2vec_q1=pd.read_pickle('data/train_doc2vec_q1_100.pkl')
#train_doc2vec_q2=pd.read_pickle('data/train_doc2vec_q2_100.pkl')
train_indicator=pd.read_pickle('data/indicator/train_indicator.pkl')
train_wmd=pd.read_pickle('data/wmd/train_wmd.pkl')
train_clique_stats=pd.read_csv('data/train_hashed_clique_stats.csv').values
train_selfpretrained_sim=pd.read_pickle('data/train_selftrained_w2v_sim_dist_external.pkl')
#train_topic_q1=pd.read_pickle('data/train_question1_external_word_prior.pkl')
#train_topic_q2=pd.read_pickle('data/train_question2_external_word_prior.pkl')
train_pretrained_glove_sim=pd.read_pickle('data/qianqian/train_pretrained_glove_sim_dist.pkl')
train_entropy_unigram=pd.read_csv('data/qianqian/train_entropy_unigram.csv')
train_entropy_dis_unigram=pd.read_csv('data/qianqian/train_entropy_distinct_unigram.csv')
train_entropy_bigram=pd.read_csv('data/qianqian/train_entropy_bigram.csv')
train_entropy_dis_bigram=pd.read_csv('data/qianqian/train_entropy_distinct_bigram.csv')
train_dis_wordnet_sim=pd.read_csv('data/qianqian/train_distinct_wordnet_stats.csv')
train_graph_neighbor=pd.read_pickle('data/train_neigh.pkl')
train_stop_basic=pd.read_pickle('data/train_stop_basic.pkl')
#train_qhash=pd.read_pickle('data/train_hash_id2.pkl')
train_max_clique_entropy=pd.read_csv('data/train_max_clique_entropy_features.csv')
train_neigh_sim2=pd.read_pickle('data/train_neigh_sim2.pkl')
train_neigh_sim_stats2=pd.read_pickle('data/train_neigh_sim_stats2.pkl')
train_neigh_sim=pd.read_pickle('data/train_neigh_sim.pkl')
train_neigh_sim_stats=pd.read_pickle('data/train_neigh_sim_stats.pkl')
train_internal_rank=pd.read_csv('data/train_internal_rank.csv')
train_q_freqs=pd.read_csv('data/train_q_freqs.csv')
train_spl=pd.read_csv('data/train_spl.csv')
train_inin=pd.read_csv('data/train_inin.csv')
train_neigh_dis=pd.read_pickle('data/train_neigh_dis.pkl')




train_basic=np.hstack([train_hhy,
                  train_position,
                  train_lda_q1,
                  train_lda_q2,
                  train_spacy,
                  train_basic,
                  train_dup,
                  train_dup_link2,
                  train_pr_edges,
                  train_pr_node,
                  train_clique,
                  train_pattern_selected,
                  train_node2vec,
                 ## train_coo_bigram,
                 ## train_coo_unigram,
                  train_doc2vec_sim,
                  train_indicator,
                  train_wmd,
                  train_clique_stats,
                  train_selfpretrained_sim,
                  train_pretrained_glove_sim,
                  train_entropy_unigram,
                  train_entropy_dis_unigram,
                  train_entropy_bigram,
                  train_entropy_dis_bigram,
                  train_dis_wordnet_sim,
                  train_graph_neighbor,
                  train_stop_basic,
                  train_max_clique_entropy,
                  train_neigh_sim2,
                  train_neigh_sim_stats2,
                  train_neigh_sim,
                  train_neigh_sim_stats,
                  train_internal_rank,
                  train_q_freqs,
		  train_spl,
		  train_inin,
		  train_neigh_dis,
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
adj_base_test=pd.read_pickle('data/adj_base_feature_test.pkl')
base_bigram_test=pd.read_csv('data/test_bigram_features.csv')
base_unigram_test=pd.read_csv('data/test_unigram_features.csv')
num_digit_test=pd.read_pickle('data/test_number_diff.pkl')
Ab_feature_test=pd.read_csv('data/test_features.csv')


# In[11]:

cols_base_unigram=['count_q1_in_q2', 'ratio_q1_in_q2',
       'ratio_of_unique_question1', 'ratio_of_unique_question2']
base_unigram_test=base_unigram_test[cols_base_unigram]

cols_base_bigram=['jaccard', 'sorensen', 'count_q1_in_q2', 'ratio_q1_in_q2',
       'count_of_question1', 'count_of_question2',
       'count_of_unique_question1', 'count_of_unique_question2',
       'ratio_of_unique_question1', 'ratio_of_unique_question2']
base_bigram_test=base_bigram_test[cols_base_bigram]

cols_Ab=['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'wmd', 'norm_wmd',
       'cosine_distance', 'cityblock_distance', 'jaccard_distance',
       'canberra_distance', 'euclidean_distance', 'minkowski_distance',
       'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec',
       'kur_q2vec']
Ab_feature_test=Ab_feature_test[cols_Ab].astype(float).fillna(0.0)
Ab_feature_test[Ab_feature_test==np.Inf]=0.0


# In[12]:

test_basic=np.hstack([
           Ab_feature_test.values,
           adj_base_test.values,                      ##########basic statistic features
           base_bigram_test.values,
           base_unigram_test.values,
           num_digit_test])







fdir='data/extra_hhy/data_test'
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


fdir='data/position/test_1_pos'
for i in range(6):
    fname=fdir+str(i)+'.pkl'
    if i==0:
        test_position=pd.read_pickle(fname)
    else:
        test_position=np.vstack([test_position,pd.read_pickle(fname)])
print 'test_position',test_position.shape

fdir='data/wmd/test_wmd'
for i in range(6):
    fname=fdir+str(i)+'.pkl'
    if i==0:
        test_wmd=pd.read_pickle(fname)
    else:
        test_wmd=np.vstack([test_wmd,pd.read_pickle(fname)])
print 'test_wmd',test_wmd.shape

fdir='data/indicator/test_indicator'
for i in range(6):
    fname=fdir+str(i)+'.pkl'
    if i==0:
        test_indicator=pd.read_pickle(fname)
    else:
        test_indicator=np.vstack([test_indicator,pd.read_pickle(fname)])
print 'test_indicator',test_indicator.shape

test_dup=pd.read_pickle('data/test_dup_stats.pkl')
test_dup_link2=pd.read_pickle('data/test_link_d2.pkl')
test_lda_q1=pd.read_pickle('data/test_question1bow_lda10.pkl')
test_lda_q2=pd.read_pickle('data/test_question2bow_lda10.pkl')
test_pr_edges=pd.read_pickle('data/test_edges_features.pkl')
test_pr_node=pd.read_pickle('data/test_pr_node_feature.pkl').reshape(-1,1)
test_clique=pd.read_pickle('data/max_clique.pkl').reshape(-1,1)[y.shape[0]:]
test_pattern_selected=pd.read_pickle('data/test_pattern_selected.pkl')
test_node2vec=pd.read_pickle('data/test_node2vec.pkl')
test_coo_bigram=pd.read_pickle('data/test_cooccurence_distinct_bigram_encoding_by_label.pkl')
test_coo_unigram=pd.read_pickle('data/test_cooccurence_distinct_encoding_by_label.pkl')
test_doc2vec_sim=pd.read_pickle('data/test_doc2vec_sim.pkl')
test_clique_stats=pd.read_csv('data/test_hashed_clique_stats.csv').values
test_selfpretrained_sim=pd.read_pickle('data/test_selftrained_w2v_sim_dist_external.pkl')
test_pretrained_glove_sim=pd.read_pickle('data/qianqian/test_pretrained_glove_sim_dist.pkl')
test_entropy_unigram=pd.read_csv('data/qianqian/test_entropy_unigram.csv')
test_entropy_dis_unigram=pd.read_csv('data/qianqian/test_entropy_distinct_unigram.csv')
test_entropy_bigram=pd.read_csv('data/qianqian/test_entropy_bigram.csv')
test_entropy_dis_bigram=pd.read_csv('data/qianqian/test_entropy_distinct_bigram.csv')
test_dis_wordnet_sim=pd.read_csv('data/qianqian/test_distinct_wordnet_stats.csv')
test_graph_neighbor=pd.read_pickle('data/test_neigh.pkl')
test_stop_basic=pd.read_pickle('data/test_stop_basic.pkl')
#test_qhash=pd.read_pickle('data/test_hash_id2.pkl')
test_max_clique_entropy=pd.read_csv('data/test_max_clique_entropy_features.csv')
test_neigh_sim2=pd.read_pickle('data/test_neigh_sim2.pkl')
test_neigh_sim_stats2=pd.read_pickle('data/test_neigh_sim_stats2.pkl')
test_neigh_sim=pd.read_pickle('data/test_neigh_sim.pkl')
test_neigh_sim_stats=pd.read_pickle('data/test_neigh_sim_stats.pkl')
test_internal_rank=pd.read_csv('data/test_internal_rank.csv')
test_q_freqs=pd.read_csv('data/test_q_freqs.csv')
test_spl=pd.read_csv('data/test_spl.csv')
test_inin=pd.read_csv('data/test_inin.csv')

fdir='data/test_neigh_dis/test_neigh_dis'
for i in range(6):
    fname=fdir+str(i)+'.pkl'
    if i==0:
        test_neigh_dis=pd.read_pickle(fname)
    else:
        test_neigh_dis=np.vstack([test_neigh_dis,pd.read_pickle(fname)])
print 'test_neigh_dis',test_neigh_dis.shape



test_basic=np.hstack([test_hhy,
                  test_position,
                  test_lda_q1,
                  test_lda_q2,
                  test_spacy,
                  test_basic,
                  test_dup,
                  test_dup_link2,
                  test_pr_edges,
                  test_pr_node,
                  test_clique,
                  test_pattern_selected,
                  test_node2vec,
                 ## test_coo_bigram,
                 ## test_coo_unigram,
                  test_doc2vec_sim,
                  test_indicator,
                  test_wmd,
                  test_clique_stats,
                  test_selfpretrained_sim,
                  test_pretrained_glove_sim,
                  test_entropy_unigram,
                  test_entropy_dis_unigram,
                  test_entropy_bigram,
                  test_entropy_dis_bigram,
                  test_dis_wordnet_sim,
                  test_graph_neighbor,
                  test_stop_basic,
                  test_max_clique_entropy,
                  test_neigh_sim2,
                  test_neigh_sim_stats2,
                  test_neigh_sim,
                  test_neigh_sim_stats,
                  test_internal_rank,
                  test_q_freqs,
		  test_spl,
		  test_inin,
                  test_neigh_dis,
                  #test_mf
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

y_train=pd.read_csv('data/train.csv')['is_duplicate'].values


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

pd.to_pickle(te_pred,'stack/lr_model1.train')

#res=res/5.0

lr=LogisticRegression(n_jobs=16,random_state=1123,C=2.0,dual=True)
lr.fit(X_train,y_train)
res=lr.predict_proba(X_test)[:,1]

pd.to_pickle(res,'stack/lr_model1.test')

sub = pd.DataFrame()
sub['test_id'] = range(res.shape[0])
sub['is_duplicate'] = res



def adj(x,te=0.173,tr=0.369): 
    a=te/tr 
    b=(1-te)/(1-tr) 
    return a*x/(a*x+b*(1-x))




