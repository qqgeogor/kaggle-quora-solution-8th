
# coding: utf-8

# In[33]:

#import xgboost as xgb
import scipy.stats as sps
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import warnings
import itertools
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
warnings.filterwarnings(action='ignore')



##############use svd 20##############



#svd20_tfidf_co_train=pd.read_pickle('data/train_svd_20_question1_unigram_question2_unigram_tfidf.pkl')
#svd20_dis_tfidf_co_train=pd.read_pickle('data/train_svd_20_distinct_question1_unigram_question2_unigram_tfidf.pkl')
#svd20_tfidf_q1_unigram_train=pd.read_pickle('data/train_svd_20_question1_unigram_tfidf.pkl')
#svd20_tfidf_q2_unigram_train=pd.read_pickle('data/train_svd_20_question2_unigram_tfidf.pkl')
#svd20_tfidf_q1_bigram_train=pd.read_pickle('data/train_svd_20_question1_bigram_tfidf.pkl')
#svd20_tfidf_q2_bigram_train=pd.read_pickle('data/train_svd_20_question2_bigram_tfidf.pkl')



#NMF6_tfidf_co_train=pd.read_pickle('data/train_NMF_6_question1_unigram_question2_unigram_tfidf.pkl')
#NMF6_dis_tfidf_co_train=pd.read_pickle('data/train_NMF_6_distinct_question1_unigram_question2_unigram_tfidf.pkl')
#NMF6_tfidf_q1_unigram_train=pd.read_pickle('data/train_NMF_6_question1_unigram_tfidf.pkl')
#NMF6_tfidf_q2_unigram_train=pd.read_pickle('data/train_NMF_6_question2_unigram_tfidf.pkl')
#NMF6_tfidf_q1_bigram_train=pd.read_pickle('data/train_NMF_6_question1_bigram_tfidf.pkl')
#NMF6_tfidf_q2_bigram_train=pd.read_pickle('data/train_NMF_6_question2_bigram_tfidf.pkl')



#pretrain_w2c_train=pd.read_pickle('data/train_pretrained_w2v_sim_dist.pkl')
#selftrain_w2c_train=pd.read_pickle('data/train_selftrained_w2v_sim_dist.pkl')
#raw_jaccard_train=pd.read_pickle('data/train_jaccard.pkl').values.reshape(-1,1)
#raw_interaction_train=pd.read_pickle('data/train_interaction.pkl').values.reshape(-1,1)
#train_bigram_tfidf_sim=pd.read_pickle('data/train_bigram_tfidf_sim.pkl')
#train_unigram_tfidf_sim=pd.read_pickle('data/train_unigram_tfidf_sim.pkl')
#train_bigram_lsi_sim=pd.read_pickle('data/train_bigram_lsi_100_sim.pkl')
#train_unigram_lsi_sim=pd.read_pickle('data/train_unigram_lsi_100_sim.pkl')



#train_tfidf=np.hstack([
#           svd20_tfidf_co_train,             ############part two features tfidf and w2v
#           svd20_dis_tfidf_co_train,
#           svd20_tfidf_q1_unigram_train,
#           svd20_tfidf_q2_unigram_train,
#           svd20_tfidf_q1_bigram_train,
#           svd20_tfidf_q2_bigram_train,
#           NMF6_tfidf_co_train,
#           NMF6_dis_tfidf_co_train,
#           NMF6_tfidf_q1_unigram_train,
#           NMF6_tfidf_q2_unigram_train,
#           NMF6_tfidf_q1_bigram_train,
#           NMF6_tfidf_q2_bigram_train,
#           pretrain_w2c_train,
#           selftrain_w2c_train,
#           raw_jaccard_train,
#           raw_interaction_train,   
#           train_bigram_tfidf_sim.reshape(-1,1),
#           train_unigram_tfidf_sim.reshape(-1,1),
#           train_bigram_lsi_sim.reshape(-1,1),
#           train_unigram_lsi_sim.reshape(-1,1)])


# In[46]:

#X_train.shape


# In[47]:

#pd.to_pickle(X_train,'data/train_svd_nmf_sim_tfidf_features.pkl')


# In[48]:

##################################
#######use svd 100 ###########


# In[19]:

svd100_tfidf_co_train=pd.read_pickle('data/train_svd_100_question1_unigram_question2_unigram_tfidf.pkl')
svd100_dis_tfidf_co_train=pd.read_pickle('data/train_svd_100_distinct_question1_unigram_question2_unigram_tfidf.pkl')
svd20_tfidf_q1_unigram_train=pd.read_pickle('data/train_svd_20_question1_unigram_tfidf.pkl')
svd20_tfidf_q2_unigram_train=pd.read_pickle('data/train_svd_20_question2_unigram_tfidf.pkl')
svd20_tfidf_q1_bigram_train=pd.read_pickle('data/train_svd_20_question1_bigram_tfidf.pkl')
svd20_tfidf_q2_bigram_train=pd.read_pickle('data/train_svd_20_question2_bigram_tfidf.pkl')
NMF6_tfidf_co_train=pd.read_pickle('data/train_NMF_6_question1_unigram_question2_unigram_tfidf.pkl')
NMF6_dis_tfidf_co_train=pd.read_pickle('data/train_NMF_6_distinct_question1_unigram_question2_unigram_tfidf.pkl')
NMF6_tfidf_q1_unigram_train=pd.read_pickle('data/train_NMF_6_question1_unigram_tfidf.pkl')
NMF6_tfidf_q2_unigram_train=pd.read_pickle('data/train_NMF_6_question2_unigram_tfidf.pkl')
NMF6_tfidf_q1_bigram_train=pd.read_pickle('data/train_NMF_6_question1_bigram_tfidf.pkl')
NMF6_tfidf_q2_bigram_train=pd.read_pickle('data/train_NMF_6_question2_bigram_tfidf.pkl')
pretrain_w2c_train=pd.read_pickle('data/train_pretrained_w2v_sim_dist.pkl')
selftrain_w2c_train=pd.read_pickle('data/train_selftrained_w2v_sim_dist.pkl')
raw_jaccard_train=pd.read_pickle('data/train_jaccard.pkl').values.reshape(-1,1)
raw_interaction_train=pd.read_pickle('data/train_interaction.pkl').values.reshape(-1,1)
train_bigram_tfidf_sim=pd.read_pickle('data/train_bigram_tfidf_sim.pkl')
train_unigram_tfidf_sim=pd.read_pickle('data/train_unigram_tfidf_sim.pkl')
train_bigram_lsi_sim=pd.read_pickle('data/train_bigram_lsi_100_sim.pkl')
train_unigram_lsi_sim=pd.read_pickle('data/train_unigram_lsi_100_sim.pkl')
train_tfidf=np.hstack([
           svd100_tfidf_co_train,             ############part two features tfidf and w2v
           svd100_dis_tfidf_co_train,
           svd20_tfidf_q1_unigram_train,
           svd20_tfidf_q2_unigram_train,
           svd20_tfidf_q1_bigram_train,
           svd20_tfidf_q2_bigram_train,
           NMF6_tfidf_co_train,
           NMF6_dis_tfidf_co_train,
           NMF6_tfidf_q1_unigram_train,
           NMF6_tfidf_q2_unigram_train,
           NMF6_tfidf_q1_bigram_train,
           NMF6_tfidf_q2_bigram_train,
           pretrain_w2c_train,
           selftrain_w2c_train,
           raw_jaccard_train,
           raw_interaction_train,   
           train_bigram_tfidf_sim.reshape(-1,1),
           train_unigram_tfidf_sim.reshape(-1,1),
           train_bigram_lsi_sim.reshape(-1,1),
           train_unigram_lsi_sim.reshape(-1,1)])
print train_tfidf.shape
#pd.to_pickle(TRAIN_IDF,'data/train_svd_nmf_sim_tfidf_all_features.pkl')


# In[20]:

########################


# In[21]:

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


train_basic=pd.DataFrame(train_basic).fillna(0.0).values
print train_basic.shape

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


X_train=np.hstack([train_hhy,
                  train_position,
                  train_lda_q1,
                  train_lda_q2,
                  train_spacy,
                  train_tfidf,
                  train_basic,
                  train_dup,
                  train_dup_link2,
                  train_pr_edges,
                  train_pr_node,
                  train_clique,
                  train_pattern_selected,
                  train_node2vec,
                  #train_coo_bigram,
                  #train_coo_unigram,
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
                  #train_topic_q1,
                  #train_topic_q2,
                  #train_mf,
                  ])









# In[29]:
X_train=pd.DataFrame(X_train).fillna(0.0).values
print X_train.shape
#X_train=np.array(X_train)
from sklearn.metrics import log_loss
# In[30]:
clf=ExtraTreesClassifier(n_estimators=2000, criterion='gini', max_depth=25, 
                     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                     max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, 
                     bootstrap=False, oob_score=False, n_jobs=14, random_state=1123, verbose=0,
                     warm_start=False, class_weight=None)



te_pred=np.zeros(X_train.shape[0])
#best_it=18000
cnt=0
skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=1024)
for ind_tr, ind_te in skf:
#    break
    train = X_train[ind_tr]
    test = X_train[ind_te]
    train_y = y[ind_tr]
    test_y = y[ind_te]

    clf.fit(train,train_y)
    print log_loss(test_y,clf.predict_proba(test)[:,1]),log_loss(train_y,clf.predict_proba(train)[:,1])
    #print clf.best_iteration
    #best_it=clf.best_iteration
    te_pred[ind_te]=clf.predict_proba(test)[:,1]
    print('end fold:{}'.format(cnt))
    cnt+=1

pd.to_pickle(te_pred,'stack/et_model1.train')
#print best_it
clf.fit(X_train,y)
pd.to_pickle(clf,'model/et_5_11.model')


# In[23]:

#################################
#clf=pd.read_pickle('model/lgb_5_8.model')

# In[3]:

svd100_tfidf_co_test=pd.read_pickle('data/test_svd_100_question1_unigram_question2_unigram_tfidf.pkl')
svd100_dis_tfidf_co_test=pd.read_pickle('data/test_svd_100_distinct_question1_unigram_question2_unigram_tfidf.pkl')
svd20_tfidf_q1_unigram_test=pd.read_pickle('data/test_svd_20_question1_unigram_tfidf.pkl')
svd20_tfidf_q2_unigram_test=pd.read_pickle('data/test_svd_20_question2_unigram_tfidf.pkl')
svd20_tfidf_q1_bigram_test=pd.read_pickle('data/test_svd_20_question1_bigram_tfidf.pkl')
svd20_tfidf_q2_bigram_test=pd.read_pickle('data/test_svd_20_question2_bigram_tfidf.pkl')


# In[4]:

NMF6_tfidf_co_test=pd.read_pickle('data/test_NMF_6_question1_unigram_question2_unigram_tfidf.pkl')
NMF6_dis_tfidf_co_test=pd.read_pickle('data/test_NMF_6_distinct_question1_unigram_question2_unigram_tfidf.pkl')
NMF6_tfidf_q1_unigram_test=pd.read_pickle('data/test_NMF_6_question1_unigram_tfidf.pkl')
NMF6_tfidf_q2_unigram_test=pd.read_pickle('data/test_NMF_6_question2_unigram_tfidf.pkl')
NMF6_tfidf_q1_bigram_test=pd.read_pickle('data/test_NMF_6_question1_bigram_tfidf.pkl')
NMF6_tfidf_q2_bigram_test=pd.read_pickle('data/test_NMF_6_question2_bigram_tfidf.pkl')


# In[5]:

pretrain_w2c_test=pd.read_pickle('data/test_pretrained_w2v_sim_dist.pkl')
selftrain_w2c_test=pd.read_pickle('data/test_selftrained_w2v_sim_dist.pkl')
raw_jaccard_train=pd.read_pickle('data/test_jaccard.pkl').values.reshape(-1,1)
raw_interaction_train=pd.read_pickle('data/test_interaction.pkl').values.reshape(-1,1)
test_bigram_tfidf_sim=pd.read_pickle('data/test_bigram_tfidf_sim.pkl')
test_unigram_tfidf_sim=pd.read_pickle('data/test_unigram_tfidf_sim.pkl')
test_bigram_lsi_sim=pd.read_pickle('data/test_bigram_lsi_100_sim.pkl')
test_unigram_lsi_sim=pd.read_pickle('data/test_unigram_lsi_100_sim.pkl')


# In[6]:

test_tfidf=np.hstack([
           svd100_tfidf_co_test,
           svd100_dis_tfidf_co_test,
           svd20_tfidf_q1_unigram_test,
           svd20_tfidf_q2_unigram_test,
           svd20_tfidf_q1_bigram_test,
           svd20_tfidf_q2_bigram_test,
           NMF6_tfidf_co_test,
           NMF6_dis_tfidf_co_test,
           NMF6_tfidf_q1_unigram_test,
           NMF6_tfidf_q2_unigram_test,
           NMF6_tfidf_q1_bigram_test,
           NMF6_tfidf_q2_bigram_test,
           pretrain_w2c_test,
           selftrain_w2c_test,
           raw_jaccard_train,
           raw_interaction_train,
           test_bigram_tfidf_sim.reshape(-1,1),
           test_unigram_tfidf_sim.reshape(-1,1),
           test_bigram_lsi_sim.reshape(-1,1),
           test_unigram_lsi_sim.reshape(-1,1)])


# In[7]:

print 'Test tfidf',test_tfidf.shape


# In[10]:

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


test_basic=pd.DataFrame(test_basic).fillna(0.0).values

print 'Test basic',test_basic.shape

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




X_test=np.hstack([test_hhy,
                  test_position,
                  test_lda_q1,
                  test_lda_q2,
                  test_spacy,
                  test_tfidf,
                  test_basic,
                  test_dup,
                  test_dup_link2,
                  test_pr_edges,
                  test_pr_node,
                  test_clique,
                  test_pattern_selected,
                  test_node2vec,
                  ##test_coo_bigram,
                  ##test_coo_unigram,
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


print 'X_test',X_test.shape
#print type(X_test)
X_test=pd.DataFrame(X_test).fillna(0.0).values
#X_test=np.array(X_test)
#print type(X_test)
# In[ ]:

#dtest=lgb.Dataset(X_test)
ans=clf.predict_proba(X_test)[:,1]
res=pd.DataFrame()
res['test_id']=range(len(ans))
res['is_duplicate']=ans

pd.to_pickle(ans,'stack/et_model1.test')
# In[ ]:

#res.to_csv('res/lgb_5_12.csv',index=False)


# In[ ]:

def adj(x,te=0.173,tr=0.369): 
    a=te/tr 
    b=(1-te)/(1-tr) 
    return a*x/(a*x+b*(1-x))


# In[ ]:

#res.is_duplicate=res.is_duplicate.apply(adj)
#res.to_csv('res/lgb_5_12_adj.csv',index=False)

