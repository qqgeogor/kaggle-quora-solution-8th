import pandas as pd
import numpy as np
import os
import scipy.stats as sps

def get_tfidf_train():

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

    return train_tfidf

def get_other_train():

    adj_base_train=pd.read_pickle('data/adj_base_feature_train.pkl')
    base_bigram_train=pd.read_csv('data/train_bigram_features.csv')
    base_unigram_train=pd.read_csv('data/train_unigram_features.csv')
    num_digit_train=pd.read_pickle('data/train_number_diff.pkl')
    Ab_feature_train=pd.read_csv('data/train_features.csv')


    cols_base_unigram=['count_q1_in_q2', 'ratio_q1_in_q2',
       'ratio_of_unique_question1', 'ratio_of_unique_question2']
    base_unigram_train=base_unigram_train[cols_base_unigram]


    cols_base_bigram=['jaccard', 'sorensen', 'count_q1_in_q2', 'ratio_q1_in_q2',
       'count_of_question1', 'count_of_question2',
       'count_of_unique_question1', 'count_of_unique_question2',
       'ratio_of_unique_question1', 'ratio_of_unique_question2']
    base_bigram_train=base_bigram_train[cols_base_bigram]


    cols_Ab=['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'wmd', 'norm_wmd',
       'cosine_distance', 'cityblock_distance', 'jaccard_distance',
       'canberra_distance', 'euclidean_distance', 'minkowski_distance',
       'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec',
       'kur_q2vec']
    Ab_feature_train=Ab_feature_train[cols_Ab].astype(float).fillna(0.0)
    Ab_feature_train[Ab_feature_train==np.Inf]=0.0

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
    train_hashed_clique_w=pd.read_csv('data/train_hashed_clique_w.csv')
    train_shared_nodes=pd.read_csv('data/train_shared_nodes.csv')
    train_shared_nodes_w=pd.read_csv('data/train_shared_nodes_w.csv')
    train_neigh_glove=pd.read_csv('data/train_neighbour_embedding_dist.csv')
    train_neigh_w2v=pd.read_csv('data/train_neighbour_embedding_dist_w2v.csv')
    
    #train_neigh_long_match=pd.read_pickle('data/train_neigh_long_match.pkl')
    #train_neigh_jarccard=pd.read_pickle('data/train_neigh_jarccard.pkl')
    #train_neigh_dice=pd.read_pickle('data/train_neigh_dice.pkl')
    #train_neigh_edit=pd.read_pickle('data/train_neigh_edit.pkl')
    #train_neigh_jarccard2=pd.read_pickle('data/train_neigh_jarccard2.pkl')
    #train_neigh_dice2=pd.read_pickle('data/train_neigh_dice2.pkl')
    #train_neigh_edit2=pd.read_pickle('data/train_neigh_edit2.pkl')
    #train_neigh_long2=pd.read_pickle('data/train_neigh_long2.pkl')
    #train_neigh_jarccard3=pd.read_pickle('data/train_neigh_jarccard3.pkl')
    #train_neigh_jarccard4=pd.read_pickle('data/train_neigh_jarccard4.pkl')
    #train_neigh_dice4=pd.read_pickle('data/train_neigh_dice4.pkl')
    train_neighbour_embedding_dist_fast=pd.read_csv('data/train_neighbour_embedding_dist_fast.csv')
    #train_neigh_wmd=pd.read_pickle('data/train_neigh_wmd.pkl')

    X_train=np.hstack([train_hhy,
                  train_position,
                  train_lda_q1,
                  train_lda_q2,
                  train_spacy,
                  #train_tfidf,
                  train_basic,
                  train_dup,
                  train_dup_link2,
                  train_pr_edges,
                  train_pr_node,
                  train_clique,
                  train_pattern_selected,
                  train_node2vec,
  ##                train_coo_bigram,                 ######only use for level 2
  ##                train_coo_unigram,                ######only use for level 2
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
                  train_hashed_clique_w,
                  train_shared_nodes,
                  train_shared_nodes_w,
		  train_neigh_glove,
		  train_neigh_w2v,
                  
       		  #train_neigh_long_match,
                  #train_neigh_jarccard,
                  #train_neigh_dice,
                  #train_neigh_edit,
                  #train_neigh_jarccard2,
                  #train_neigh_dice2,
                  #train_neigh_edit2,
                  #train_neigh_long2,
                  #train_neigh_jarccard3,
                  #train_neigh_jarccard4,
                  #train_neigh_dice4,
		  train_neighbour_embedding_dist_fast,
		  #train_neigh_wmd,
                  ])

    print X_train.shape
    return X_train




def get_tfidf_test():
    svd100_tfidf_co_test=pd.read_pickle('data/test_svd_100_question1_unigram_question2_unigram_tfidf.pkl')
    svd100_dis_tfidf_co_test=pd.read_pickle('data/test_svd_100_distinct_question1_unigram_question2_unigram_tfidf.pkl')
    svd20_tfidf_q1_unigram_test=pd.read_pickle('data/test_svd_20_question1_unigram_tfidf.pkl')
    svd20_tfidf_q2_unigram_test=pd.read_pickle('data/test_svd_20_question2_unigram_tfidf.pkl')
    svd20_tfidf_q1_bigram_test=pd.read_pickle('data/test_svd_20_question1_bigram_tfidf.pkl')
    svd20_tfidf_q2_bigram_test=pd.read_pickle('data/test_svd_20_question2_bigram_tfidf.pkl')



    NMF6_tfidf_co_test=pd.read_pickle('data/test_NMF_6_question1_unigram_question2_unigram_tfidf.pkl')
    NMF6_dis_tfidf_co_test=pd.read_pickle('data/test_NMF_6_distinct_question1_unigram_question2_unigram_tfidf.pkl')
    NMF6_tfidf_q1_unigram_test=pd.read_pickle('data/test_NMF_6_question1_unigram_tfidf.pkl')
    NMF6_tfidf_q2_unigram_test=pd.read_pickle('data/test_NMF_6_question2_unigram_tfidf.pkl')
    NMF6_tfidf_q1_bigram_test=pd.read_pickle('data/test_NMF_6_question1_bigram_tfidf.pkl')
    NMF6_tfidf_q2_bigram_test=pd.read_pickle('data/test_NMF_6_question2_bigram_tfidf.pkl')



    pretrain_w2c_test=pd.read_pickle('data/test_pretrained_w2v_sim_dist.pkl')
    selftrain_w2c_test=pd.read_pickle('data/test_selftrained_w2v_sim_dist.pkl')
    raw_jaccard_train=pd.read_pickle('data/test_jaccard.pkl').values.reshape(-1,1)
    raw_interaction_train=pd.read_pickle('data/test_interaction.pkl').values.reshape(-1,1)
    test_bigram_tfidf_sim=pd.read_pickle('data/test_bigram_tfidf_sim.pkl')
    test_unigram_tfidf_sim=pd.read_pickle('data/test_unigram_tfidf_sim.pkl')
    test_bigram_lsi_sim=pd.read_pickle('data/test_bigram_lsi_100_sim.pkl')
    test_unigram_lsi_sim=pd.read_pickle('data/test_unigram_lsi_100_sim.pkl')

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



    print 'Test tfidf',test_tfidf.shape
    return test_tfidf


def get_other_test():


    adj_base_test=pd.read_pickle('data/adj_base_feature_test.pkl')
    base_bigram_test=pd.read_csv('data/test_bigram_features.csv')
    base_unigram_test=pd.read_csv('data/test_unigram_features.csv')
    num_digit_test=pd.read_pickle('data/test_number_diff.pkl')
    Ab_feature_test=pd.read_csv('data/test_features.csv')



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


    test_basic=np.hstack([
           Ab_feature_test.values,
           adj_base_test.values,                      ##########basic statistic features
           base_bigram_test.values,
           base_unigram_test.values,
           num_digit_test])


    test_basic=pd.DataFrame(test_basic).fillna(0.0).values

    print 'Test basic',test_basic.shape

    fdir = 'data/extra_hhy/data_test'
    for i in range(6):
        fname = fdir + str(i) + '.pkl'
        if i == 0:
            test_hhy = pd.read_pickle(fname)
        else:
            test_hhy = np.vstack([test_hhy, pd.read_pickle(fname)])
    print 'test_hhy', test_hhy.shape
    test_hhy = np.array(test_hhy)

    fdir = 'data/spacy/test_spacy'
    for i in range(6):
        fname = fdir + str(i) + '.pkl'
        if i == 0:
            test_spacy = pd.read_pickle(fname)
        else:
            test_spacy = np.vstack([test_spacy, pd.read_pickle(fname)])
    print 'test_spacy', test_spacy.shape

    fdir = 'data/position/test_1_pos'
    for i in range(6):
        fname = fdir + str(i) + '.pkl'
        if i == 0:
            test_position = pd.read_pickle(fname)
        else:
            test_position = np.vstack([test_position, pd.read_pickle(fname)])
    print 'test_position', test_position.shape

    fdir = 'data/wmd/test_wmd'
    for i in range(6):
        fname = fdir + str(i) + '.pkl'
        if i == 0:
            test_wmd = pd.read_pickle(fname)
        else:
            test_wmd = np.vstack([test_wmd, pd.read_pickle(fname)])
    print 'test_wmd', test_wmd.shape

    fdir = 'data/indicator/test_indicator'
    for i in range(6):
        fname = fdir + str(i) + '.pkl'
        if i == 0:
            test_indicator = pd.read_pickle(fname)
        else:
            test_indicator = np.vstack([test_indicator, pd.read_pickle(fname)])
    print 'test_indicator', test_indicator.shape

    y=pd.read_csv('data/train.csv')['is_duplicate'].values
    test_dup = pd.read_pickle('data/test_dup_stats.pkl')
    test_dup_link2 = pd.read_pickle('data/test_link_d2.pkl')
    test_lda_q1 = pd.read_pickle('data/test_question1bow_lda10.pkl')
    test_lda_q2 = pd.read_pickle('data/test_question2bow_lda10.pkl')
    test_pr_edges = pd.read_pickle('data/test_edges_features.pkl')
    test_pr_node = pd.read_pickle('data/test_pr_node_feature.pkl').reshape(-1, 1)
    test_clique = pd.read_pickle('data/max_clique.pkl').reshape(-1, 1)[y.shape[0]:]
    test_pattern_selected = pd.read_pickle('data/test_pattern_selected.pkl')
    test_node2vec = pd.read_pickle('data/test_node2vec.pkl')
    test_coo_bigram = pd.read_pickle('data/test_cooccurence_distinct_bigram_encoding_by_label.pkl')
    test_coo_unigram = pd.read_pickle('data/test_cooccurence_distinct_encoding_by_label.pkl')
    test_doc2vec_sim = pd.read_pickle('data/test_doc2vec_sim.pkl')
    test_clique_stats = pd.read_csv('data/test_hashed_clique_stats.csv').values
    test_selfpretrained_sim = pd.read_pickle('data/test_selftrained_w2v_sim_dist_external.pkl')
    test_pretrained_glove_sim = pd.read_pickle('data/qianqian/test_pretrained_glove_sim_dist.pkl')
    test_entropy_unigram = pd.read_csv('data/qianqian/test_entropy_unigram.csv')
    test_entropy_dis_unigram = pd.read_csv('data/qianqian/test_entropy_distinct_unigram.csv')
    test_entropy_bigram = pd.read_csv('data/qianqian/test_entropy_bigram.csv')
    test_entropy_dis_bigram = pd.read_csv('data/qianqian/test_entropy_distinct_bigram.csv')
    test_dis_wordnet_sim = pd.read_csv('data/qianqian/test_distinct_wordnet_stats.csv')
    test_graph_neighbor = pd.read_pickle('data/test_neigh.pkl')
    test_stop_basic = pd.read_pickle('data/test_stop_basic.pkl')
    # test_qhash=pd.read_pickle('data/test_hash_id2.pkl')
    test_max_clique_entropy = pd.read_csv('data/test_max_clique_entropy_features.csv')
    test_neigh_sim2 = pd.read_pickle('data/test_neigh_sim2.pkl')
    test_neigh_sim_stats2 = pd.read_pickle('data/test_neigh_sim_stats2.pkl')
    test_neigh_sim = pd.read_pickle('data/test_neigh_sim.pkl')
    test_neigh_sim_stats = pd.read_pickle('data/test_neigh_sim_stats.pkl')
    test_internal_rank = pd.read_csv('data/test_internal_rank.csv')
    test_q_freqs = pd.read_csv('data/test_q_freqs.csv')
    test_spl = pd.read_csv('data/test_spl.csv')
    test_inin = pd.read_csv('data/test_inin.csv')
    # test_hashed_clique_stats_sep=pd.read_csv('data/test_hashed_clique_stats_sep.csv')
    # test_neigh_dis=pd.read_pickle('data/test_neigh_dis.pkl')
    # print test_qhash.shape,test_max_clique_entropy.shape
    fdir = 'data/test_neigh_dis/test_neigh_dis'
    for i in range(6):
        fname = fdir + str(i) + '.pkl'
        if i == 0:
            test_neigh_dis = pd.read_pickle(fname)
        else:
            test_neigh_dis = np.vstack([test_neigh_dis, pd.read_pickle(fname)])
    print 'test_neigh_dis', test_neigh_dis.shape

    test_hashed_clique_w = pd.read_csv('data/test_hashed_clique_w.csv')
    test_shared_nodes = pd.read_csv('data/test_shared_nodes.csv')
    test_shared_nodes_w = pd.read_csv('data/test_shared_nodes_w.csv')
    test_neigh_glove=pd.read_csv('data/test_neighbour_embedding_dist.csv')
    test_neigh_w2v=pd.read_csv('data/test_neighbour_embedding_dist_w2v.csv')
    #test_neigh_long_match=pd.read_pickle('data/test_neigh_long_match.pkl')
    #test_neigh_jarccard=pd.read_pickle('data/test_neigh_jarccard.pkl')
    #test_neigh_dice=pd.read_pickle('data/test_neigh_dice.pkl')
    #test_neigh_edit=pd.read_pickle('data/test_neigh_edit.pkl')
    #test_neigh_jarccard2=pd.read_pickle('data/test_neigh_jarccard2.pkl')
    #test_neigh_dice2=pd.read_pickle('data/test_neigh_dice2.pkl')
    #test_neigh_edit2=pd.read_pickle('data/test_neigh_edit2.pkl')
    #test_neigh_long2=pd.read_pickle('data/test_neigh_long2.pkl')
    #test_neigh_jarccard3=pd.read_pickle('data/test_neigh_jarccard3.pkl')
    #test_neigh_jarccard4=pd.read_pickle('data/test_neigh_jarccard4.pkl')
    #test_neigh_dice4=pd.read_pickle('data/test_neigh_dice4.pkl')
    test_neighbour_embedding_dist_fast=pd.read_csv('data/test_neighbour_embedding_dist_fast.csv')
    #test_neigh_wmd=pd.read_pickle('data/test_neigh_wmd.pkl')

    X_test = np.hstack([test_hhy,
                        test_position,
                        test_lda_q1,
                        test_lda_q2,
                        test_spacy,
                        #test_tfidf,
                        test_basic,
                        test_dup,
                        test_dup_link2,
                        test_pr_edges,
                        test_pr_node,
                        test_clique,
                        test_pattern_selected,
                        test_node2vec,
##                        test_coo_bigram,
##                        test_coo_unigram,
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
                        test_hashed_clique_w,
                        test_shared_nodes,
                        test_shared_nodes_w,
			test_neigh_glove,
			test_neigh_w2v,
			#test_neigh_long_match,
       			#test_neigh_jarccard,
    			#test_neigh_dice,
    			#test_neigh_edit,
    			#test_neigh_jarccard2,
    			#test_neigh_dice2,
    			#test_neigh_edit2,
    			#test_neigh_long2,
    			#test_neigh_jarccard3,
    			#test_neigh_jarccard4,
   			#test_neigh_dice4,
                        test_neighbour_embedding_dist_fast,
			#test_neigh_wmd,
                        ])

    print X_test.shape
    return X_test



def get_extra_train():
##############################extra features##################################
    train_simhash_features=pd.read_csv('data/extra_feature/train_simhash_features.csv')
    train_selftrained_w2v_sim_dist=pd.read_pickle('data/extra_feature/train_selftrained_w2v_sim_dist.pkl')
    train_selftrained_glove_sim_dist=pd.read_pickle('data/extra_feature/train_selftrained_glove_sim_dist.pkl')
    train_pretrained_w2v_sim_dist=pd.read_pickle('data/extra_feature/train_pretrained_w2v_sim_dist.pkl')
    train_distinct_word_stats_selftrained_glove=pd.read_csv('data/extra_feature/train_distinct_word_stats_selftrained_glove.csv')
    train_distinct_word_stats_pretrained=pd.read_csv('data/extra_feature/train_distinct_word_stats_pretrained.csv')
    train_distinct_word_stats=pd.read_csv('data/extra_feature/train_distinct_word_stats.csv')


    X_train=np.hstack([train_simhash_features,
    		train_selftrained_w2v_sim_dist,
    		train_selftrained_glove_sim_dist,
    		train_pretrained_w2v_sim_dist,
    		train_distinct_word_stats_selftrained_glove,
    		train_distinct_word_stats_pretrained,
    		train_distinct_word_stats,])


    print X_train.shape

    return X_train





def get_extra_test():  
##############################extra features##################################
    test_simhash_features=pd.read_csv('data/extra_feature/test_simhash_features.csv')
    test_selftrained_w2v_sim_dist=pd.read_pickle('data/extra_feature/test_selftrained_w2v_sim_dist.pkl')
    test_selftrained_glove_sim_dist=pd.read_pickle('data/extra_feature/test_selftrained_glove_sim_dist.pkl')
    test_pretrained_w2v_sim_dist=pd.read_pickle('data/extra_feature/test_pretrained_w2v_sim_dist.pkl')
    test_distinct_word_stats_selftrained_glove=pd.read_csv('data/extra_feature/test_distinct_word_stats_selftrained_glove.csv')
    test_distinct_word_stats_pretrained=pd.read_csv('data/extra_feature/test_distinct_word_stats_pretrained.csv')
    test_distinct_word_stats=pd.read_csv('data/extra_feature/test_distinct_word_stats.csv')


    X_test=np.hstack([    test_simhash_features,
    test_selftrained_w2v_sim_dist,
    test_selftrained_glove_sim_dist,
    test_pretrained_w2v_sim_dist,
    test_distinct_word_stats_selftrained_glove,
    test_distinct_word_stats_pretrained,
    test_distinct_word_stats,])


    print X_test.shape

    return X_test 


def get_mf_train():
    train_lgb1=pd.read_pickle('stack/mf_3/lgb_model3.train').reshape(-1,1)
    train_rf1=pd.read_pickle('stack/mf_3/rf_model1.train').reshape(-1,1)
    train_et1=pd.read_pickle('stack/mf_3/et_model1.train').reshape(-1,1)
    train_lr1=pd.read_pickle('stack/mf_3/lr_model1.train').reshape(-1,1)
    train_nn1=pd.read_pickle('stack/mf_3/nn_model1.train').reshape(-1,1)
    train_nn2=pd.read_pickle('stack/mf_3/nn_model2.train').reshape(-1,1)
    train_xgb1=pd.read_pickle('stack/mf_3/xgb_model1.train').reshape(-1,1)
#train_lstm1=pd.read_pickle('stack/mf_3/lstm_model1.train').reshape(-1,1)
#train_lstm2=pd.read_pickle('stack/mf_3/lstm_model2.train').reshape(-1,1)
#train_lgb2=pd.read_pickle('stack/mf_3/lgb_model2.train').reshape(-1,1)
    train_siamse=pd.read_pickle('stack/mf_3/train_siamse_match_mf.pkl').reshape(-1,1)
    train_bowen1=pd.read_csv('stack/mf_3/jasonnetwork.bs300.h512.att_concat.drop20.watt1.b0.l0.train.csv')['is_duplicate'].values.reshape(-1,1)
    train_bowen2=pd.read_csv('stack/mf_3/jasonnetwork.bs300.h512.att_concat.drop25.watt1.b0.l0.train.csv')['is_duplicate'].values.reshape(-1,1)
    train_bowen3=pd.read_csv('stack/mf_3/jasonnetwork.bs300.h512.att_concat.drop20.watt1.b0.l0.train2.csv')['is_duplicate'].values.reshape(-1,1)
    train_hhy=pd.read_pickle('stack/mf_3/dp_train_2_mf.pkl').reshape(-1,1)
##############################################
    train_mf=np.hstack([train_lgb1,train_rf1,train_et1,
                        train_lr1,train_nn1,train_nn2,train_xgb1,train_siamse,train_bowen1,train_bowen2,train_bowen3,train_hhy])
    print 'train_mf:',train_mf.shape

    len_train=train_lgb1.shape[0]
    meta_fe=['_lr_tfidf_2_0_','_nn_3_0_','_et_3_0_','_rf_3_0_','_lgb_3_0_','_lsvc_tfidf_2_0_']
    for mf in meta_fe:
        train_mf=np.hstack([train_mf,pd.read_pickle('stack/mf_3/X_mf'+mf+'random.pkl').reshape(len_train,-1)])
    print 'train_mf:',train_mf.shape

    meta_fe=['_lstm','_lstm_end2end','_lstm_att_sia']
    for mf in meta_fe:
        train_mf=np.hstack([train_mf,pd.read_pickle('stack/mf_3/X_mf'+mf+'.pkl').reshape(-1,1)])
    print 'train_mf:',train_mf.shape
    
    return train_mf


def get_mf_test():
##################meta feature#####################
    test_lgb1=pd.read_pickle('stack/mf_3/lgb_model3.test').reshape(-1,1)
    test_rf1=pd.read_pickle('stack/mf_3/rf_model1.test').reshape(-1,1)
    test_et1=pd.read_pickle('stack/mf_3/et_model1.test').reshape(-1,1)
    test_lr1=pd.read_pickle('stack/mf_3/lr_model1.test').reshape(-1,1)
    test_nn1=pd.read_pickle('stack/mf_3/nn_model1.test').reshape(-1,1)
    test_nn2=pd.read_pickle('stack/mf_3/nn_model2.test').reshape(-1,1)
    test_xgb1=pd.read_pickle('stack/mf_3/xgb_model1.test').reshape(-1,1)
    #test_lstm1=pd.read_pickle('stack/mf_3/lstm_model1.test').reshape(-1,1)
    #test_lstm2=pd.read_pickle('stack/mf_3/lstm_model2.test').reshape(-1,1)
    test_siamse=pd.read_pickle('stack/mf_3/test_siamse_match_mf.pkl').reshape(-1,1)
    test_bowen1=pd.read_csv('stack/mf_3/jasonnetwork.bs300.h512.att_concat.drop20.watt1.b0.l0.test.csv')['is_duplicate'].values.reshape(-1,1)
    test_bowen2=pd.read_csv('stack/mf_3/jasonnetwork.bs300.h512.att_concat.drop25.watt1.b0.l0.test.csv')['is_duplicate'].values.reshape(-1,1)
    test_bowen3=pd.read_csv('stack/mf_3/jasonnetwork.bs300.h512.att_concat.drop20.watt1.b0.l0.test2.csv')['is_duplicate'].values.reshape(-1,1)
    test_hhy=pd.read_pickle('stack/mf_3/dp_test_2_mf.pkl').reshape(-1,1)


    test_mf=np.hstack([test_lgb1,test_rf1,test_et1,
                        test_lr1,test_nn1,test_nn2,test_xgb1,test_siamse,test_bowen1,test_bowen2,test_bowen3,test_hhy])
    len_test=test_lgb1.shape[0]
    print 'test_mf:',test_mf.shape
    meta_fe=['_lr_tfidf_2_0_','_nn_3_0_','_et_3_0_','_rf_3_0_','_lgb_3_0_','_lsvc_tfidf_2_0_']
    for mf in meta_fe:
        test_mf=np.hstack([test_mf,pd.read_pickle('stack/mf_3/X_t_mf'+mf+'random.pkl').reshape(len_test,-1)])
    print 'test_mf:',test_mf.shape

    meta_fe=['_lstm','_lstm_end2end','_lstm_att_sia']
    for mf in meta_fe:
        test_mf=np.hstack([test_mf,pd.read_pickle('stack/mf_3/X_t_mf'+mf+'.pkl').reshape(-1,1)])
    print 'test_mf:',test_mf.shape
#meta_fe=range(40)
#for mf in meta_fe:
#   test_mf=np.hstack([test_mf,pd.read_pickle('stack/mf_3/mf_3_random_lgb/X_t_mf_lgb_random_'+str(mf)+'_random_r.pkl')])
#print 'test_mf:',test_mf.shape
    return test_mf
