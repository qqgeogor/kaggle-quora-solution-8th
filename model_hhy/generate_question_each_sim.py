import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .utils import np_utils,nlp_utils,dist_utils,split_data
from tqdm import tqdm


#embedd path
path = '../data/'
vector_size = 100
glove_dir = path+'glove.6B.{0}d.txt'.format(vector_size)
Embedd_model = nlp_utils._get_embedd_Index(glove_dir)


train = pd.read_pickle(path+'train_final_clean.pkl')
test = pd.read_pickle(path+'test_final_clean.pkl')
feats= ['question1','question2']
train_value = train[feats].values

data_all = pd.concat([train,test])[feats].values

MISSING_VALUE_NUMERIC = -1


def w2w_sim(obs, target):
    val_list = []
    obs_tokens = nlp_utils._tokenize(obs)
    target_tokens = nlp_utils._tokenize(target)
    for i,obs_token in enumerate(obs_tokens):
        _val_list = []
        if obs_token in Embedd_model:
            for j,target_token in enumerate(target_tokens):
                if i==j:continue
                if target_token in Embedd_model:
                    sim = dist_utils._calc_similarity(Embedd_model[obs_token], Embedd_model[target_token])
                    _val_list.append(sim)
        if len(_val_list) == 0:
            _val_list = [MISSING_VALUE_NUMERIC]
        val_list.append(_val_list)
    if len(val_list) == 0:
        val_list = [[MISSING_VALUE_NUMERIC]]
    return val_list

def _aggregate_w2w_sim(score):
    aggregation_mode_prev = ['max', 'mean', 'min', 'median']  # ["mean", "max", "median"]
    aggregation_mode = ["mean", "std", "max", "median"]
    aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]
    aggregator_prev = [None if m == "" else getattr(np, m) for m in aggregation_mode_prev]
    N = len(score)
    fea_sim = np.zeros((N, len(aggregator_prev) * len(aggregator)), dtype=float)
    for it in tqdm(np.arange(N),miniters=1000):
        for m, agg_pre in enumerate(aggregator_prev):
            for n, agg in enumerate(aggregator):
                idx = m * len(aggregator) + n
                if len(score)==0:
                    fea_sim[it,idx] = -1
                    continue
                # process in a safer way
                try:
                    tmp = []
                    for l in score[it]:
                        try:
                            s = agg_pre(l)
                        except:
                            s = -1
                        tmp.append(s)
                except:
                    tmp = [-1]
                try:
                    s = agg(tmp)
                except:
                    s = -1
                fea_sim[it, idx] = s
    return fea_sim

w2w_q1 = []
w2w_q2 = []
for it in tqdm(np.arange(data_all.shape[0]),miniters=1000):
    w2w_q1.append(w2w_sim(data_all[it][0],data_all[it][0]))
    w2w_q2.append(w2w_sim(data_all[it][1],data_all[it][1]))

fea_q1 = _aggregate_w2w_sim(w2w_q1)
fea_q2 = _aggregate_w2w_sim(w2w_q2)

fea_all = np.hstack([fea_q1,fea_q2])
train_fea = fea_all[:train.shape[0]]
test_fea = fea_all[train_fea[0]:]

pd.to_pickle(train_fea,'../X_v2/train_self_sim.pkl')


# import scipy.stats as sps
# y_train = pd.read_csv('../data/train.csv')['is_duplicate']
# sps.spearmanr(fea_q1[:,0],y_train[:500])



