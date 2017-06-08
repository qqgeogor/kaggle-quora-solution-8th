import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import dist_utils,split_data,nlp_utils
import scipy.stats as sps

seed = 1024
np.random.seed(seed)

path = '../data/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
data_all = pd.concat([train, test])[['question1','question2']]

vector_size = 100
glove_dir =   '../data/glove.6B.{0}d.txt'.format(vector_size)
Embedd_model = nlp_utils._get_embedd_Index(glove_dir)

#dup index
q_all = pd.DataFrame(np.hstack([train['question1'], test['question1'],
                   train['question2'], test['question2']]), columns=['question'])
q_all = pd.DataFrame(q_all.question.value_counts()).reset_index()

q_num = dict(q_all.values)
q_index = {}
for i,key in enumerate(q_num.keys()):
    q_index[key] = i
index_q = {}
for i,key in enumerate(q_index.keys()):
    index_q[i] = key
data_all['q1_index'] = data_all['question1'].map(q_index)
data_all['q2_index'] = data_all['question2'].map(q_index)


#link edges
q_list = {}
dd = data_all[['q1_index','q2_index']].values
for i in tqdm(np.arange(data_all.shape[0])):
#for i in np.arange(dd.shape[0]):
    q1,q2=dd[i]
    if q_list.setdefault(q1,[q2])!=[q2]:
        q_list[q1].append(q2)
    if q_list.setdefault(q2,[q1])!=[q1]:
        q_list[q2].append(q1)


#get neighbor sim
#qid embedd
def _wrapper_qindex_embedd(qind):
    if qind not in index_q:
        return np.zeros(vector_size)
    q = index_q[qind]
    wl = str(q).lower().split()
    centroid = np.zeros(vector_size)
    k = 0
    for w in wl:
        if w in Embedd_model:
            centroid+= Embedd_model[w]
            k+=1
    if k==0:
        return np.zeros(vector_size)
    centroid/=k
    return centroid
#neighs and q
def calc_sim_q(neighs,qind):
    q_emb = _wrapper_qindex_embedd(qind)
    sim_fea = []
    for i in neighs:
        nei_emb = _wrapper_qindex_embedd(i)
        sim_fea.append(dist_utils._calc_similarity(q_emb,nei_emb))
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]
    score = []
    for n, agg in enumerate(aggregator):
        if len(sim_fea) == 0:
            s = -1
        try:
            s = agg(sim_fea)
        except:
            s = -1
        score.append(s)
    return score

#1重的2度stats
def calc_sim_q_2(neighs,qind):
    q_emb = _wrapper_qindex_embedd(qind)
    sim_fea = []
    for i in neighs:
        nei_emb = _wrapper_qindex_embedd(i)
        sim_fea.append(dist_utils._calc_similarity(q_emb,nei_emb))
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    prev_aggregation_mode = ['mean','max','min','median','std']
    aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]
    prev_aggregator = [None if m == "" else getattr(np, m) for m in prev_aggregation_mode]
    score = []
    for n, agg in enumerate(aggregator):
        tmp = []
        if len(sim_fea) == 0:
            tmp = 5*[-1]
        else:
            for m,pagg in enumerate(prev_aggregator):
                try:
                    s = pagg(sim_fea)
                except:
                    s = -1
                tmp.append(s)
        try:
            s_2 = agg(tmp)
        except:
            s_2 = -1
        score.append(s_2)
    return score

def calc_vdif_q(neighs,qind):
    q_emb = _wrapper_qindex_embedd(qind)
    sim_fea = []
    for i in neighs:
        nei_emb = _wrapper_qindex_embedd(i)
        sim_fea.append(dist_utils._vdiff(q_emb, nei_emb))
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]
    score = []
    for n, agg in enumerate(aggregator):
        if len(sim_fea) == 0:
            s = -1
        try:
            s = agg(sim_fea)
        except:
            s = -1
        score.append(s)
    return score


# q1 q1_neighr   q2 q2_neighr  cv:0.0015
fea_q1 = []
fea_q2 = []
for i in tqdm(np.arange(data_all.shape[0])):
    q1,q2 = dd[i]
    if (q1 not in q_list)|(q2 not in q_list):
        fea_q1.append(5*[0])
        fea_q2.append(5*[0])
        continue
    nei_q1 = set(q_list[q1])
    fea_q1.append(calc_sim_q(nei_q1,q1))
    nei_q2 = set(q_list[q2])
    fea_q2.append(calc_sim_q(nei_q2,q2))

fea_q1 = np.array(fea_q1)
fea_q2 = np.array(fea_q2)

all_fea = np.hstack([fea_q1,fea_q2])
train_fea = all_fea[:train.shape[0]]
test_fea = all_fea[train.shape[0]:]

pd.to_pickle(train_fea,'../X_v2/train_neigh_sim.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_sim.pkl')


# stats_fea_q1 = np.vstack([fea_q1.mean(axis=1),fea_q1.max(axis=1),fea_q1.min(axis=1),fea_q1.std(axis=1)]).T
# stats_fea_q2 = np.vstack([fea_q2.mean(axis=1),fea_q2.max(axis=1),fea_q2.min(axis=1),fea_q2.std(axis=1)]).T

train_stats = np.vstack([train_fea.mean(axis=1),train_fea.max(axis=1),train_fea.min(axis=1),
                         train_fea.std(axis=1)]).T
test_stats = np.vstack([test_fea.mean(axis=1),test_fea.max(axis=1),test_fea.min(axis=1),
                        test_fea.std(axis=1)]).T
pd.to_pickle(train_stats,'../X_v2/train_neigh_sim_stats.pkl')
pd.to_pickle(test_stats,'../X_v2/test_neigh_sim_stats.pkl')




fea_q1 = []
fea_q2 = []
for i in tqdm(np.arange(data_all.shape[0])):
    q1,q2 = dd[i]
    if (q1 not in q_list)|(q2 not in q_list):
        fea_q1.append(5*[0])
        fea_q2.append(5*[0])
        continue
    nei_q1 = set(q_list[q1])
    fea_q1.append(calc_vdif_q(nei_q1,q1))
    nei_q2 = set(q_list[q2])
    fea_q2.append(calc_vdif_q(nei_q2,q2))

# fea_q1 = np.array(fea_q1)
# fea_q2 = np.array(fea_q2)
#
# all_fea = np.hstack([fea_q1,fea_q2])
# train_fea = all_fea[:train.shape[0]]
# test_fea = all_fea[train.shape[0]:]

def get_fea_add_stats(q1_fea,q2_fea):
    q1_fea = np.array(q1_fea)
    q2_fea = np.array(q2_fea)
    all_fea = np.hstack([q1_fea, q2_fea])
    train_fea = all_fea[:train.shape[0]]
    test_fea = all_fea[train.shape[0]:]
    train_stats = np.vstack([train_fea.mean(axis=1), train_fea.max(axis=1), train_fea.min(axis=1),
                             train_fea.std(axis=1)]).T
    test_stats = np.vstack([test_fea.mean(axis=1), test_fea.max(axis=1), test_fea.min(axis=1),
                            test_fea.std(axis=1)]).T

    train_fea = np.hstack([train_fea, train_stats])
    test_fea = np.hstack([test_fea, test_stats])

    return train_fea,test_fea

train_fea,test_fea = get_fea_add_stats(fea_q1,fea_q2)

pd.to_pickle(train_fea,'../X_v2/train_neigh_vdif.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_vdif.pkl')

# sps.spearmanr(train_fea,train['is_duplicate'])[0]


fea_q1 = []
fea_q2 = []
for i in tqdm(np.arange(data_all.shape[0])):
    q1,q2 = dd[i]
    if (q1 not in q_list)|(q2 not in q_list):
        fea_q1.append(5*[0])
        fea_q2.append(5*[0])
        continue
    nei_q1 = set(q_list[q1])
    fea_q1.append(calc_sim_q_2(nei_q1,q1))
    nei_q2 = set(q_list[q2])
    fea_q2.append(calc_sim_q_2(nei_q2,q2))

train_fea,test_fea = get_fea_add_stats(fea_q1,fea_q2)
pd.to_pickle(train_fea,'../X_v2/train_neigh_sim_2.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_sim_2.pkl')



# q1 q2_neighr   q2 q1_neighr  cv:0.0005
fea_q1 = []
fea_q2 = []
for i in tqdm(np.arange(data_all.shape[0])):
    q1,q2 = dd[i]
    if (q1 not in q_list)|(q2 not in q_list):
        fea_q1.append(5*[0])
        fea_q2.append(5*[0])
        continue
    nei_q1 = set(q_list[q1])
    nei_q2 = set(q_list[q2])
    fea_q1.append(calc_sim_q(nei_q2,q1))
    fea_q2.append(calc_sim_q(nei_q1,q2))

all_fea = np.hstack([fea_q1,fea_q2])
train_fea = all_fea[:train.shape[0]]
test_fea = all_fea[train.shape[0]:]

pd.to_pickle(train_fea,'../X_v2/train_neigh_sim2.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_sim2.pkl')

train_stats = np.vstack([train_fea.mean(axis=1),train_fea.max(axis=1),train_fea.min(axis=1),
                         train_fea.std(axis=1)]).T
test_stats = np.vstack([test_fea.mean(axis=1),test_fea.max(axis=1),test_fea.min(axis=1),
                        test_fea.std(axis=1)]).T
pd.to_pickle(train_stats,'../X_v2/train_neigh_sim_stats2.pkl')
pd.to_pickle(test_stats,'../X_v2/test_neigh_sim_stats2.pkl')
#sps.spearmanr(fea_q1[:,0],train['is_duplicate'])[0]


#neighs and neighs
def calc_sim_nei(neighs1,neighs2,index_emb):
    sim_fea = []
    if len(neighs1)<1:
        return 5*[-1]
    if len(neighs2)<1:
        return 5*[-1]

    for i in neighs1:
        for j in neighs2:
            if (i in index_emb) and (j in index_emb):
                nei_emb = index_emb[i]
                nei_emb2 = index_emb[j]
                sim_fea.append(dist_utils._calc_similarity(nei_emb, nei_emb2))
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]
    score = []
    for n, agg in enumerate(aggregator):
        if len(sim_fea) == 0:
            s = -1
        try:
            s = agg(sim_fea)
        except:
            s = -1
        score.append(s)
    return score

#neighs self sim
def calc_sim_self(neighs,index_emb):
    sim_fea = []
    if len(neighs)<=1:
        return 5*[-1]
    for i1,i in enumerate(neighs):
        for i2,j in enumerate(neighs):
            if i1==i2:
                continue
            if (i in index_emb) and (j in index_emb):
                nei_emb = index_emb[i]
                nei_emb2 = index_emb[j]
                sim_fea.append(dist_utils._calc_similarity(nei_emb, nei_emb2))
            else:
                sim_fea.append(-1)
    aggregation_mode = ["mean", "std", "max", "min","median"]
    aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]
    score = []
    for n, agg in enumerate(aggregator):
        s = agg(sim_fea)
        score.append(s)
    return score


#q1_neigh q1_neigh self compare
#self compare
#q1_neigh  q2_neigh  cv:0.0002
index_emb = {}
for key in tqdm(index_q.keys()):
    index_emb[key] = _wrapper_qindex_embedd(key)


q1_fea = []
q2_fea = []
for i in tqdm(np.arange(data_all.shape[0])):
    q1,q2 = dd[i]
    if (q1 not in q_list)|(q2 not in q_list):
        q1_fea.append(5*[-1])
        q2_fea.append(5 * [0])
        continue
    ls = q_list[q1]
    ls2 = q_list[q2]
    if len(ls)>20:
        ls = ls[:20]
    if len(ls2)>20:
        ls2 = ls2[:20]
    nei_q1 = set(ls)
    nei_q2 = set(ls2)
    q1_fea.append(calc_sim_self(nei_q1,index_emb))
    q2_fea.append(calc_sim_self(nei_q2,index_emb))

q1_fea = np.array(q1_fea)
q2_fea = np.array(q2_fea)
all_fea = np.hstack([q1_fea,q2_fea])
all_fea = pd.DataFrame(all_fea).fillna(-1)
all_fea = all_fea.values
train_fea = all_fea[:train.shape[0]]
test_fea = all_fea[train.shape[0]:]
train_stats = np.vstack([train_fea.mean(axis=1),train_fea.max(axis=1),train_fea.min(axis=1),
                         train_fea.std(axis=1)]).T
test_stats = np.vstack([test_fea.mean(axis=1),test_fea.max(axis=1),test_fea.min(axis=1),
                        test_fea.std(axis=1)]).T

train_fea = np.hstack([train_fea,train_stats])
test_fea = np.hstack([test_fea,test_stats])

sps.spearmanr(train_fea,train['is_duplicate'])[0]

pd.to_pickle(train_fea,'../X_v2/train_neigh_sim3.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_sim3.pkl')



all_fea = []
for i in tqdm(np.arange(data_all.shape[0])):
    q1,q2 = dd[i]
    if (q1 not in q_list)|(q2 not in q_list):
        all_fea.append(5*[0])
        continue
    ls = q_list[q1]
    ls2 = q_list[q2]
    if len(ls)>30:
        ls = ls[:30]
    if len(ls2)>30:
        ls2 = ls2[:30]
    nei_q1 = set(ls)
    nei_q2 = set(ls2)
    self_sim = calc_sim_nei(nei_q1,nei_q2,index_emb)
    all_fea.append(self_sim)

all_fea = np.array(all_fea)
train_fea = all_fea[:train.shape[0]]
test_fea = all_fea[train.shape[0]:]
train_stats = np.vstack([train_fea.mean(axis=1),train_fea.max(axis=1),train_fea.min(axis=1),
                         train_fea.std(axis=1)]).T
test_stats = np.vstack([test_fea.mean(axis=1),test_fea.max(axis=1),test_fea.min(axis=1),
                        test_fea.std(axis=1)]).T

train_fea = np.hstack([train_fea,train_stats])
test_fea = np.hstack([test_fea,test_stats])

pd.to_pickle(train_fea,'../X_v2/train_neigh_sim4.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_sim4.pkl')