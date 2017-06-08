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


def calc_edit_dist(neighs,qind):
    if qind not in index_q:
        return 5*[-1]
    q_str = index_q[qind]
    sim_fea = []
    for i in neighs:
        if i in index_q:
            nei_str = index_q[i]
            sim_fea.append(dist_utils._edit_dist(q_str, nei_str))
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

def calc_dice_dist(neighs,qind):
    if qind not in index_q:
        return 5*[-1]
    q_str = index_q[qind]
    sim_fea = []
    for i in neighs:
        if i in index_q:
            nei_str = index_q[i]
            s1 = set(q_str.lower().split())
            s2 = set(nei_str.lower().split())
            sim_fea.append(dist_utils._dice_dist(s1, s2))
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

def calc_dis_jarccard(neighs,qind):
    if qind not in index_q:
        return 5*[-1]
    q_str = index_q[qind]
    sim_fea = []
    for i in neighs:
        if i in index_q:
            nei_str = index_q[i]
            s1 = set(q_str.lower().split())
            s2 = set(nei_str.lower().split())
            sim_fea.append(dist_utils._jaccard_coef(s1, s2))
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

def calc_longest_match(neighs,qind):
    if qind not in index_q:
        return 5*[-1]
    q_str = index_q[qind]
    sim_fea = []
    for i in neighs:
        if i in index_q:
            nei_str = index_q[i]
            sim_fea.append(dist_utils._longest_match_size(q_str, nei_str))
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

def get_dist_feature(dist_fun):
    fea_q1 = []
    fea_q2 = []
    for i in tqdm(np.arange(data_all.shape[0])):
        q1,q2 = dd[i]
        if (q1 not in q_list)|(q2 not in q_list):
            fea_q1.append(5*[0])
            fea_q2.append(5*[0])
            continue
        nei_q1 = set(q_list[q1])
        fea_q1.append(dist_fun(nei_q1,q1))
        nei_q2 = set(q_list[q2])
        fea_q2.append(dist_fun(nei_q2,q2))

    fea_q1 = np.array(fea_q1)
    fea_q2 = np.array(fea_q2)
    all_fea = np.hstack([fea_q1,fea_q2])
    train_fea = all_fea[:train.shape[0]]
    test_fea = all_fea[train.shape[0]:]
    train_stats = np.vstack([train_fea.mean(axis=1),train_fea.max(axis=1),train_fea.min(axis=1),
                             train_fea.std(axis=1)]).T
    test_stats = np.vstack([test_fea.mean(axis=1),test_fea.max(axis=1),test_fea.min(axis=1),
                            test_fea.std(axis=1)]).T

    train_fea = np.hstack([train_fea,train_stats])
    test_fea = np.hstack([test_fea,test_stats])

    return train_fea,test_fea


train_fea,test_fea = get_dist_feature(calc_longest_match)
pd.to_pickle(train_fea,'../X_v2/train_neigh_long_match.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_long_match.pkl')


train_fea,test_fea = get_dist_feature(calc_dis_jarccard)
pd.to_pickle(train_fea,'../X_v2/train_neigh_jarccard.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_jarccard.pkl')

train_fea,test_fea = get_dist_feature(calc_dice_dist)
pd.to_pickle(train_fea,'../X_v2/train_neigh_dice.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_dice.pkl')


train_fea,test_fea = get_dist_feature(calc_edit_dist)
pd.to_pickle(train_fea,'../X_v2/train_neigh_edit.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_edit.pkl')


def get_dist_feature2(dist_fun):
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
        fea_q1.append(dist_fun(nei_q2,q1))
        fea_q2.append(dist_fun(nei_q1,q2))

    fea_q1 = np.array(fea_q1)
    fea_q2 = np.array(fea_q2)
    all_fea = np.hstack([fea_q1,fea_q2])
    train_fea = all_fea[:train.shape[0]]
    test_fea = all_fea[train.shape[0]:]
    train_stats = np.vstack([train_fea.mean(axis=1),train_fea.max(axis=1),train_fea.min(axis=1),
                             train_fea.std(axis=1)]).T
    test_stats = np.vstack([test_fea.mean(axis=1),test_fea.max(axis=1),test_fea.min(axis=1),
                            test_fea.std(axis=1)]).T

    train_fea = np.hstack([train_fea,train_stats])
    test_fea = np.hstack([test_fea,test_stats])

    return train_fea,test_fea

train_fea,test_fea = get_dist_feature2(calc_dis_jarccard)
pd.to_pickle(train_fea,'../X_v2/train_neigh_jarccard2.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_jarccard2.pkl')

train_fea,test_fea = get_dist_feature2(calc_dice_dist)
pd.to_pickle(train_fea,'../X_v2/train_neigh_dice2.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_dice2.pkl')

#rest
train_fea,test_fea = get_dist_feature2(calc_edit_dist)
pd.to_pickle(train_fea,'../X_v2/train_neigh_edit2.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_edit2.pkl')

train_fea,test_fea = get_dist_feature2(calc_longest_match)
pd.to_pickle(train_fea,'../X_v2/train_neigh_long2.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_long2.pkl')



def calc_dice_dist2(neighs,neighs2):
    sim_fea = []
    for i in neighs:
        for j in neighs2:
            if i == j: continue
            if (j in index_q) and (i in index_q):
                q_str = index_q[i]
                nei_str = index_q[j]
                s1 = set(q_str.lower().split())
                s2 = set(nei_str.lower().split())
                sim_fea.append(dist_utils._dice_dist(s1, s2))
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


def calc_dis_jarccard2(neighs,neighs2):
    sim_fea = []
    for i in neighs:
        for j in neighs2:
            if i==j:continue
            if (j in index_q) and (i in index_q):
                q_str = index_q[i]
                nei_str = index_q[j]
                s1 = set(q_str.lower().split())
                s2 = set(nei_str.lower().split())
                sim_fea.append(dist_utils._jaccard_coef(s1, s2))
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


# sps.spearmanr(train_fea,train['is_duplicate'])[0]
def get_dist_feature3(dist_fun):
    fea_q1 = []
    fea_q2 = []
    for i in tqdm(np.arange(data_all.shape[0])):
        q1,q2 = dd[i]
        if (q1 not in q_list)|(q2 not in q_list):
            fea_q1.append(5*[0])
            fea_q2.append(5*[0])
            continue
        ls1 = q_list[q1]
        ls2 = q_list[q2]
        if len(ls1)>30:
            ls1 = ls1[:30]
        if len(ls2)>30:
            ls2 = ls2[:30]
        nei_q1 = set(ls1)
        nei_q2 = set(ls2)
        fea_q1.append(dist_fun(nei_q1,nei_q1))
        fea_q2.append(dist_fun(nei_q2,nei_q2))

    fea_q1 = np.array(fea_q1)
    fea_q2 = np.array(fea_q2)
    all_fea = np.hstack([fea_q1,fea_q2])
    train_fea = all_fea[:train.shape[0]]
    test_fea = all_fea[train.shape[0]:]
    train_stats = np.vstack([train_fea.mean(axis=1),train_fea.max(axis=1),train_fea.min(axis=1),
                             train_fea.std(axis=1)]).T
    test_stats = np.vstack([test_fea.mean(axis=1),test_fea.max(axis=1),test_fea.min(axis=1),
                            test_fea.std(axis=1)]).T

    train_fea = np.hstack([train_fea,train_stats])
    test_fea = np.hstack([test_fea,test_stats])

    return train_fea,test_fea


train_fea,test_fea = get_dist_feature3(calc_dis_jarccard2)
train_fea = pd.DataFrame(train_fea).fillna(-1)
test_fea = pd.DataFrame(test_fea).fillna(-1)
train_fea = train_fea.values
test_fea = test_fea.values

# sps.spearmanr(train_fea,train['is_duplicate'])[0]

pd.to_pickle(train_fea,'../X_v2/train_neigh_jarccard3.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_jarccard3.pkl')





#neigh and neigh
def get_dist_feature4(dist_fun):
    all_fea = []
    for i in tqdm(np.arange(data_all.shape[0])):
        q1,q2 = dd[i]
        if (q1 not in q_list)|(q2 not in q_list):
            all_fea.append(5*[0])
            continue
        ls1 = q_list[q1]
        ls2 = q_list[q2]
        if len(ls1)>10:
            ls1 = ls1[:10]
        if len(ls2)>10:
            ls2 = ls2[:10]
        nei_q1 = set(ls1)
        nei_q2 = set(ls2)
        all_fea.append(dist_fun(nei_q1,nei_q2))

    all_fea = np.array(all_fea)
    train_fea = all_fea[:train.shape[0]]
    test_fea = all_fea[train.shape[0]:]

    return train_fea,test_fea

train_fea,test_fea = get_dist_feature4(calc_dis_jarccard2)
train_fea = pd.DataFrame(train_fea).fillna(-1)
test_fea = pd.DataFrame(test_fea).fillna(-1)
train_fea = train_fea.values
test_fea = test_fea.values

pd.to_pickle(train_fea,'../X_v2/train_neigh_jarccard4.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_jarccard4.pkl')


train_fea,test_fea = get_dist_feature4(calc_dice_dist2)
train_fea = pd.DataFrame(train_fea).fillna(-1)
test_fea = pd.DataFrame(test_fea).fillna(-1)
train_fea = train_fea.values
test_fea = test_fea.values


pd.to_pickle(train_fea,'../X_v2/train_neigh_dice4.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_dice4.pkl')