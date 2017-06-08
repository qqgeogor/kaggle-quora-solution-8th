import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import dist_utils,split_data,nlp_utils
import scipy.stats as sps
import nltk


seed = 1024
np.random.seed(seed)

path = '../data/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
data_all = pd.concat([train, test])[['question1','question2']]



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


def _mv_pos_max(q1,q2):
    obs = nlp_utils._tokenize(q1)
    target = nlp_utils._tokenize(q2)
    pos_list = []
    if len(obs) != 0:
        pos_list = [abs(i-target.index(o)) for i,o in enumerate(obs) if o in target]
        if len(pos_list) == 0:
            pos_list.append(0)
    return np.max(pos_list)

def _mv_pos_mean(q1,q2):
    obs = nlp_utils._tokenize(q1)
    target = nlp_utils._tokenize(q2)
    pos_list = []
    if len(obs) != 0:
        pos_list = [abs(i-target.index(o)) for i,o in enumerate(obs) if o in target]
        if len(pos_list) == 0:
            pos_list.append(0)
    return np.mean(pos_list)

def calc_neigh_pos_mv(neighs,qind):
    if qind not in index_q:
        return 5*[-1]
    q_str = index_q[qind]
    sim_fea = []
    for i in neighs:
        if i in index_q:
            nei_str = index_q[i]
            sim_fea.append(_mv_pos_max(q_str, nei_str))
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


fea_q1 = []
fea_q2 = []
for i in tqdm(np.arange(data_all.shape[0])):
    q1,q2 = dd[i]
    if (q1 not in q_list)|(q2 not in q_list):
        fea_q1.append(5*[0])
        fea_q2.append(5*[0])
        continue
    nei_q1 = set(q_list[q1])
    fea_q1.append(calc_neigh_pos_mv(nei_q1,q1))
    nei_q2 = set(q_list[q2])
    fea_q2.append(calc_neigh_pos_mv(nei_q2,q2))

fea_q1 = np.array(fea_q1)
fea_q2 = np.array(fea_q2)
all_fea = np.hstack([fea_q1, fea_q2])
train_fea = all_fea[:train.shape[0]]
test_fea = all_fea[train.shape[0]:]
train_stats = np.vstack([train_fea.mean(axis=1), train_fea.max(axis=1), train_fea.min(axis=1),
                         train_fea.std(axis=1)]).T
test_stats = np.vstack([test_fea.mean(axis=1), test_fea.max(axis=1), test_fea.min(axis=1),
                        test_fea.std(axis=1)]).T

train_fea = np.hstack([train_fea, train_stats])
test_fea = np.hstack([test_fea, test_stats])

# sps.spearmanr(train_fea,train['is_duplicate'])[0]

pd.to_pickle(train_fea,'../X_v2/train_neigh_pos_mv.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_pos_mv.pkl')



def calc_neigh_pos_mv(neighs,qind):
    if qind not in index_q:
        return 5*[-1]
    q_str = index_q[qind]
    sim_fea = []
    for i in neighs:
        if i in index_q:
            nei_str = index_q[i]
            sim_fea.append(_mv_pos_max(q_str, nei_str))
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
    fea_q1.append(calc_neigh_pos_mv(nei_q2,q1))
    fea_q2.append(calc_neigh_pos_mv(nei_q1,q2))

fea_q1 = np.array(fea_q1)
fea_q2 = np.array(fea_q2)
all_fea = np.hstack([fea_q1, fea_q2])
train_fea = all_fea[:train.shape[0]]
test_fea = all_fea[train.shape[0]:]
train_stats = np.vstack([train_fea.mean(axis=1), train_fea.max(axis=1), train_fea.min(axis=1),
                         train_fea.std(axis=1)]).T
test_stats = np.vstack([test_fea.mean(axis=1), test_fea.max(axis=1), test_fea.min(axis=1),
                        test_fea.std(axis=1)]).T

train_fea = np.hstack([train_fea, train_stats])
test_fea = np.hstack([test_fea, test_stats])

# sps.spearmanr(train_fea,train['is_duplicate'])[0]

pd.to_pickle(train_fea,'../X_v2/train_neigh_pos_mv2.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_pos_mv2.pkl')


