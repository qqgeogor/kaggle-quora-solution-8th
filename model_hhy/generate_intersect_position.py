import numpy as np
import pandas as pd
from .utils import nlp_utils,dist_utils,split_data
seed = 1024
np.random.seed(seed)


path = '../data/'
train = pd.read_pickle(path+'clean_train.pickle')
test = pd.read_pickle(path+'clean_test.pickle')
data_all = pd.concat([train, test])

#position
def _inter_pos_list(q1,q2):
    obs = nlp_utils._tokenize(q1)
    target = nlp_utils._tokenize(q2)
    pos_list = [0]
    if len(obs) != 0:
        pos_list = [abs(i-target.find(o)) for i,o in enumerate(obs, start=1) if o in target]
        if len(pos_list) == 0:
            pos_list = [0]
    return pos_list

def _linear_aggregate(x):
    agg_mode = ["mean","max", "min", "median",'std']
    aggregator = [None if m == "" else getattr(np, m) for m in agg_mode]
    N = x.shape[0]
    res = np.zeros((N, len(aggregator)), dtype=float)
    for m, aggregator in enumerate(aggregator):
        for i in range(N):
            try:
                s = aggregator(x[i])
            except:
                s = -1
            res[i, m] = s
    return res

data_all['q1_q2_inter_pos'] = data_all.apply(lambda x:(_inter_pos_list(x['clean_question1'],x['clean_question2'])),axis=1)
data_all['q2_q1_inter_pos'] = data_all.apply(lambda x:(_inter_pos_list(x['clean_question2'],x['clean_question1'])),axis=1)
#aggregate
q1_q2_pos_agg = _linear_aggregate(data_all['q1_q2_inter_pos'].values)
q2_q1_pos_agg = _linear_aggregate(data_all['q2_q1_inter_pos'].values)
q_pos_agg = np.hstack([q1_q2_pos_agg,q1_q2_pos_agg])

train_pos = q_pos_agg[:train.shape[0]]
test_pos = q_pos_agg[train.shape[0]:]
pd.to_pickle(train_pos,'../X_v2/train_pos.pkl')
test_x = split_data.split_test(test_pos)
for i in range(6):
    pd.to_pickle(test_x[i],'../X_v2/test_pos{0}.pkl'.format(i))

