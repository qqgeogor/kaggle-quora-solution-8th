import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import dist_utils,split_data,nlp_utils
import scipy.stats as sps
from nltk.corpus import stopwords


seed = 1024
np.random.seed(seed)

path = '../data/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
data_all = pd.concat([train, test])[['question1','question2']]
stops = set(stopwords.words("english"))

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



def wc_diff_unique_stop(str1,str2):
    return abs(len([x for x in set(str1) if x not in stops]) - len([x for x in set(str2) if x not in stops]))

def char_diff(str1,str2):
    return abs(len(''.join(str1)) - len(''.join(str2)))

def total_unique_words(str1,str2):
    return len(set(str1).union(str2))

def gram_2_diff(str1,str2):
    q1_list = str1.lower().split()
    q2_list = str2.lower().split()
    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])
    shared_2gram = q1_2gram.intersection(q2_2gram)
    return len(shared_2gram)


def calc_stats_words(neighs,qind,fun):
    if qind not in index_q:
        return 5*[-1]
    q_str = index_q[qind]
    sim_fea = []
    for i in neighs:
        if i in index_q:
            nei_str = index_q[i]
            sim_fea.append(fun(q_str, nei_str))
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


def get_dist_feature(stats_fun):
    fea_q1 = []
    fea_q2 = []
    for i in tqdm(np.arange(data_all.shape[0])):
        q1,q2 = dd[i]
        if (q1 not in q_list)|(q2 not in q_list):
            fea_q1.append(5*[0])
            fea_q2.append(5*[0])
            continue
        nei_q1 = set(q_list[q1])
        fea_q1.append(calc_stats_words(nei_q1,q1,stats_fun))
        nei_q2 = set(q_list[q2])
        fea_q2.append(calc_stats_words(nei_q2,q2,stats_fun))
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



train_fea,test_fea = get_dist_feature(total_unique_words)

# sps.spearmanr(train_fea,train['is_duplicate'])[0]
pd.to_pickle(train_fea,'../X_v2/train_neigh_unique_words.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_unique_words.pkl')


train_fea,test_fea = get_dist_feature(char_diff)
# sps.spearmanr(train_fea,train['is_duplicate'])[0]
pd.to_pickle(train_fea,'../X_v2/train_neigh_char_dif.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_char_dif.pkl')


train_fea,test_fea = get_dist_feature(gram_2_diff)
# sps.spearmanr(train_fea,train['is_duplicate'])[0]
pd.to_pickle(train_fea,'../X_v2/train_neigh_2gram.pkl')
pd.to_pickle(test_fea,'../X_v2/test_neigh_2gram.pkl')

# sps.spearmanr(train_fea,train['is_duplicate'])[0]








