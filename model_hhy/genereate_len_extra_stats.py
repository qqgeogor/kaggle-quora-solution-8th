import numpy as np
import pandas as pd
import datetime
import operator
from collections import Counter
import matplotlib.pyplot as plt
from  tqdm import tqdm
import scipy.stats as sps

seed = 1024
np.random.seed(seed)

path = '../data/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
data_all = pd.concat([train, test])[['question1','question2']]


def generate_len_stats(data_all):
    data_all['caps_count_q1'] = data_all['question1'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
    data_all['caps_count_q2'] = data_all['question2'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
    data_all['diff_caps'] = data_all['caps_count_q1'] - data_all['caps_count_q2']

    data_all['len_char_q1'] = data_all['question1'].apply(lambda x: len(str(x).replace(' ', '')))
    data_all['len_char_q2'] = data_all['question2'].apply(lambda x: len(str(x).replace(' ', '')))
    data_all['diff_len_char'] = data_all['len_char_q1'] - data_all['len_char_q2']

    data_all['len_word_q1'] = data_all['question1'].apply(lambda x: len(str(x).split()))
    data_all['len_word_q2'] = data_all['question2'].apply(lambda x: len(str(x).split()))
    data_all['diff_len_word'] = data_all['len_word_q1'] - data_all['len_word_q2']

    data_all['avg_world_len1'] = data_all['len_char_q1'] / data_all['len_word_q1']
    data_all['avg_world_len2'] = data_all['len_char_q2'] / data_all['len_word_q2']
    data_all['diff_avg_word'] = data_all['avg_world_len1'] - data_all['avg_world_len2']

    data_all.drop(['question1','question2'],axis=1,inplace=1)
    return data_all[['caps_count_q1','caps_count_q2','diff_caps','diff_len_char','avg_world_len1',
                     'avg_world_len2','diff_avg_word']]


fea = generate_len_stats(data_all.copy())

train_len = fea[:train.shape[0]]
test_len = fea[train.shape[0]:]

# sps.spearmanr(train_len,train['is_duplicate'])[0]

pd.to_pickle(train_len,'../X_v2/train_len.pkl')
pd.to_pickle(test_len,'../X_v2/test_len.pkl')