import string
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

np.random.seed(1024)
path = '../data/'

MISSING_VALUE_NUMERIC = -1.
TRAIN_DATA_FILE = path+'train_clean.pkl'
TEST_DATA_FILE = path+ 'test_clean.pkl'
ft = ['clean_question1','clean_question2']

train_data = pd.read_pickle(TRAIN_DATA_FILE)[ft]
test_data = pd.read_pickle(TEST_DATA_FILE)[ft]
data_all = np.vstack([train_data,test_data])

def basic_token(text):
    return list(text.strip().lower().split())


class WordNet_Similarity():
    def __init__(self, metric="path",double_aggregator=False):
        """
        :param metric: path lch and wup metric
        :param double_aggregator:
        """
        self.metric = metric
        self.aggregation_mode_prev = ['max','mean','median']#["mean", "max", "median"]
        self.aggregation_mode =["mean", "std", "max", "min", "median"]
        self.aggregator = [None if m == "" else getattr(np, m) for m in self.aggregation_mode]
        self.aggregator_prev = [None if m == "" else getattr(np, m) for m in self.aggregation_mode_prev]
        self.double_aggregator = double_aggregator
        if self.metric == "path":# scene shortest path
            self.metric_func = lambda syn1, syn2: wn.path_similarity(syn1, syn2)
        elif self.metric == "lch":
            self.metric_func = lambda syn1, syn2: wn.lch_similarity(syn1, syn2)
        elif self.metric == "wup":# words' depth and ancestor depth + shortest path
            self.metric_func = lambda syn1, syn2: wn.wup_similarity(syn1, syn2)
        else:
            raise (ValueError("Wrong similarity metric: %s, should be one of path/lch/wup." % self.metric))

    def _maximum_similarity_for_two_synset_list(self, syn_list1, syn_list2):
        s = 0.
        if syn_list1 and syn_list2:
            for syn1 in syn_list1:
                for syn2 in syn_list2:
                    try:
                        _s = self.metric_func(syn1, syn2)
                    except:
                        _s = MISSING_VALUE_NUMERIC
                    if _s and _s > s:
                        s = _s
        return s

    def transform_one(self, obs, target):
        obs_tokens = basic_token(obs)
        target_tokens = basic_token(target)
        obs_synset_list = [wn.synsets(obs_token) for obs_token in obs_tokens]
        target_synset_list = [wn.synsets(target_token) for target_token in target_tokens]
        val_list = []
        for obs_synset in obs_synset_list:
            _val_list = []
            for target_synset in target_synset_list:#与obs_synset同义词中metric 最大的
                _s = self._maximum_similarity_for_two_synset_list(obs_synset, target_synset)
                _val_list.append(_s)
            if len(_val_list) == 0:
                _val_list = [MISSING_VALUE_NUMERIC]
            val_list.append(_val_list)
        if len(val_list) == 0:
            val_list = [[MISSING_VALUE_NUMERIC]]
        return val_list

    def fit_transform(self,data_all):
        """
        :param data_all: the numpy fields dim 0 is obs dim1 is target
        :return: sim
        """
        score = list(map(self.transform_one, data_all[:, 0], data_all[:, 1]))
        self.N = data_all.shape[0]
        if self.double_aggregator:
            res = np.zeros((self.N, len(self.aggregator_prev) * len(self.aggregator)), dtype=float)
            for m, aggregator_prev in enumerate(self.aggregator_prev):
                for n, aggregator in enumerate(self.aggregator):
                    idx = m * len(self.aggregator) + n
                    for i in range(self.N):
                        # process in a safer way
                        try:
                            tmp = []
                            for l in score[i]:
                                try:
                                    s = aggregator_prev(l)
                                except:
                                    s = MISSING_VALUE_NUMERIC
                                tmp.append(s)
                        except:
                            tmp = [MISSING_VALUE_NUMERIC]
                        try:
                            s = aggregator(tmp)
                        except:
                            s = MISSING_VALUE_NUMERIC
                        res[i, idx] = s
        else:
            res = np.zeros((self.N, len(self.aggregator)), dtype=float)
            for m, aggregator in enumerate(self.aggregator):
                for i in range(self.N):
                    # process in a safer way
                    try:
                        s = aggregator(score[i])
                    except:
                        s = MISSING_VALUE_NUMERIC
                    res[i, m] = s
        return res

def split_data(data,size=100000):
    data_x = []
    N = data.shape[0]
    stride = int(N/size)+1
    for i in range(stride):
        idx = i*size
        idx2 = (i+1)*size
        if i==(stride-1):
            data_x.append(data[idx:])
        else:
            data_x.append(data[idx:idx2])
    return data_x

data_x = split_data(data_all)

path = '../X_v2/wordnet/'
WN = WordNet_Similarity(double_aggregator=True)
for i in range(data_x.shape[0]):
    print('iter ',i)
    WordNetSim = WN.fit_transform(data_x[i])
    pd.to_pickle(WordNetSim,path+'wordnet_sim{0}.pkl'.format(i))
    print('end ',i)


train_wordnet = WordNetSim[:train_data.shape[0]]
test_wordnet = WordNetSim[train_data.shape[0]:]

pd.to_pickle(train_wordnet,path+'wordnet_sim_train.pkl')
pd.to_pickle(test_wordnet,path+'wordnet_sim_test.pkl')