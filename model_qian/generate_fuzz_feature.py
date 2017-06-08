"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur
"""

import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')
from config import path


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


train = pd.read_csv(path+"train.csv").astype(str)
test = pd.read_csv(path+"test.csv").astype(str)


train['fuzz_qratio'] = train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
train['fuzz_WRatio'] = train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
train['fuzz_partial_ratio'] = train.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
train['fuzz_partial_token_set_ratio'] = train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
train['fuzz_partial_token_sort_ratio'] = train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
train['fuzz_token_set_ratio'] = train.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
train['fuzz_token_sort_ratio'] = train.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

test['fuzz_qratio'] = test.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
test['fuzz_WRatio'] = test.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
test['fuzz_partial_ratio'] = test.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
test['fuzz_partial_token_set_ratio'] = test.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
test['fuzz_partial_token_sort_ratio'] = test.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
test['fuzz_token_set_ratio'] = test.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
test['fuzz_token_sort_ratio'] = test.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

fuzz = [
'fuzz_qratio',
'fuzz_WRatio',
'fuzz_partial_ratio',
'fuzz_partial_token_set_ratio',
'fuzz_partial_token_sort_ratio',
'fuzz_token_set_ratio',
'fuzz_token_sort_ratio',
]
train[fuzz].to_csv(path+'train_fuzz_feature.csv', index=False)
test[fuzz].to_csv(path+'test_fuzz_feature.csv', index=False)
