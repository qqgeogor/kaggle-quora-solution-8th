# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text

from collections import Counter
from itertools import izip
import operator

import logging
import warnings
import pickle


__authors__ = ['bowenwu']


warnings.filterwarnings(action='ignore')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()


def stem_str(sen):
    sen = text.re.sub('[^a-zA-Z0-9]', ' ', sen)
    sen = nltk.word_tokenize(sen.lower())
    sen = map(snowball_stemmer.stem, sen)
    sen = map(wordnet_lemmatizer.lemmatize, sen)
    return (' '.join(sen)).lower()


def preprocess(sentences):
    new_sens = []
    for sen in sentences:
        new_sen = sen.replace(' i ', ' <U> ')
        new_sen = new_sen.replace(' you ', ' <U> ')
        new_sen = new_sen.replace(' he ', ' <U> ')
        new_sen = new_sen.replace(' she ', ' <U> ')
        new_sen = new_sen.replace(' him ', ' <U> ')
        new_sen = new_sen.replace(' me ', ' <U> ')
        new_sen = new_sen.replace(' they ', ' <U> ')
        new_sen = new_sen.replace(' them ', ' <U> ')
        new_sen = new_sen.replace(' us ', ' <U> ')
        new_sen = new_sen.replace(' our ', ' <U> ')
        new_sen = new_sen.replace(' my ', ' <U> ')
        new_sen = new_sen.replace(' your ', ' <U> ')
        new_sen = new_sen.replace(' his ', ' <U> ')
        new_sen = new_sen.replace(' her ', ' <U> ')
        new_sen = new_sen.replace(' their ', ' <U> ')
        new_sen = new_sen.replace(' s ', ' <is> ')
        new_sen = new_sen.replace(' is ', ' <is> ')
        new_sen = new_sen.replace(' am ', ' <is> ')
        new_sen = new_sen.replace(' are ', ' <is> ')
        new_sen = new_sen.replace(' don t ', ' do not ')
        new_sens.append(new_sen)
    return new_sens


def mining_pattern_withkprefix(sentences, firstk=1, topk=100, prefixs={}):
    logger.info('Mining pattern for first %d among %d', firstk, len(sentences))
    counts = Counter()
    for sen in sentences:
        words = tuple(sen.split()[:firstk])
        if prefixs and words[:-1] not in prefixs:
            continue
        counts[words] += 1
    lists = sorted(counts.iteritems(), key=operator.itemgetter(1), reverse=True)
    new_prefixs = {}
    for words, count in lists[:topk]:
        new_prefixs[words] = count
    logger.info('\tfind %d', len(new_prefixs))
    return new_prefixs


def mining_pattern(sentences, firstk, topk, fout, less2backup=3):
    prefixs = {}
    new_prefixs = {}
    for i in xrange(1, firstk + 1):
        new_prefixs = mining_pattern_withkprefix(sentences, firstk=i, topk=topk, prefixs=new_prefixs)
        if i >= less2backup:
            for words, count in new_prefixs.items():
                if words[:-1] in prefixs and float(count) / prefixs[words[:-1]] > 0.8:
                    del prefixs[words[:-1]]
                prefixs[words] = count
    with open(fout, 'w') as fo:
        lists = sorted(prefixs.iteritems(), key=operator.itemgetter(1), reverse=True)
        for words, count in lists:
            fo.write(str(count) + '\t' + ' '.join(words) + '\n')


def load_patterns(fname):
    prefixs = []
    with open(fname, 'r') as fp:
        for line in fp:
            count, words = line.strip().split('\t')
            prefixs.append(words)
    return prefixs


def gen_feature_one_hot(q1s, q2s, prefixs=[]):
    features = []
    features_raw = []
    feautres_col = []
    total_features = len(prefixs)
    for raw, (q1, q2) in enumerate(izip(q1s, q2s)):
        for i, prefix in enumerate(prefixs):
            if prefix in q1:
                features.append(1)
                features_raw.append(raw)
                feautres_col.append(i)
            if prefix in q2:
                features.append(1)
                features_raw.append(raw)
                feautres_col.append(i + total_features)
    features = np.array(features, dtype='int32')
    features_raw = np.array(features_raw, dtype='int32')
    feautres_col = np.array(feautres_col, dtype='int32')
    return csr_matrix((features, (features_raw, feautres_col)), shape=(raw + 1, total_features * 2))


def gen_features(q1s, q2s, prefixs=[]):
    features = []
    for raw, (q1, q2) in enumerate(izip(q1s, q2s)):
        q1_with_pattern = 0
        q2_with_pattern = 0
        have_diff_pattern = 0
        for i, prefix in enumerate(prefixs):
            t_q1, t_q2 = False, False
            if prefix in q1:
                t_q1 = True
                q1_with_pattern = 1
            if prefix in q2:
                t_q2 = True
                q2_with_pattern = 1
            if t_q1 != t_q2:
                have_diff_pattern = 1
        # features - both no pattern
        features.append(0 if q1_with_pattern or q2_with_pattern else 1)
        # features - both have pattern
        features.append(1 if q1_with_pattern and q2_with_pattern else 0)
        # features - pattern diff
        features.append(have_diff_pattern)
    return np.array(features, dtype='int32')


path = "../data/"
is_train = False

logger.info('Loading data')
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
test['is_duplicated'] = [-1] * test.shape[0]

logger.info('Clean and format data')
train['q1_clean'] = train['question1'].astype(str).apply(lambda x: stem_str(x))
train['q2_clean'] = train['question2'].astype(str).apply(lambda x: stem_str(x))
train_q1 = train['q1_clean'].tolist()
train_q2 = train['q2_clean'].tolist()

if is_train:
    sentences = train_q1 + train_q2
    sentences = preprocess(sentences)
    mining_pattern(sentences, 5, 50, 'pattern.txt', less2backup=3)
else:
    prefixs = load_patterns('pattern.txt')
    train_q1 = preprocess(train_q1)
    train_q2 = preprocess(train_q2)
    logger.info('Gen one hot feature for training data')
    train_oh_features = gen_feature_one_hot(train_q1, train_q2, prefixs)
    logger.info('Gen feature for training data')
    train_features = gen_features(train_q1, train_q2, prefixs)
    logger.info('Back up features')
    with open('./features/train.pattern.onehot.pkl', 'w') as fo:
        pickle.dump(train_oh_features, fo)
    with open('./features/train.pattern.pkl', 'w') as fo:
        pickle.dump(train_features, fo)
    # test
    logger.info('Clean and format test data')
    test['q1_clean'] = test['question1'].astype(str).apply(lambda x: stem_str(x))
    test['q2_clean'] = test['question2'].astype(str).apply(lambda x: stem_str(x))
    test_q1 = test['q1_clean'].tolist()
    test_q2 = test['q2_clean'].tolist()
    test_q1 = preprocess(test_q1)
    test_q2 = preprocess(test_q2)
    logger.info('Gen one hot feature for test data')
    test_oh_features = gen_feature_one_hot(test_q1, test_q2, prefixs)
    logger.info('Gen feature for test data')
    test_features = gen_features(test_q1, test_q2, prefixs)
    logger.info('Back up features')
    with open('./features/test.pattern.onehot.pkl', 'w') as fo:
        pickle.dump(test_oh_features, fo)
    with open('./features/test.pattern.pkl', 'w') as fo:
        pickle.dump(test_features, fo)
