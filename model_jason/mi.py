# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text

from utils.inverted_index import InvertedIndex

from itertools import izip
import math

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


def concurrence(w1, w2, ii1, ii2):
    count = 0.0
    try:
        _set = set(ii2.term_doc[w2])
        for item in ii1.term_doc[w1]:
            if item in _set:
                count += 1
    except Exception:
        return 0
    return count


term_mi_hash = {}


def term_mi(w1, w2, ii1, ii2, pos=1):
    if (w1, w2, pos) in term_mi_hash:
        return term_mi_hash[(w1, w2, pos)]
    if (w2, w1, pos) in term_mi_hash:
        return term_mi_hash[(w2, w1, pos)]
    if ii1.get_word_appear(w1) == 0 or ii2.get_word_appear(w2) == 0:
        return 0
    coo = concurrence(w1, w2, ii1, ii2)
    to_log = coo * ii1.get_num_docs() / (ii1.get_word_appear(w1) * ii2.get_word_appear(w2))
    p_12 = coo / ii1.get_num_docs()
    result = 0
    if to_log != 0:
        result = p_12 * math.log(to_log)
    term_mi_hash[(w1, w2, pos)] = result
    return result


def mi(s1, s2, ii1, ii2, pos=1):
    res = 0.
    count = 0
    for w1 in s1.split():
        for w2 in s2.split():
            res += term_mi(w1, w2, ii1, ii2, pos=pos)
            count += 1
    return 0. if count == 0 else res / count


def prepare(q1s, q2s, labels, counting_label=0):
    iindex_q1 = InvertedIndex()
    iindex_q2 = InvertedIndex()
    for q1, q2, label in izip(q1s, q2s, labels):
        if counting_label != label:
            continue
        if counting_label:
            iindex_q1.add_input_document(q1.strip())
            iindex_q2.add_input_document(q2.strip())
        else:
            q1_words = q1.strip().split()
            q2_words = q2.strip().split()
            q1_wordset = set(q1_words)
            q2_wordset = set(q2_words)
            q1_diff_words = []
            q2_diff_words = []
            for w in q1_words:
                if w not in q2_wordset:
                    q1_diff_words.append(w)
            for w in q2_words:
                if w not in q1_wordset:
                    q2_diff_words.append(w)
            iindex_q1.add_input_document(' '.join(q1_diff_words))
            iindex_q2.add_input_document(' '.join(q2_diff_words))
    return iindex_q1, iindex_q2


def compute_mi_feature(q1s, q2s, ii1, ii2, nii1, nii2):
    total = len(q1s)
    features = []
    for i, (q1, q2) in enumerate(izip(q1s, q2s)):
        q1_words = q1.strip().split()
        q2_words = q2.strip().split()
        q1_wordset = set(q1_words)
        q2_wordset = set(q2_words)
        q1_diff_words = []
        q2_diff_words = []
        for w in q1_words:
            if w not in q2_wordset:
                q1_diff_words.append(w)
        for w in q2_words:
            if w not in q1_wordset:
                q2_diff_words.append(w)
        features.append([mi(q1, q2, ii1, ii2, pos=1),
            mi(' '.join(q1_diff_words), ' '.join(q2_diff_words), nii1, nii2, pos=0)])
        if i and i % 10000 == 0:
            logger.info('\t%d features of %d, hashed %d', i, total, len(term_mi_hash))
    return np.array(features, dtype='float32')


path = "../data/"

logger.info('Loading data')
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

logger.info('Clean and format data')
train['q1_clean'] = train['question1'].astype(str).apply(lambda x: stem_str(x))
train['q2_clean'] = train['question2'].astype(str).apply(lambda x: stem_str(x))
train_q1 = train['q1_clean'].tolist()
train_q2 = train['q2_clean'].tolist()
labels = train['is_duplicate'].astype(int).tolist()

logger.info('Prepare inverted index')
iindex_q1, iindex_q2 = prepare(train_q1, train_q2, labels, counting_label=1)
niindex_q1, niindex_q2 = prepare(train_q1, train_q2, labels, counting_label=0)

logger.info('Gen feature for training data')
train_features = compute_mi_feature(train_q1, train_q2, iindex_q1, iindex_q2, niindex_q1, niindex_q2)
logger.info('Back up features')
with open('./features/train.mi.pkl', 'w') as fo:
    pickle.dump(train_features, fo)

logger.info('Clean and format test data')
test['q1_clean'] = test['question1'].astype(str).apply(lambda x: stem_str(x))
test['q2_clean'] = test['question2'].astype(str).apply(lambda x: stem_str(x))
test_q1 = test['q1_clean'].tolist()
test_q2 = test['q2_clean'].tolist()
logger.info('Gen feature for test data')
test_features = compute_mi_feature(test_q1, test_q2, iindex_q1, iindex_q2, niindex_q1, niindex_q2)
logger.info('Back up features')
with open('./features/test.mi.pkl', 'w') as fo:
    pickle.dump(test_features, fo)
