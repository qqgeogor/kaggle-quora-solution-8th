import numpy as np
import pandas as pd
import datetime
import operator
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from  tqdm import tqdm
import scipy.stats as sps

seed = 1024
np.random.seed(seed)

path = '../data/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
data_all = pd.concat([train, test])[['question1','question2']]
stops = set(stopwords.words("english"))


def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)


def word_share(row):
    q1_list = str(row[0]).lower().split()
    q1 = set(q1_list)
    q2_list = str(row[1]).lower().split()
    q2 = set(q2_list)
    q1words = q1.difference(stops)
    q2words = q2.difference(stops)

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0] == i[1]) / max(len(q1_list), len(q2_list))

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)
    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)
    shared_words = q1words.intersection(q2words)

    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q1_weights

    R1 = np.sum(shared_weights) / np.sum(total_weights)  # tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)+1)  # count share
    R31 = len(q1stops) / (len(q1words)+1)  # stops in q1
    R32 = len(q2stops) / (len(q2words)+1)  # stops in q2
    Rcosine_denominator = (np.sqrt(np.dot(q1_weights, q1_weights)) * np.sqrt(np.dot(q2_weights, q2_weights)))
    Rcosine = np.dot(shared_weights, shared_weights) / Rcosine_denominator
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    fea = []
    fea.append(len(q1stops))
    fea.append(len(q2stops))
    fea.append(words_hamming)
    fea.append(R1)
    fea.append(R2)
    fea.append(R31)
    fea.append(R32)
    fea.append(Rcosine)
    fea.append(R2gram)
    return fea

#calc all words
all_list = data_all['question1'].astype(str).tolist() + data_all['question2'].astype(str).tolist()
words = (" ".join(all_list)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

dd = data_all.values
fea = []
for i in tqdm(np.arange(dd.shape[0])):
    fea.append(word_share(dd[i]))

fea = np.array(fea)
train_fea = fea[:train.shape[0]]
test_fea = fea[train.shape[0]:]



pd.to_pickle(train_fea,'../X_v2/train_stop_basic.pkl')
pd.to_pickle(test_fea,'../X_v2/test_stop_basic.pkl')
