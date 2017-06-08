import pandas as pd
import numpy as np
import nltk
import scipy.stats as sps
from .utils import ngram_utils,split_data
from tqdm import tqdm

seed = 1024
np.random.seed(seed)

path = '../data/'

train = pd.read_pickle(path+'train_clean.pkl')
test = pd.read_pickle(path+'test_clean.pkl')
test['is_duplicated']=[-1]*test.shape[0]
y_train = train['is_duplicate']
feats= ['clean_question1','clean_question2']
train_value = train[feats].values

data_all = pd.concat([train,test])[feats].values

def get_n_gram_bleu_score(q1,q2,n=1):
    w1 = q1.lower().split()
    w2 = q2.lower().split()
    ngram_w1 = ngram_utils._ngrams(w1,n)
    ngram_w2 = ngram_utils._ngrams(w2,n)
    return  nltk.translate.bleu_score.sentence_bleu([ngram_w1], ngram_w2)

_ngram_str_map = {
    1: "Unigram",
    2: "Bigram",
    3: "Trigram",
    4: "Fourgram",
    5: "Fivegram",
    12: "UBgram",
    123: "UBTgram",
}

def generate_n_gram_bleu(data,ngram=1):
    fea_list = []
    for i in tqdm(np.arange(data.shape[0])):
        q1 = get_n_gram_bleu_score(data[i][0], data[i][1],ngram)
        q2 = get_n_gram_bleu_score(data[i][1],data[i][0],ngram)
        _min = min(q1,q2)
        _max = max(q1,q2)
        fea_list.append([q1,q2,_min,_max])
    return np.array(fea_list)


ngram_fea = np.empty((data_all.shape[0],0))
for i in range(4):
    print('generate '+_ngram_str_map[i+1]+' bleu')
    gram_fea = generate_n_gram_bleu(data_all,i+1)
    ngram_fea = np.hstack([ngram_fea,gram_fea])
    print(ngram_fea.shape)

train_gram = ngram_fea[:train.shape[0]]
test_gram = ngram_fea[train.shape[0]:]
test_x = split_data.split_test(test_gram)

pd.to_pickle(train_gram,'../X_v2/train_ngram_bleu.pkl')
for i in range(6):
    pd.to_pickle(test_x[i],'../X_v2/test_ngram_bleu{0}.pkl'.format(i))


def drop_feature(data):
    drop_list = []
    for i in range(data.shape[1]):
        for j in range(i,data.shape[1]):
            s = sps.spearmanr(data[:,i],data[:,j])[0]
            if abs(s)>0.8:
                drop_list.append(j)
    drop_list = set(drop_list)
    return  drop_list


#select imp feature
train_gram_stats = np.vstack([train_gram.mean(axis=1),train_gram.min(axis=1),train_gram.max(axis=1)]).T
train_gram = pd.read_pickle('../X_v2/belu/train_ngram_bleu.pkl')
train_gram = np.hstack([train_gram[:,[2,7,10]],train_gram_stats])
pd.to_pickle(train_gram,'../X_v2/train_ngram_bleu.pkl')

for i in range(6):
    test_x = pd.read_pickle('../X_v2/belu/test_ngram_bleu{0}.pkl'.format(i))
    test_x_stat = np.vstack([test_x.mean(axis=1),test_x.min(axis=1),test_x.max(axis=1)]).T
    test_x = np.hstack([test_x[:,[2,7,10]],test_x_stat])
    pd.to_pickle(test_x,'../X_v2/test_ngram_bleu{0}.pkl'.format(i))

#sps.spearmanr(train['q1_big_bleu'].values,y_train)[0]

