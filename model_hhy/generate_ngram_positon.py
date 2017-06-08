import pandas as pd
import numpy as np
import nltk
import scipy.stats as sps
from .utils import ngram_utils,split_data
from tqdm import tqdm

seed = 1024
np.random.seed(seed)

path = '../data/'

train = pd.read_csv(path+'train_porter.csv')
test = pd.read_csv(path+'test_porter.csv')
test['is_duplicated']=[-1]*test.shape[0]
y_train = train['is_duplicate']
feats= ['question1_porter','question2_porter']
train_value = train[feats].values

data_all = pd.concat([train,test])[feats].values


def get_uni_gram_(q):
    w = str(q).lower().split()
    return ngram_utils._ngrams(w,1)


def generate_indicator_pos(gram_q1,gram_q2,N):
    len_gram_q1 = list(map(len,gram_q1))
    len_gram_q2 = list(map(len,gram_q2))
    max_len = max(max(len_gram_q1),max(len_gram_q2))
    q1_indicator = np.zeros((N,max_len+1))
    q2_indicator = np.zeros((N,max_len+1))
    for i in tqdm(np.arange(N)):
        q1_str = ' '.join(gram_q1[i])
        q2_str = ' '.join(gram_q2[i])
        for j,w in enumerate(gram_q1[i]):
            p = q2_str.find(w)
            q1_indicator[i,j] = abs(p-j)
        for j,w in enumerate(gram_q2[i]):
            p = q1_str.find(w)
            q2_indicator[i,j] = abs(p-j)
    return q1_indicator,q2_indicator

uni_gram_q1 = list(map(get_uni_gram_, data_all[:, 0]))
uni_gram_q2 = list(map(get_uni_gram_, data_all[:, 1]))

uni_q1,uni_q2 = generate_indicator_pos(uni_gram_q1,uni_gram_q2,data_all.shape[0])


sps.spearmanr(uni_q1[0:y_train.shape[0]].std(axis=1),y_train)[0]
from scipy.sparse import csr_matrix
csr_q1 = csr_matrix(uni_q1)
csr_q2 = csr_matrix(uni_q2)
pd.to_pickle(csr_q1,'../X_v2/uni_gram_q1_pos.pkl')
pd.to_pickle(csr_q2,'../X_v2/uni_gram_q2_pos.pkl')
