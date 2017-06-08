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

def get_uni_gram_(q):
    w = q.lower().split()
    return ngram_utils._ngrams(w,1)

def get_big_gram_(q):
    w = q.lower().split()
    return ngram_utils._ngrams(w,2)

def get_tri_gram_(q):
    w = q.lower().split()
    return ngram_utils._ngrams(w,3)

def generate_indicator_(gram_q1,gram_q2,N):
    len_gram_q1 = list(map(len,gram_q1))
    len_gram_q2 = list(map(len,gram_q2))
    max_len = max(max(len_gram_q1),max(len_gram_q2))
    q1_indicator = np.zeros((N,max_len))
    q2_indicator = np.zeros((N,max_len))
    for i in tqdm(np.arange(N)):
        for j,w in enumerate(gram_q1[i]):
            if w in gram_q2[i]:
                q1_indicator[i,j] = 1
        for j,w in enumerate(gram_q2[i]):
            if w in gram_q1[i]:
                q2_indicator[i,j] = 1
    return q1_indicator,q2_indicator
    # sps.spearmanr(q1_indicator[:,1],y_train)[0]

uni_gram_q1 = list(map(get_uni_gram_, data_all[:, 0]))
uni_gram_q2 = list(map(get_uni_gram_, data_all[:, 1]))
uni_q1,uni_q2 = generate_indicator_(uni_gram_q1,uni_gram_q2,data_all.shape[0])

big_gram_q1 = list(map(get_big_gram_,data_all[:,0]))
big_gram_q2 = list(map(get_big_gram_,data_all[:,1]))
big_q1,big_q2 = generate_indicator_(big_gram_q1,big_gram_q2,data_all.shape[0])

tri_gram_q1 = list(map(get_tri_gram_,data_all[:,0]))
tri_gram_q2 = list(map(get_tri_gram_,data_all[:,1]))
tri_q1,tri_q2 = generate_indicator_(tri_gram_q1,tri_gram_q2,data_all.shape[0])

from scipy.sparse import csr_matrix
csr_q1 = csr_matrix(uni_q1)
csr_q2 = csr_matrix(uni_q2)
pd.to_pickle(csr_q1,'../X_v2/uni_gram_q1.pkl')
pd.to_pickle(csr_q2,'../X_v2/uni_gram_q2.pkl')

csr_q1 = csr_matrix(big_q1)
csr_q2 = csr_matrix(big_q2)
pd.to_pickle(csr_q1,'../X_v2/big_gram_q1.pkl')
pd.to_pickle(csr_q2,'../X_v2/big_gram_q2.pkl')

csr_q1 = csr_matrix(tri_q1)
csr_q2 = csr_matrix(tri_q2)
pd.to_pickle(csr_q1,'../X_v2/tri_gram_q1.pkl')
pd.to_pickle(csr_q2,'../X_v2/tri_gram_q2.pkl')
