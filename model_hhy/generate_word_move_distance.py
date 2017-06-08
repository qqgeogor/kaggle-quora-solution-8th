import pandas as pd
import numpy as np
import nltk
import scipy.stats as sps
from .utils import ngram_utils,split_data,nlp_utils,dist_utils
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import spacy
import wmd
from  wmd import WMD

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

nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)


fea = []
for it in tqdm(np.arange(data_all.shape[0])):
    doc1 = nlp(str(data_all[it][0]))
    doc2 = nlp(str(data_all[it][1]))
    try:
        sim = doc1.similarity(doc2)
        fea.append(sim)
    except:
        fea.append(-1)


train_fea = np.array(fea[:train.shape[0]]).reshape(-1,1)
test_fea = np.array(fea[train.shape[0]:]).reshape(-1,1)

pd.to_pickle(train_fea,'../X_v2/train_wmd.pkl')
test_x = split_data.split_test(test_fea)
for i in range(6):
    pd.to_pickle(test_x[i],'../X_v2/test_wmd{0}.pkl'.format(i))
