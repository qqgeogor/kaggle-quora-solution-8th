import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings(action='ignore')
seed = 1024
np.random.seed(seed)
path = "../input/"

def calc_cosine_dist(text_a ,text_b):
    return pairwise_distances(text_a, text_b, metric='cosine')[0][0]



print('LSI unigram Similarity part')

train_question1_tfidf = pd.read_pickle(path+'train_svd_100_question1_unigram_tfidf.pkl')
test_question1_tfidf = pd.read_pickle(path+'test_svd_100_question1_unigram_tfidf.pkl')
train_question2_tfidf = pd.read_pickle(path+'train_svd_100_question2_unigram_tfidf.pkl')
test_question2_tfidf = pd.read_pickle(path+'test_svd_100_question2_unigram_tfidf.pkl')

ss=StandardScaler()
ss.fit(np.vstack([train_question1_tfidf,
                  train_question2_tfidf]))
train_question1_tfidf=ss.transform(train_question1_tfidf)
train_question2_tfidf=ss.transform(train_question2_tfidf)
test_question1_tfidf=ss.transform(test_question1_tfidf)
test_question2_tfidf=ss.transform(test_question2_tfidf)


print('standardscaler.....')
###################################################
#############     undo
###################################################

train_tfidf_sim = []
for r1,r2 in zip(train_question1_tfidf,train_question2_tfidf):
    train_tfidf_sim.append(calc_cosine_dist(r1,r2))
test_tfidf_sim = []
for r1,r2 in zip(test_question1_tfidf,test_question2_tfidf):
    test_tfidf_sim.append(calc_cosine_dist(r1,r2))
train_tfidf_sim = np.array(train_tfidf_sim)
test_tfidf_sim = np.array(test_tfidf_sim)
pd.to_pickle(train_tfidf_sim,path+"train_unigram_lsi_100_sim.pkl")
pd.to_pickle(test_tfidf_sim,path+"test_unigram_lsi_100_sim.pkl")

del train_question1_tfidf
del test_question1_tfidf
del train_question2_tfidf
del test_question2_tfidf

print('LSI bigram Similarity part')
train_question1_bigram_tfidf = pd.read_pickle(path+'train_svd_100_question1_bigram_tfidf.pkl')
test_question1_bigram_tfidf = pd.read_pickle(path+'test_svd_100_question1_bigram_tfidf.pkl')
train_question2_bigram_tfidf = pd.read_pickle(path+'train_svd_100_question2_bigram_tfidf.pkl')
test_question2_bigram_tfidf = pd.read_pickle(path+'test_svd_100_question2_bigram_tfidf.pkl')

ss=StandardScaler()
ss.fit(np.vstack([train_question1_bigram_tfidf,
                  train_question2_bigram_tfidf]))
train_question1_bigram_tfidf=ss.transform(train_question1_bigram_tfidf)
train_question2_bigram_tfidf=ss.transform(train_question2_bigram_tfidf)
test_question1_bigram_tfidf=ss.transform(test_question1_bigram_tfidf)
test_question2_bigram_tfidf=ss.transform(test_question2_bigram_tfidf)

print('standardscaler.....')

train_bigram_tfidf_sim = []
for r1,r2 in zip(train_question1_bigram_tfidf,train_question2_bigram_tfidf):
    train_bigram_tfidf_sim.append(calc_cosine_dist(r1,r2))
test_bigram_tfidf_sim = []
for r1,r2 in zip(test_question1_bigram_tfidf,test_question2_bigram_tfidf):
    test_bigram_tfidf_sim.append(calc_cosine_dist(r1,r2))
train_bigram_tfidf_sim = np.array(train_bigram_tfidf_sim)
test_bigram_tfidf_sim = np.array(test_bigram_tfidf_sim)
pd.to_pickle(train_bigram_tfidf_sim,path+"train_bigram_lsi_100_sim.pkl")
pd.to_pickle(test_bigram_tfidf_sim,path+"test_bigram_lsi_100_sim.pkl")

del train_question1_bigram_tfidf
del test_question1_bigram_tfidf
del train_question2_bigram_tfidf
del test_question2_bigram_tfidf

