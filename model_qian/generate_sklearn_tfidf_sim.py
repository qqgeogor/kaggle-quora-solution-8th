import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 1024
np.random.seed(seed)
from config import path

ft = ['question1','question2','question1_porter','question2_porter']
train = pd.read_csv(path+"train_porter.csv")[ft]
test = pd.read_csv(path+"test_porter.csv")[ft]
# test['is_duplicated']=[-1]*test.shape[0]


len_train = train.shape[0]

data_all = pd.concat([train,test])


def calc_cosine_dist(text_a ,text_b):
    return pairwise_distances(text_a, text_b, metric='cosine')[0][0]



print('Tfidf raw Similarity part')

train_question1_tfidf = pd.read_pickle(path+'train_question1_tfidf.pkl')
test_question1_tfidf = pd.read_pickle(path+'test_question1_tfidf.pkl')
train_question2_tfidf = pd.read_pickle(path+'train_question2_tfidf.pkl')
test_question2_tfidf = pd.read_pickle(path+'test_question2_tfidf.pkl')

train_tfidf_sim = []
for r1,r2 in zip(train_question1_tfidf,train_question2_tfidf):
    train_tfidf_sim.append(calc_cosine_dist(r1,r2))
test_tfidf_sim = []
for r1,r2 in zip(test_question1_tfidf,test_question2_tfidf):
    test_tfidf_sim.append(calc_cosine_dist(r1,r2))
train_tfidf_sim = np.array(train_tfidf_sim)
test_tfidf_sim = np.array(test_tfidf_sim)
pd.to_pickle(train_tfidf_sim,path+"train_tfidf_sim.pkl")
pd.to_pickle(test_tfidf_sim,path+"test_tfidf_sim.pkl")

# del train_question1_tfidf
# del test_question1_tfidf
# del train_question2_tfidf
# del test_question2_tfidf

# print('Tfidf porter Similarity part')
# train_question1_porter_tfidf = pd.read_pickle(path+'train_question1_porter_tfidf.pkl')
# test_question1_porter_tfidf = pd.read_pickle(path+'test_question1_porter_tfidf.pkl')
# train_question2_porter_tfidf = pd.read_pickle(path+'train_question2_porter_tfidf.pkl')
# test_question2_porter_tfidf = pd.read_pickle(path+'test_question2_porter_tfidf.pkl')

# train_porter_tfidf_sim = []
# for r1,r2 in zip(train_porter_question1_tfidf,train_porter_question2_tfidf):
#     train_porter_tfidf_sim.append(calc_cosine_dist(r1,r2))
# test_porter_tfidf_sim = []
# for r1,r2 in zip(test_porter_question1_tfidf,test_porter_question2_tfidf):
#     test_porter_tfidf_sim.append(calc_cosine_dist(r1,r2))
# train_porter_tfidf_sim = np.array(train_porter_tfidf_sim)
# test_porter_tfidf_sim = np.array(test_porter_tfidf_sim)
# pd.to_pickle(train_porter_tfidf_sim,path+"train_porter_tfidf_sim.pkl")
# pd.to_pickle(test_porter_tfidf_sim,path+"test_porter_tfidf_sim.pkl")

