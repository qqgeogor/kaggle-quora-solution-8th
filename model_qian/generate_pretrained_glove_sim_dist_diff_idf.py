import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import pickle
import string

seed = 1024
np.random.seed(seed)
from config import path

ft = ['question1','question2']
train = pd.read_csv(path+"train.csv")[ft]
test = pd.read_csv(path+"test.csv")[ft]
# test['is_duplicated']=[-1]*test.shape[0]


string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

def remove_punctuation(x):
    new_line = [ w for w in list(x) if w not in string.punctuation]
    new_line = ''.join(new_line)
    return new_line

ft = ['question1','question2']
for f in ft:
    train[f] = train[f].astype(str).apply(lambda x:remove_punctuation(x))
    test[f] = test[f].astype(str).apply(lambda x:remove_punctuation(x))
    


len_train = train.shape[0]
def read_emb(path):
    count=0
    f = open(path,'r')
    emb_dict = dict()
    for line in f:
        # if count==0:
        #     count+=1
        #     continue
        line = line.strip().split(' ')
        id = line[0]
        
        weights = line[1:]
        weights = np.array([float(i) for i in weights])
        count+=1
        emb_dict[id] = weights
    return emb_dict
# data_all = pd.concat([train,test])
# model = Word2Vec.load_word2vec_format(path+'glove.840B.300d.txt', binary=False)  # C binary format
model = read_emb(path+'glove.840B.300d.txt')

idf_dict = pickle.load(open(path+'idf_dict.pkl','rb'))

from  scipy.spatial.distance import cosine
def calc_glove_sim(row,embedder,idf_dict):
    '''
    Calc glove similarities and diff of centers of query\title
    '''
    a2 = [x for x in remove_punctuation(row['question1']).lower().split() if x in embedder]
    b2 = [x for x in remove_punctuation(row['question2']).lower().split() if x in embedder]

    # if len(a2)>0 and len(b2)>0:
    #     glove_sim = embedder.n_similarity(a2, b2)
    # else:
    #     return((-1, -1, np.zeros(300)))
    
    vectorA = np.zeros(300)
    for w in a2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        vectorA += coef*embedder[w]
    vectorA /= len(a2)
    
    vectorB = np.zeros(300)
    for w in b2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        vectorB += coef*embedder[w]
    vectorB /= len(b2)
    
    vector_diff = (vectorA - vectorB)
    glove_sim = cosine(vectorA,vectorB)
    glove_vdiff_dist = np.sqrt(np.sum(vector_diff**2))
    return (glove_sim,glove_vdiff_dist, vector_diff)


print('Generate pretrained glove sim,distance and diff')
X_glove = []
sim_list = []
dist_list = []
for i,row in train.astype(str).iterrows():
    sim, dist, vdiff = calc_glove_sim(row,model,idf_dict)
    X_glove.append(vdiff)
    sim_list.append(sim)
    dist_list.append(dist)
X_glove_tr = np.array(X_glove)
train['glove_sim'] = np.array(sim_list)
train['glove_dist'] = np.array(dist_list)

features = ['glove_sim','glove_dist']
glove_sim_dist_train = train[features].values
# pd.to_pickle(X_glove_tr,path+'train_pretrained_glove_diff.pkl')
np.save(path+'train_pretrained_glove_diff.pkl',X_glove_tr)


del X_glove_tr
pd.to_pickle(glove_sim_dist_train,path+'train_pretrained_glove_sim_dist.pkl')
del glove_sim_dist_train

X_glove = []
sim_list = []
dist_list = []
for i,row in test.astype(str).iterrows():
    sim, dist, vdiff = calc_glove_sim(row,model,idf_dict)
    X_glove.append(vdiff)
    sim_list.append(sim)
    dist_list.append(dist)
X_glove_te = np.array(X_glove)
test['glove_sim'] = np.array(sim_list)
test['glove_dist'] = np.array(dist_list)

# pd.to_pickle(X_glove_te,path+'test_pretrained_glove_diff.pkl')
np.save(path+'test_pretrained_glove_diff.pkl',X_glove_te)


del X_glove_te
glove_sim_dist_test = test[features].values
pd.to_pickle(glove_sim_dist_test,path+'test_pretrained_glove_sim_dist.pkl')
