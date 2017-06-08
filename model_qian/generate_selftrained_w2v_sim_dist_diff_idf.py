import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import pickle
import string
from config import path
seed = 1024
np.random.seed(seed)




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
train = pd.read_csv(path+"train.csv")[ft]
test = pd.read_csv(path+"test.csv")[ft]
# test['is_duplicated']=[-1]*test.shape[0]
# data_all = pd.concat([train,test])

len_train = train.shape[0]

feats= ['question1','question2']

corpus = []
for f in feats:
    train[f] = train[f].astype(str).apply(lambda x:remove_punctuation(x))
    test[f] = test[f].astype(str).apply(lambda x:remove_punctuation(x))
    corpus+=train[f].values.tolist()

corpus = [d.lower().split(" ") for d in corpus]
# model = Word2Vec(corpus, size=200, window=5, min_count=5, workers=7)
# model.save(path+'my_w2v.mdl')
model = Word2Vec.load(path+'my_w2v.mdl')
idf_dict = pickle.load(open(path+'idf_dict.pkl','rb'))

def calc_w2v_sim(row,embedder,idf_dict):
    '''
    Calc w2v similarities and diff of centers of query\title
    '''
    a2 = [x for x in row['question1'].lower().split() if x in embedder.vocab]
    b2 = [x for x in row['question2'].lower().split() if x in embedder.vocab]

    if len(a2)>0 and len(b2)>0:
        w2v_sim = embedder.n_similarity(a2, b2)
    else:
        return((-1, -1, np.zeros(200)))
    
    vectorA = np.zeros(200)
    for w in a2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        vectorA += coef*embedder[w]
    vectorA /= len(a2)
    
    vectorB = np.zeros(200)
    for w in b2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        vectorB += coef*embedder[w]
    vectorB /= len(b2)
    
    vector_diff = (vectorA - vectorB)
    
    w2v_vdiff_dist = np.sqrt(np.sum(vector_diff**2))
    return (w2v_sim, w2v_vdiff_dist, vector_diff)

print('Generate selftrained w2v sim,distance and diff')
X_w2v = []
sim_list = []
dist_list = []
for i,row in train.astype(str).iterrows():
    sim, dist, vdiff = calc_w2v_sim(row,model,idf_dict)
    # X_w2v.append(vdiff)
    sim_list.append(sim)
    dist_list.append(dist)
X_w2v_tr = np.array(X_w2v)
train['w2v_sim'] = np.array(sim_list)
train['w2v_dist'] = np.array(dist_list)

features = ['w2v_sim','w2v_dist']
w2v_sim_dist_train = train[features].values
# pd.to_pickle(X_w2v_tr,path+'train_selftrained_w2v_diff.pkl')
# np.savetxt(path+'train_selftrained_w2v_diff.txt',X_w2v_tr)
# del X_w2v_tr
pd.to_pickle(w2v_sim_dist_train,path+'train_selftrained_w2v_sim_dist.pkl')
del w2v_sim_dist_train

X_w2v = []
sim_list = []
dist_list = []
for i,row in test.astype(str).iterrows():
    sim, dist, vdiff = calc_w2v_sim(row,model,idf_dict)
    # X_w2v.append(vdiff)
    sim_list.append(sim)
    dist_list.append(dist)
X_w2v_te = np.array(X_w2v)
test['w2v_sim'] = np.array(sim_list)
test['w2v_dist'] = np.array(dist_list)

# pd.to_pickle(X_w2v_te,path+'test_selftrained_w2v_diff.pkl')
# np.savetxt(path+'test_selftrained_w2v_diff.txt',X_w2v_te)

# del X_w2v_te
w2v_sim_dist_test = test[features].values
pd.to_pickle(w2v_sim_dist_test,path+'test_selftrained_w2v_sim_dist.pkl')
