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


def calc_w2v_sim(row,embedder,idf_dict,dim):
    '''
    Calc w2v similarities and diff of centers of query\title
    '''
    a2 = [x for x in row['question1'].lower().split() if x in embedder.vocab]
    b2 = [x for x in row['question2'].lower().split() if x in embedder.vocab]

    
    vectorA = np.zeros(dim)
    for w in a2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        vectorA += coef*embedder[w]
    if len(a2)!=0:
        vectorA /= len(a2)
    
    vectorB = np.zeros(dim)
    for w in b2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        vectorB += coef*embedder[w]
    if len(b2)!=0:
        vectorB /= len(b2)
    
    return (vectorA,vectorB)

def gen_w2v(train,dim=100):
    corpus = []
    ft = ['question1','question2']
    for f in ft:
        train[f] = train[f].astype(str).apply(lambda x:remove_punctuation(x))

    model = Word2Vec.load(path+'my_w2v_%s.mdl'%dim)
    idf_dict = pickle.load(open(path+'idf_dict.pkl','rb'))

    X_w2v = []
    sim_list = []
    dist_list = []
    for i,row in train.astype(str).iterrows():
        va, vb= calc_w2v_sim(row,model,idf_dict,dim)
        vdiff = np.concatenate([va,vb])
        X_w2v.append(vdiff)

    X_w2v_tr = np.array(X_w2v)
    return X_w2v_tr

if __name__ == '__main__':
    dim = 100
    ft = ['question1','question2']
    train = pd.read_csv(path+"train.csv")[ft]
    test = pd.read_csv(path+"test.csv")[ft]

    corpus = []
    for f in ft:
        train[f] = train[f].astype(str).apply(lambda x:remove_punctuation(x))
        test[f] = test[f].astype(str).apply(lambda x:remove_punctuation(x))
        corpus+=train[f].values.tolist()

    corpus = [d.lower().split(" ") for d in corpus]
    model = Word2Vec(corpus, size=dim, window=5, min_count=5, workers=7)
    model.save(path+'my_w2v_%s.mdl'%dim)
    
    print('Generate selftrained w2v sim,distance and diff')

    X_w2v_tr = gen_w2v(train,dim=dim)
    np.savetxt(path+'train_selftrained_w2v.txt',X_w2v_tr)
    del X_w2v_tr

    X_w2v_te = gen_w2v(test,dim=dim)
    np.savetxt(path+'test_selftrained_w2v.txt',X_w2v_te)
