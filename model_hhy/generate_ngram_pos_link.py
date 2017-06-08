import pandas as pd
import numpy as np
import nltk
import scipy.stats as sps
from .utils import ngram_utils,split_data,nlp_utils,dist_utils
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer


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

#embedd
vector_size = 300
glove_dir =   'D:/glove/glove.840B.{0}d.txt'.format(vector_size)
Embedd_model = nlp_utils._get_embedd_Index(glove_dir)

def getPOSLinks(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    text = nltk.word_tokenize(text)
    pos = nltk.pos_tag(text)
    links = []
    link = []
    active = False
    for w in pos:
        part = w[1]
        word = w[0]
        if(not active and (part[:2] == "DT" or part == "WP" or part == "VB" or part == "IN")):
            active = True
        if(active):
            link.append(wordnet_lemmatizer.lemmatize(word))
        #extract main body
        if(active and (part == "PRP" or part[:2] == "NN" or part == "." )):
            active = False
            links.append(" ".join(link))
            link = []
    return links

def links_pos_match(q1,q2):
    shared_links_in_q1 = [w for w in q1 if w in q2]
    shared_links_in_q2 = [w for w in q2 if w in q1]
    len1 = len(q1)
    len2 = len(q2)
    if len1 + len2==0:
        return 0
    R = (len(shared_links_in_q1) + len(shared_links_in_q2)) * 1.0 / (len1+len2)
    return R

def _wrapper_link_cos(q1, q2):
    link_emb_q1 = []
    for phr in q1:
        emb_q1 = np.zeros(100)
        wl = phr.lower().split()
        for w in wl:
            if w in Embedd_model:
                emb_q1 += Embedd_model[w]
        link_emb_q1.append(emb_q1)
    link_emb_q2 = []
    for phr in q2:
        emb_q2 = np.zeros(100)
        wl = phr.lower().split()
        for w in wl:
            if w in Embedd_model:
                emb_q2 += Embedd_model[w]
        link_emb_q2.append(emb_q2)

    #calc cos
    cos_lis = []
    for e1 in link_emb_q1:
        _q1_cos = []
        for e2 in link_emb_q2:
            _q1_cos.append(dist_utils._calc_similarity(e1,e2))
        if len(_q1_cos) == 0:
            _q1_cos = [-1]
        cos_lis.append(_q1_cos)
    return cos_lis

def _aggregate_w2w_sim(score):
    aggregation_mode_prev = ['max', 'mean', 'min', 'median']  # ["mean", "max", "median"]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]
    aggregator_prev = [None if m == "" else getattr(np, m) for m in aggregation_mode_prev]
    N = len(score)
    fea_sim = np.zeros((N, len(aggregator_prev) * len(aggregator)), dtype=float)
    for it in tqdm(np.arange(N)):
        for m, agg_pre in enumerate(aggregator_prev):
            for n, agg in enumerate(aggregator):
                idx = m * len(aggregator) + n
                if len(score)==0:
                    fea_sim[it,idx] = -1
                    continue
                # process in a safer way
                try:
                    tmp = []
                    for l in score[it]:
                        try:
                            s = agg_pre(l)
                        except:
                            s = -1
                        tmp.append(s)
                except:
                    tmp = [-1]
                try:
                    s = agg(tmp)
                except:
                    s = -1
                fea_sim[it, idx] = s
    return fea_sim

#link pos
links_q1 = []
links_q2 = []
for it in tqdm(np.arange(data_all.shape[0])):
    links_q1.append(getPOSLinks(str(data_all[it,0])))
    links_q2.append(getPOSLinks(str(data_all[it,1])))


#generate features
#match fea
fea_mat = []
for it in tqdm(range(len(links_q1))):
    fea_mat.append(links_pos_match(links_q1[it],links_q2[it]))
fea_mat = np.array(fea_mat).reshape(-1,1)

#aggregate sim fea
link_sim = []
for it in tqdm(np.arange(len(links_q1))):
    link_sim.append(_wrapper_link_cos(links_q1[it],links_q2[it]))
fea_sim  = _aggregate_w2w_sim(link_sim)

fea_ = np.hstack([fea_sim,fea_mat])
train_fea = fea_[:train.shape[0]]
test_fea = fea_[train.shape[0]:]

pd.to_pickle(train_fea,'../X_v2/train_pos_link.pkl')
test_x = split_data.split_data(test_fea)
for i in range(6):
    print(test_x[i].shape)
    pd.to_pickle(test_x[i],'../X_v2/test_pos_link{0}.pkl'.format(i))

