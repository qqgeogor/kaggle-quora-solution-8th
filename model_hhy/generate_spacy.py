import spacy
import string
import numpy as np
import pandas as pd
from spacy.en import English
import scipy.stats as sps
from .utils import subject_object_extraction,split_data,pos_utils,nlp_utils,dist_utils
from  tqdm import tqdm

seed = 1024
np.random.seed(seed)
path = '../data/'


train = pd.read_pickle(path+'train_clean.pkl')
test = pd.read_pickle(path+'test_clean.pkl')
test['is_duplicated']=[-1]*test.shape[0]
y_train = train['is_duplicate']
feats= ['clean_question1','clean_question2']

batch_test = split_data.split_test(test[feats].values)
train_value = train[feats].values

#utils
parser  = English()
vector_size = 100
glove_dir = path + 'glove.6B.{0}d.txt'.format(vector_size)
Embedd_util = nlp_utils.Embedd_generator(glove_dir)


class spacy_generator:
    def __init__(self,data=[]):
        self.corpus = data
        self.has_data = False

    def fit(self,data):
        self.corpus = data
        self.parse_q1 = {}
        self.parse_q2 = {}
        for i in tqdm(np.arange(self.corpus.shape[0])):
            self.parse_q1[i] = parser(str(self.corpus[i][0]))
            self.parse_q2[i] = parser(str(self.corpus[i][1]))
        self.has_data = True


    def set_parse(self,data,parse_q1,parse_q2):
        self.corpus = data
        self.parse_q1 = parse_q1
        self.parse_q2 = parse_q2
        self.has_data = True

    def generate_token_prob(self):
        f_raw_1 = []
        f_raw_2 = []
        for i in tqdm(np.arange(self.corpus.shape[0])):
            raw_1 = []
            raw_2 = []
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            for token in parse_1:
                raw_1.append(token.prob)
            for token in parse_2:
                raw_2.append(token.prob)
            f_raw_1.append(np.array(raw_1))
            f_raw_2.append(np.array(raw_2))
        return f_raw_1, f_raw_2

    def generate_token_cluster(self):
        f_raw_1 = []
        f_raw_2 = []
        for i in tqdm(np.arange(self.corpus.shape[0])):
            raw_1 = []
            raw_2 = []
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            for token in parse_1:
                raw_1.append(token.cluster)
            for token in parse_2:
                raw_2.append(token.cluster)
            f_raw_1.append(raw_1)
            f_raw_2.append(raw_2)
        return f_raw_1, f_raw_2

    def generate_token_pos(self):
        f_raw_1 = []
        f_raw_2 = []
        for i in tqdm(np.arange(self.corpus.shape[0])):
            raw_1 = []
            raw_2 = []
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            for token in parse_1:
                raw_1.append(token.pos)
            for token in parse_2:
                raw_2.append(token.pos)
            f_raw_1.append(raw_1)
            f_raw_2.append(raw_2)
        return f_raw_1, f_raw_2

    def generate_token_dep(self):
        f_raw_1 = []
        f_raw_2 = []
        for i in tqdm(np.arange(self.corpus.shape[0])):
            raw_1 = []
            raw_2 = []
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            for token in parse_1:
                raw_1.append(token.dep)
            for token in parse_2:
                raw_2.append(token.dep)#dep hash number
            f_raw_1.append(raw_1)
            f_raw_2.append(raw_2)
        return f_raw_1,f_raw_2
    # entity label
    def generate_entity_label(self):
        f_raw_1 = []
        f_raw_2 = []
        for i in tqdm(np.arange(self.corpus.shape[0])):
            raw_1 = []
            raw_2 = []
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            for ent in parse_1.ents:
                raw_1.append(ent.label)
            for ent in parse_2.ents:
                raw_2.append(ent.label)
            f_raw_1.append(raw_1)
            f_raw_2.append(raw_2)
        return f_raw_1, f_raw_2

    #entity synmatic
    def generate_entity_(self):
        f_raw_1 = []
        f_raw_2 = []
        for i in tqdm(np.arange(self.corpus.shape[0])):
            raw_1 = []
            raw_2 = []
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            for ent in parse_1.ents:
                raw_1.append(str(ent))
            for ent in parse_2.ents:
                raw_2.append(str(ent))
            f_raw_1.append(raw_1)
            f_raw_2.append(raw_2)
        return f_raw_1, f_raw_2

    def generate_SVO(self):
        f_raw_1 = []
        f_raw_2 = []
        for i in tqdm(np.arange(self.corpus.shape[0])):
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            raw_1 = subject_object_extraction.findSVOs(parse_1)
            raw_2 = subject_object_extraction.findSVOs(parse_2)
            f_raw_1.append(raw_1)
            f_raw_2.append(raw_2)
        return f_raw_1, f_raw_2

    def generate_noun_chunk(self):
        f_raw_1 = []
        f_raw_2 = []
        for i in tqdm(np.arange(self.corpus.shape[0])):
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            raw_1 = list(parse_1.noun_chunks)
            raw_2 = list(parse_2.noun_chunks)
            f_raw_1.append(raw_1)
            f_raw_2.append(raw_2)
        return f_raw_1, f_raw_2

def aggregate(fea_q1,fea_q2,mode):
    fun = getattr(np, mode)
    r_p_1= np.array(list(map(lambda x:fun(x),fea_q1)))
    r_p_2 = np.array(list(map(lambda x:fun(x),fea_q2)))
    fea = np.hstack([r_p_1.reshape(-1,1),r_p_2.reshape(-1,1)])
    return fea

# generate useful feature in train dataset
def generate_feature(data):
    spacy_gen = spacy_generator()
    spacy_gen.fit(data)
    #token log prob
    print('generate token log prob--------------')
    prob_fea_q1,prob_fea_q2 = spacy_gen.generate_token_prob()
    fea_min = aggregate(prob_fea_q1,prob_fea_q2,'min') #min max
    fea_max = aggregate(prob_fea_q1,prob_fea_q2,'max') #max std
    fea_mean = aggregate(prob_fea_q1,prob_fea_q2,'mean')#min std
    fea_std = aggregate(prob_fea_q1,prob_fea_q2,'std')#std max
    fea_median= aggregate(prob_fea_q1,prob_fea_q2,'median')#std

    fea_prob = np.hstack([fea_min.min(axis=1).reshape(-1,1),fea_min.max(axis=1).reshape(-1,1),
                       fea_max.max(axis=1).reshape(-1,1),fea_max.std(axis=1).reshape(-1,1),
                       fea_mean.min(axis=1).reshape(-1,1),fea_mean.std(axis=1).reshape(-1,1),
                       fea_std.std(axis=1).reshape(-1,1),fea_std.max(axis=1).reshape(-1,1),
                       fea_median.std(axis=1).reshape(-1,1)])
    print(fea_prob.shape)
    print('generate token brown cluster--------------')
    #token brown cluster
    cluster_q1,cluster_q2 = spacy_gen.generate_token_cluster()

    fea_max = aggregate(cluster_q1,cluster_q2,'max') #max std
    fea_mean = aggregate(cluster_q1,cluster_q2,'mean')#mean
    fea_std = aggregate(cluster_q1,cluster_q2,'std')#std
    fea_median= aggregate(cluster_q1,cluster_q2,'median')#min std

    fea_cluster = np.hstack([
                       fea_max.max(axis=1).reshape(-1,1),fea_max.std(axis=1).reshape(-1,1),
                       fea_mean.mean(axis=1).reshape(-1,1),
                       fea_std.std(axis=1).reshape(-1,1),
                       fea_median.std(axis=1).reshape(-1,1),fea_median.min(axis=1).reshape(-1,1)])
    print(fea_cluster.shape)

    print('generate token pos--------------')
    #token pos tag
    pos_q1,pos_q2 = spacy_gen.generate_token_pos()
    pos_i = list(map(pos_utils.pos_match,pos_q1,pos_q2))
    pos_d = list(map(pos_utils.pos_diff,pos_q1,pos_q2))
    pos_sub = list(map(pos_utils.pos_sub, pos_q1, pos_q2))
    pos_same = list(map(pos_utils.pos_same, pos_q1, pos_q2))
    pos_most = list(map(pos_utils.pos_most_same,pos_q1,pos_q2))
    pos_min = list(map(pos_utils.pos_fre_min_same,pos_q1,pos_q2))

    #token depedency
    fea_pos = np.vstack([pos_i,pos_d,pos_sub,
                         pos_same,pos_most,pos_min]).T

    print(fea_pos.shape)

    print('generate token dependency--------------')
    dep_q1,dep_q2 = spacy_gen.generate_token_dep()
    dep_i = list(map(pos_utils.pos_match, dep_q1, dep_q2))
    dep_d = list(map(pos_utils.pos_diff, dep_q1, dep_q2))
    dep_sub = list(map(pos_utils.pos_sub, dep_q1, dep_q2))
    dep_same = list(map(pos_utils.pos_same, dep_q1, dep_q2))
    dep_most = list(map(pos_utils.pos_most_same, dep_q1, dep_q2))
    dep_min = list(map(pos_utils.pos_fre_min_same, dep_q1, dep_q2))

    fea_dep = np.vstack([dep_i,dep_d,dep_sub,dep_same,dep_most,dep_min]).T
    print(fea_dep.shape)

    print('generate entity label--------------')
    #sent entity also could
    en_q1,en_q2 = spacy_gen.generate_entity_label()
    en_ma = list(map(pos_utils.pos_match_num,en_q1,en_q2))
    en_sa = list(map(pos_utils.pos_same,en_q1,en_q2))
    en_sub = list(map(pos_utils.pos_sub,en_q1,en_q2))
    fea_en = np.vstack([en_ma,en_sa,en_sub]).T
    print(fea_en.shape)

    print('generate entity synmatic--------------')
    #sent entity also could
    en_q1_,en_q2_ = spacy_gen.generate_entity_()
    en_sim_ = list(map(Embedd_util._warpper_token_embedd_cos,en_q1_,en_q2_))
    en_ma_ = list(map(pos_utils.pos_match_num,en_q1_,en_q2_))
    en_sub_ = list(map(pos_utils.pos_sub,en_q1_,en_q2_))
    fea_en_syn = np.vstack([en_sim_,en_ma_,en_sub_]).T
    print(fea_en_syn.shape)

    print('generate subject,verb,object--------------')
    #sent subject verb object
    svo_q1,svo_q2 = spacy_gen.generate_SVO()
    svo_sa = list(map(pos_utils.pos_match_num,svo_q1,svo_q2))
    svo_di = list(map(pos_utils.pos_len,svo_q1,svo_q2))
    svo_en = list(map(pos_utils.en_is_empty,svo_q1,svo_q2))
    svo_sim = list(map(Embedd_util._wrapper_sent_embedd_cos,svo_q1,svo_q2))
    s_sim = list(map(Embedd_util._wrapper_sent_subject_cos,svo_q1,svo_q2))
    v_sim = list(map(Embedd_util._wrapper_sent_verb_cos,svo_q1,svo_q2))
    o_sim = list(map(Embedd_util._wrapper_sent_object_cos,svo_q1,svo_q2))
    fea_svo = np.vstack([svo_sa,svo_en,o_sim]).T
    print(fea_svo.shape)

    print('generate noun phrase--------------')
    #sent noun phrase
    noun_q1, noun_q2 = spacy_gen.generate_noun_chunk()
    noun_di = list(map(pos_utils.pos_len, noun_q1, noun_q2))
    noun_sim = list(map(Embedd_util._warpper_token_embedd_cos,noun_q1,noun_q2))
    fea_nun = np.vstack([noun_di,noun_sim]).T
    print(fea_nun.shape)

    return np.hstack([fea_prob,fea_cluster,fea_pos,fea_dep,fea_en,fea_svo,fea_nun])


train_x = generate_feature(train_value)
pd.to_pickle(train_x,'../X_v2/train_spacy.pkl')

for i in range(len(batch_test)):
    test_x = generate_feature(batch_test[i])
    print(test_x.shape)
    pd.to_pickle(test_x,'../X_v2/test_spacy{0}.pkl'.format(i))


#select_index = list(range(0,33)) + list(range(36,39))
#sps.spearmanr(fea.max(axis=1),y_train[0:400000])[0]

