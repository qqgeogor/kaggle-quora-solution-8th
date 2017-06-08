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

parser  = English()
vector_size = 100
glove_dir = path + 'glove.6B.{0}d.txt'.format(vector_size)

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



spacy_gen = spacy_generator()
spacy_gen.fit(train_value)


pos_q1,pos_q2 = spacy_gen.generate_token_pos()
dep_q1,dep_q2 = spacy_gen.generate_token_dep()

pd.to_pickle(pos_q1,'../X_v2/train_pos_q1.pkl')
pd.to_pickle(pos_q2,'../X_v2/train_pos_q2.pkl')
pd.to_pickle(dep_q1,'../X_v2/train_dep_q1.pkl')
pd.to_pickle(dep_q2,'../X_v2/train_dep_q2.pkl')

for i in range(6):
    spacy_gen.fit(batch_test[i])
    pos_q1, pos_q2 = spacy_gen.generate_token_pos()
    dep_q1, dep_q2 = spacy_gen.generate_token_dep()

    pd.to_pickle(pos_q1, '../X_v2/test_pos_q1{0}.pkl'.format(i))
    pd.to_pickle(pos_q2, '../X_v2/test_pos_q2{0}.pkl'.format(i))
    pd.to_pickle(dep_q1, '../X_v2/test_dep_q1{0}.pkl'.format(i))
    pd.to_pickle(dep_q2, '../X_v2/test_dep_q2{0}.pkl'.format(i))


