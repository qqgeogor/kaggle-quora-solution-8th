import numpy as np
import pandas as pd
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
from .utils import dist_utils,split_data

seed = 1024
np.random.seed(seed)

path = '../data/'
train = pd.read_pickle(path+'train_clean.pkl')
test = pd.read_pickle(path+'test_clean.pkl')
data_all = pd.concat([train, test])


class VectorSpace:
    ## char based
    def _init_char_tfidf(self, include_digit=True):# should include digit
        chars = list(string.ascii_lowercase)
        if include_digit:
            chars += list(string.digits)
        vocabulary = dict(zip(chars, range(len(chars))))
        tfidf = TfidfVectorizer(strip_accents="unicode",
                                analyzer="char",
                                norm=None,
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, 1),
                                use_idf=0,
                                vocabulary=vocabulary)
        return tfidf

class CharDistribution(VectorSpace):
    def __init__(self, obs_corpus, target_corpus):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus

    def normalize(self, text):
        # pat = re.compile("[a-z0-9]")
        pat = re.compile("[a-z]")
        group = pat.findall(text.lower())
        if group is None:
            res = " "
        else:
            res = "".join(group)
            res += " "
        return res

    def preprocess(self, corpus):
        return [self.normalize(text) for text in corpus]

    def get_distribution(self):
        ## obs tfidf
        tfidf = self._init_char_tfidf()
        X_obs = tfidf.fit_transform(self.preprocess(self.obs_corpus)).todense()
        X_obs = np.asarray(X_obs)
        # apply laplacian smoothing
        s = 1.
        X_obs = (X_obs + s) / (np.sum(X_obs, axis=1)[:, None] + X_obs.shape[1] * s)
        ## targetument tfidf
        tfidf = self._init_char_tfidf()
        X_target = tfidf.fit_transform(self.preprocess(self.target_corpus)).todense()
        X_target = np.asarray(X_target)
        X_target = (X_target + s) / (np.sum(X_target, axis=1)[:, None] + X_target.shape[1] * s)
        self.X_obs = X_obs
        self.X_target = X_target
        return X_obs, X_target

    def set_distribution(self,X_obs,X_target):
        self.X_obs = X_obs
        self.X_target = X_target

    def get_ratio(self):
        self.const_A = 1.
        self.const_B = 1.
        ratio = (self.X_obs + self.const_A) / (self.X_target + self.const_B)
        return ratio

    def get_cossim(self):
        sim = list(map(dist_utils._cosine_sim, self.X_obs, self.X_target))
        sim = np.asarray(sim).squeeze()
        return sim

    def get_KL(self):
        kl = dist_utils._KL(self.X_obs, self.X_target)
        return kl


cd = CharDistribution(data_all['clean_question1'].values.tolist(),
                      data_all['clean_question2'].values.tolist())

q1_char_tfidf,q2_char_tfidf = cd.get_distribution()
char_distribution_ratio = cd.get_ratio()
char_distribution_cos = cd.get_cossim()
char_distribution_KL = cd.get_KL()

char_distribution_all = np.hstack([q1_char_tfidf,q2_char_tfidf,char_distribution_ratio])
dis_char = np.vstack([char_distribution_cos,char_distribution_KL]).T
char_distribution_all = np.hstack([char_distribution_all,dis_char])

train_char_dis = char_distribution_all[:train.shape[0]]
test_char_dis = char_distribution_all[train.shape[0]:]

pd.to_pickle(train_char_dis,'../X_v2/train_char_dis.pkl')
test_x = split_data.split_test(test_char_dis)
for i in range(6):
    pd.to_pickle(test_x[i], '../X_v2/test_char_dis{0}.pkl'.format(i))