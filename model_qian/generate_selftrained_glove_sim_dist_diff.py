import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from glove import Glove
from glove import Corpus
from gensim import corpora, models, similarities
from config import path
seed = 1024
np.random.seed(seed)


ft = ['question1','question2','question1_porter','question2_porter']
train = pd.read_csv(path+"train_porter.csv")[ft].astype(str)
test = pd.read_csv(path+"test_porter.csv")[ft].astype(str)
# test['is_duplicated']=[-1]*test.shape[0]
data_all = pd.concat([train,test])

len_train = train.shape[0]

qs = []
ts = []
ds = []
sentences = []
for q,t in zip(data_all['question1'].values.tolist(),data_all['question2'].values.tolist()):
    sentences.append(q.split(' '))
    sentences.append(t.split(' '))
    qs.append(q.split(' '))
    ts.append(t.split(' '))



corpus_model = Corpus()
corpus_model.fit(sentences, window=10)
corpus_model.save(path+'corpus.mdl')

corpus_model = Corpus.load(path+'corpus.mdl')

glove = Glove(no_components=200, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
          no_threads=7, verbose=True)
glove.add_dictionary(corpus_model.dictionary)
glove.save(path+'glove.glv')
glove = Glove.load(path+'glove.glv')
print glove

qt_sims_dists = []
qt_diff = []


def calc_cosine_dist(text_a ,text_b, metric = 'euclidean'):
    return pairwise_distances([text_a], [text_b], metric = metric)[0][0]


qs = []
ts = []
ds = []
sentences = []
for q,t in zip(train['question1'].values.tolist(),train['question2'].values.tolist()):
    sentences.append(q.split(' '))
    sentences.append(t.split(' '))
    qs.append(q.split(' '))
    ts.append(t.split(' '))
for q,t in zip(qs,ts):
    q_vec = glove.transform_paragraph(q)
    t_vec = glove.transform_paragraph(t)
    qt_cos = calc_cosine_dist(q_vec,t_vec,'cosine')
    qt_dist = calc_cosine_dist(q_vec,t_vec,'euclidean')
    qt_sims_dists.append([qt_cos,qt_dist])


qt_sims_dists = np.array(qt_sims_dists)
pd.to_pickle(qt_sims_dists,path+'train_selftrained_glove_sim_dist.pkl')

qs = []
ts = []
ds = []
sentences = []
qt_sims_dists = []
qt_diff = []
for q,t in zip(test['question1'].values.tolist(),test['question2'].values.tolist()):
    sentences.append(q.split(' '))
    sentences.append(t.split(' '))
    qs.append(q.split(' '))
    ts.append(t.split(' '))
for q,t in zip(qs,ts):
    q_vec = glove.transform_paragraph(q)
    t_vec = glove.transform_paragraph(t)
    qt_cos = calc_cosine_dist(q_vec,t_vec,'cosine')
    qt_dist = calc_cosine_dist(q_vec,t_vec,'euclidean')
    qt_sims_dists.append([qt_cos,qt_dist])


qt_sims_dists = np.array(qt_sims_dists)
pd.to_pickle(qt_sims_dists,path+'test_selftrained_glove_sim_dist.pkl')
