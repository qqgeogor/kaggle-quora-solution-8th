import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, Embedding, LSTM,GRU, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Nadam
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.utils.visualize_util import plot
from gensim.models import Word2Vec
from config import path,large_path
import spacy
nlp = spacy.load("en_default")
import pickle
seed=1024
np.random.seed(seed)

ft = ['question1','question2']
train = pd.read_csv(path+"train.csv")[ft].astype(str)
test = pd.read_csv(path+"test.csv")[ft].astype(str)
len_train = train.shape[0]

data_all = pd.concat([train,test])

# w2vmodel = Word2Vec.load(path+'my_w2v.mdl')

# w2vmodel = Word2Vec.load_word2vec_format(path+'GoogleNews-vectors-negative200.bin', binary=True)  # C binary format

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

# trmodel = Word2Vec.load_word2vec_format(path+'GoogleNews-vectors-negative200.bin', binary=True)  # C binary format
# w2vmodel = read_emb(path+'glove.840B.200d.txt')

# w2vmodel = Word2Vec.load_word2vec_format(large_path+'fasttext', binary=False)
w2vmodel= Word2Vec.load(path+'my_w2v.mdl')
# w2vmodel = read_emb(path+'glove.840B.200d.txt')
from glove import Glove
trmodel  = Glove.load(path+'glove.glv')
import collections
def get_vec(paragraph,model):
    cooccurrence = collections.defaultdict(lambda: 0.0)

    for token in paragraph:
        try:
            cooccurrence[model.dictionary[token]] += model.max_count / 10.0
        except KeyError:
            # print(token)
            pass
            # if not ignore_missing:
                # raise
    word_ids = np.array(cooccurrence.keys(), dtype=np.int32)
    return model.word_vectors[word_ids].ravel()

idf_dict = pickle.load(open(path+'idf_dict.pkl','rb'))


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def my_padding(X,maxlen,dim=200):
    if len(X)>=maxlen:
        X = X[-maxlen:]
    else:
        # X = [np.random.uniform(size=dim)]*(maxlen-len(X))+X
        X = [np.zeros(dim)]*(maxlen-len(X))+X
    return X

def batch_generator(X,q1,q2,y,batch_size=128,shuffle=True,maxlen=238,dim=200):
    sample_size = q1.shape[0]
    index_array = np.arange(sample_size)
    
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch_0 = X[batch_ids]
            X_batch_1 = []
            X_batch_2 = []

            X_batch_3 = []
            X_batch_4 = []

            for qq1,qq2 in zip(q1[batch_ids],q2[batch_ids]):
                qqq1,qqq2=calc_w2v_sim(qq1,qq2,maxlen=maxlen,dim=dim)
# 
                X_batch_1.append(qqq1)
                X_batch_2.append(qqq2)

                qqq3,qqq4=calc_w2v_sim2(qq1,qq2,embedder=trmodel,maxlen=maxlen,dim=dim)
                # 
                X_batch_3.append(qqq1)
                X_batch_4.append(qqq2)
                # print qq1.shape

            X_batch_1 = np.array(X_batch_1)
            X_batch_2 = np.array(X_batch_2)

            X_batch_3 = np.array(X_batch_3)
            X_batch_4 = np.array(X_batch_4)

            # X_batch_3 = X_batch_2-X_batch_1
            
            # X_batch = [X_batch_0,X_batch_1,X_batch_2,X_batch_3]
            X_batch = [X_batch_1,X_batch_2,X_batch_3,X_batch_4]

            y_batch = y[batch_ids]
            # y_batch = [y_batch,y_batch]
            
            yield X_batch,y_batch


def calc_w2v_sim2(q1,q2,embedder=trmodel,idf_dict=idf_dict,maxlen=50,dim=200):
    '''
    Calc w2v similarities and diff of centers of query\title
    '''
    a2 = [x for x in q1.lower().split() ]
    b2 = [x for x in q2.lower().split()]
    # tmp = []
    # for u in q1.lower().split():
    #     try:
    #         tmp.append(unicode(u))
    #     except:
    #         tmp.append("unicode_%s"%hash(u))
    # q1 = ' '.join(tmp)


    # tmp = []
    # for u in q2.lower().split():
    #     try:
    #         tmp.append(unicode(u))
    #     except:
    #         tmp.append("unicode_%s"%hash(u))
    # q2 = ' '.join(tmp)

    vectorAs = []
    for w in a2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        coef=1.0
        vectorA =  get_vec([w],embedder)
        if vectorA.shape[0]==0:
            continue
        vectorAs.append(vectorA)
        # print(vectorA.shape)
    
    
    vectorBs = []
    for w in b2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        coef=1.0
        vectorB = get_vec([w],embedder)
        if vectorB.shape[0]==0:
            continue
        vectorBs.append(vectorB)
        # print(vectorB.shape)

    # vectorAs =[s.vector*idf_dict.get(s.text,idf_dict['default_idf']) for s in nlp(unicode(q1))]
    # vectorBs =[s.vector*idf_dict.get(s.text,idf_dict['default_idf']) for s in nlp(unicode(q2))]
    # vectorAs =[s.vector for s in nlp(unicode(q1))]
    # vectorBs =[s.vector for s in nlp(unicode(q2))]
    vectorAs = my_padding(vectorAs,maxlen=maxlen,dim=dim)
    vectorBs = my_padding(vectorBs,maxlen=maxlen,dim=dim)
    vectorAs = np.vstack(vectorAs)
    # print(vectorAs)
    vectorBs = np.vstack(vectorBs)
    return vectorAs,vectorBs
 

def calc_w2v_sim(q1,q2,embedder=w2vmodel,idf_dict=idf_dict,maxlen=50,dim=200):
    '''
    Calc w2v similarities and diff of centers of query\title
    '''
    a2 = [x for x in q1.lower().split()  if x in embedder]
    b2 = [x for x in q2.lower().split()  if x in embedder]
    # tmp = []
    # for u in q1.lower().split():
    #     try:
    #         tmp.append(unicode(u))
    #     except:
    #         tmp.append("unicode_%s"%hash(u))
    # q1 = ' '.join(tmp)


    # tmp = []
    # for u in q2.lower().split():
    #     try:
    #         tmp.append(unicode(u))
    #     except:
    #         tmp.append("unicode_%s"%hash(u))
    # q2 = ' '.join(tmp)

    vectorAs = []
    for w in a2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        coef=1.0
        vectorA =  embedder[w]
        vectorAs.append(vectorA)
    
    vectorBs = []
    for w in b2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        coef=1.0
        vectorB = embedder[w]
        vectorBs.append(vectorB)
    # vectorAs =[s.vector*idf_dict.get(s.text,idf_dict['default_idf']) for s in nlp(unicode(q1))]
    # vectorBs =[s.vector*idf_dict.get(s.text,idf_dict['default_idf']) for s in nlp(unicode(q2))]
    # vectorAs =[s.vector for s in nlp(unicode(q1))]
    # vectorBs =[s.vector for s in nlp(unicode(q2))]
    vectorAs = my_padding(vectorAs,maxlen=maxlen,dim=dim)
    vectorBs = my_padding(vectorBs,maxlen=maxlen,dim=dim)
    vectorAs = np.vstack(vectorAs)
    # print(vectorAs)
    vectorBs = np.vstack(vectorBs)
    return vectorAs,vectorBs
 
def test_batch_generator(X,q1,q2,y,batch_size=128,shuffle=True,maxlen=238,dim=200):
    sample_size = q1.shape[0]
    index_array = np.arange(sample_size)
    

    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        X_batch_0 = X[batch_ids]
        X_batch_1 = []
        X_batch_2 = []

        X_batch_3 = []
        X_batch_4 = []

        for qq1,qq2 in zip(q1[batch_ids],q2[batch_ids]):
            qqq1,qqq2=calc_w2v_sim(qq1,qq2,maxlen=maxlen,dim=dim)
# 
            X_batch_1.append(qqq1)
            X_batch_2.append(qqq2)

            qqq3,qqq4=calc_w2v_sim2(qq1,qq2,embedder=trmodel,maxlen=maxlen,dim=dim)
            # 
            X_batch_3.append(qqq1)
            X_batch_4.append(qqq2)
            # print qq1.shape

        X_batch_1 = np.array(X_batch_1)
        X_batch_2 = np.array(X_batch_2)

        X_batch_3 = np.array(X_batch_3)
        X_batch_4 = np.array(X_batch_4)

        # X_batch_3 = X_batch_2-X_batch_1
        
        # X_batch = [X_batch_0,X_batch_1,X_batch_2,X_batch_3]
        X_batch = [X_batch_1,X_batch_2,X_batch_3,X_batch_4]

        y_batch = y[batch_ids]
        # y_batch = [y_batch,y_batch]
        
        yield X_batch,y_batch

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return y_true * K.log(y_true / y_pred)

def myloss(x):
    q1,q2,y = x
    return K.mean(kullback_leibler_divergence(q1,q2)*y)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

# coding=utf8
from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim


def build_model(dim0,maxlen=238,n=1e5,dim=200,hidden=512):
    inputs = []
    # inputs_0 = Input(shape=(dim0,),name='input_0')
    # inputs.append(inputs_0)
    inputs_q1 = Input(shape=(maxlen,dim),name='input_q1')
    inputs.append(inputs_q1)
    inputs_q2 = Input(shape=(maxlen,dim),name='input_q2')
    inputs.append(inputs_q2)

    inputs_q3 = Input(shape=(maxlen,dim),name='input_q3')
    inputs.append(inputs_q3)
    inputs_q4 = Input(shape=(maxlen,dim),name='input_q4')
    inputs.append(inputs_q4)


    # inputs_q3 = Input(shape=(maxlen,dim),name='input_q3')
    # inputs.append(inputs_q3)

    conv1 = Convolution1D(64, 3, border_mode='same',activation='relu')
    pool1 = MaxPooling1D(pool_length=2)

    conv2 = Convolution1D(128, 3, border_mode='same',activation='relu')
    pool2 = MaxPooling1D(pool_length=2)

    # conv3 = Convolution1D(256, 3, border_mode='same',activation='relu')
    # pool3 = MaxPooling1D(pool_length=2)

    emb_q1 = inputs_q1
    emb_q2 = inputs_q2


    emb_q3 = inputs_q3
    emb_q4 = inputs_q4
    # emb_q3 = inputs_q3


    # conv1_q1 = conv1(emb_q1)
    # conv1_q2 = conv1(emb_q2)
    # pool1_q1 = pool1(conv1_q1)
    # pool1_q2 = pool1(conv1_q2)

    # conv2_q1 = conv2(pool1_q1)
    # conv2_q2 = conv2(pool1_q2)
    # pool2_q1 = pool2(conv2_q1)
    # pool2_q2 = pool2(conv2_q2)

    # conv3_q1 = conv3(pool2_q1)
    # conv3_q2 = conv3(pool2_q2)
    # pool3_q1 = pool3(conv3_q1)
    # pool3_q2 = pool3(conv3_q2)

    # flatten = Flatten()
    # flatten_q1 = flatten(pool3_q1)
    # flatten_q2 = flatten(pool3_q2)
    
    # latent = Dense(128,activation='sigmoid')
    # latent_q1 = latent(flatten_q1)
    # latent_q2 = latent(flatten_q2)
    
    # latent = Dense(128,activation='sigmoid')
    # latent_q1 = latent(flatten_q1)
    # latent_q2 = latent(flatten_q2)
    


    # lstm1 = LSTM(256,dropout=0.1, recurrent_dropout=0.05,return_sequences=True)
    # lstm2 = LSTM(256,dropout=0.1, recurrent_dropout=0.05,return_sequences=True)
    
    lstm1 = LSTM(256,dropout=0.1, recurrent_dropout=0.05,return_sequences=True)
    lstm2 = LSTM(256,dropout=0.1, recurrent_dropout=0.05,return_sequences=True)
    
    bilstm1 = (lstm1)
    bilstm2 = (lstm2)
    # lstm3 = LSTM(128)
    
    atten1 = Attention(maxlen)
    atten2 = Attention(maxlen)

    # emb_q1 = shared_emb1(inputs_q1)
    # emb_q1 = Dropout(0.2)(emb_q1)
    # emb_q2 = shared_emb1(inputs_q2)
    # emb_q2 = Dropout(0.2)(emb_q2)
    
    latent_q1 = bilstm1(emb_q1)
    latent_q2 = bilstm1(emb_q2)
    latent_q1 = atten1(latent_q1)
    latent_q2 = atten1(latent_q2)

    latent_q3 = bilstm2(emb_q3)
    latent_q4 = bilstm2(emb_q4)

    latent_q3 = atten2(latent_q3)
    latent_q4 = atten2(latent_q4)


    # latent_q3 = lstm3(emb_q3)
    # latent = Dense(128,activation='tanh')
    # flatten_q1 = merge([latent_q1,latent_q3,],mode='concat')
    # flatten_q2 = merge([latent_q2,latent_q4,],mode='concat')

    # latent_q1 = latent(flatten_q1)
    # latent_q2 = latent(flatten_q2)
    
    # outputs_contrastive_loss = Lambda(euclidean_distance,output_shape=(1,),name='contrastive_loss')([
    #         latent_q1,latent_q2
    #         ])

    merge_layer = merge([latent_q1,latent_q2,latent_q3,latent_q4],mode='concat')
    merge_layer = BatchNormalization()(merge_layer)

    fc1 = Dense(hidden)(merge_layer)
    fc1 = PReLU()(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.2)(fc1)

    fc1 = Dense(hidden)(fc1)
    fc1 = PReLU()(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.2)(fc1)

    output_logloss = Dense(1,activation='sigmoid',name='prediction_loss')(fc1)
    
    outputs = [output_logloss,]

    # outputs = [outputs_contrastive_loss,]
    
    # outputs = [output_logloss,outputs_contrastive_loss,]

    model = Model(input=inputs, output=outputs)
    
    model.compile(
                optimizer='rmsprop',
                loss = {
                'prediction_loss':'binary_crossentropy',
                # 'contrastive_loss':contrastive_loss,
                }
              )
    
    return model
maxlen = 40
n = 2**18

import string

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

len_train = train.shape[0]

feats= ['question1','question2']

corpus = []
for f in feats:
    train[f] = train[f].astype(str).apply(lambda x:remove_punctuation(x))
    test[f] = test[f].astype(str).apply(lambda x:remove_punctuation(x))

X_q1 = train['question1'].values
X_q2 = train['question2'].values

X_t_q1 = test['question1'].values
X_t_q2 = test['question2'].values

feats= ['question1_unigram','question2_unigram','question1_bigram','question2_bigram','question1_distinct_unigram','question2_distinct_unigram','question1_distinct_bigram','question2_distinct_bigram','question1_unigram_question2_unigram','question1_distinct_unigram_question2_distinct_unigram']
X = []
for f in feats:
    X.append(pd.read_pickle(path+'train_%s_tfidf_nmf.pkl'%f))
    # X.append(pd.read_pickle(path+'train_%s_tfidf_svd.pkl'%f))
# X.append(np.loadtxt(path+'train_spacy_diff_pretrained.txt'))

feats2= [
'question1',
'question2',
'question1_porter',
'question2_porter',
]
for f in feats2:
    X.append(pd.read_pickle(path+'train_%s_nmf.pkl'%f))
    # X.append(pd.read_pickle(path+'train_%s_svd.pkl'%f))

X = np.hstack(X)#.tocsr()


# train_unigram_features = pd.read_csv(path+'train_unigram_features.csv').values
# train_bigram_features = pd.read_csv(path+'train_bigram_features.csv').values
# train_distinct_unigram_features = pd.read_csv(path+'train_distinct_unigram_features.csv').values
# train_porter_stop_features = pd.read_csv(path+'train_porter_stop_features.csv').values

# train_position_index = pd.read_csv(path+'train_position_index.csv').values
# train_position_normalized_index = pd.read_csv(path+'train_position_normalized_index.csv').values
# train_idf_stats_features = pd.read_csv(path+'train_idf_stats_features.csv').values



# train_len =pd.read_pickle(path+'train_len.pkl')
# train_selftrained_w2v_sim_dist =pd.read_pickle(path+'train_selftrained_w2v_sim_dist.pkl')
# train_pretrained_w2v_sim_dist =pd.read_pickle(path+'train_pretrained_w2v_sim_dist.pkl')
# # train_selftrained_glove_sim_dist =pd.read_pickle(path+'train_selftrained_glove_sim_dist.pkl')
# train_gensim_tfidf_sim = pd.read_pickle(path+'train_gensim_tfidf_sim.pkl')[:].reshape(-1,1)

# train_hashed_idf = pd.read_csv(path+'train_hashed_idf.csv')
# train_hashed_idf['hash_count_same'] = (train_hashed_idf['question1_hash_count']==train_hashed_idf['question2_hash_count']).astype(int)
# train_hashed_idf['dup_max']=train_hashed_idf.apply(lambda x:max(x['question1_hash_count'],x['question2_hash_count']),axis=1)
# train_hashed_idf['dup_min']=train_hashed_idf.apply(lambda x:min(x['question1_hash_count'],x['question2_hash_count']),axis=1)
# train_hashed_idf['dup_dis']=train_hashed_idf['dup_max']-train_hashed_idf['dup_min']
# weight = train_hashed_idf['dup_max'].values
# train_hashed_idf = train_hashed_idf.values

# # train_distinct_word_stats = pd.read_csv(path+'train_distinct_word_stats.csv').values
# # train_distinct_word_stats_pretrained = pd.read_csv(path+'train_distinct_word_stats_pretrained.csv').values

# # train_spacy_sim_pretrained = pd.read_csv(path+'train_spacy_sim_pretrained.csv')[['spacy_sim']].values
# # print weight.shape
# # train_pattern = pd.read_pickle(path+'train.pattern.pkl').reshape((X.shape[0],3))
# # print train_pattern.shape
# # train_tfidf_sim = pd.read_pickle(path+'train_tfidf_sim.pkl').reshape(-1,1)

# train_basic_features = np.hstack([
#     train_unigram_features,
#     train_bigram_features,
#     train_porter_stop_features,
#     train_distinct_unigram_features,
#     train_position_index,
#     train_position_normalized_index,
#     train_idf_stats_features,
#     train_len,
#     train_selftrained_w2v_sim_dist,
#     train_pretrained_w2v_sim_dist,
#     # train_selftrained_glove_sim_dist,
#     train_gensim_tfidf_sim,
#     train_hashed_idf,
#     # train_distinct_word_stats,
#     # train_distinct_word_stats_pretrained,
#     # train_spacy_sim_pretrained,
#     # train_pattern,
#     ])

# X = np.hstack([X,train_basic_features])#.tocsr()
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
dim0=X.shape[1]


# X_q1 = np.array(X_q1)
# X_t_q1 = np.array(X_t_q1)
# X_q2 = np.array(X_q2)
# X_t_q2 = np.array(X_t_q2)

y = pd.read_csv(path+"train.csv")['is_duplicate'].values
# y[y==0]=-1

X_mf = np.zeros(X_q1.shape[0])
X_t_mf = np.zeros(X_t_q1.shape[0])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_q1,y)
for ind_tr, ind_te in skf:
    X_q1_train = X_q1[ind_tr]
    X_q2_train = X_q2[ind_tr]
    X_q1_test = X_q1[ind_te]
    X_q2_test = X_q2[ind_te]
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]


    model = build_model(dim0=dim0,maxlen=maxlen,n=n,dim=200,hidden=128)
    model_name = 'cnn1d.hdf5'
    plot(model, to_file=path+'%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True)

    batch_size = 384
    tr_gen = batch_generator(X_train,X_q1_train,X_q2_train,y_train,batch_size=batch_size,shuffle=True,maxlen=maxlen,dim=200)
    te_gen = batch_generator(X_test,X_q1_test,X_q2_test,y_test,batch_size=batch_size,shuffle=False,maxlen=maxlen,dim=200)
    model.fit_generator(
            tr_gen, 
            # samples_per_epoch=X_q1_train.shape[0], 
            steps_per_epoch = int(X_q1_train.shape[0]/batch_size),
            nb_epoch=6, 
            verbose=1, 
            validation_data=te_gen, 
            # nb_val_samples=X_q1_test.shape[0], 
            validation_steps = int(X_q1_test.shape[0]/batch_size),
            max_q_size=10,
            callbacks = [model_checkpoint]
            )
    # model.load_weights(path+model_name)
    
    y_pred = []
    test_gen = test_batch_generator(X_test,X_q1_test,X_q2_test,y_test,batch_size=batch_size*3,shuffle=False,maxlen=maxlen,dim=200)
    for X_batch,y_batch in test_gen:
        y_p = model.predict_on_batch(X_batch).ravel()
        y_pred.append(y_p)
    y_pred = np.concatenate(y_pred).ravel()
    X_mf[ind_te]+=y_pred
    from sklearn.metrics import log_loss
    score = log_loss(y_test, y_pred)
    print score

    y_pred = []
    test_gen = test_batch_generator(X_t_q1,X_t_q1,X_t_q2,np.zeros(X_t_q1.shape[0]),batch_size=batch_size*3,shuffle=False,maxlen=maxlen,dim=200)
    for X_batch,y_batch in test_gen:
        y_p = model.predict_on_batch(X_batch).ravel()
        y_pred.append(y_p)
    y_pred = np.concatenate(y_pred).ravel()
    X_t_mf+=y_pred

X_t_mf /=5.0
pd.to_pickle(X_mf,path+'X_mf_lstm_att.pkl')
pd.to_pickle(X_t_mf,path+'X_t_mf_lstm_att.pkl')
