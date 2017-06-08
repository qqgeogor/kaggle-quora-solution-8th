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
from config import path
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

def read_emb(path):
    count=0
    f = open(path,'r')
    emb_dict = dict()
    for line in f:
        #     continue
        line = line.strip().split(' ')
        id = line[0]
        
        weights = line[1:]
        weights = np.array([float(i) for i in weights])
        count+=1
        emb_dict[id] = weights
    return emb_dict

trmodel = Word2Vec.load_word2vec_format(path+'GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
w2vmodel = read_emb(path+'glove.840B.300d.txt')


idf_dict = pickle.load(open(path+'idf_dict.pkl','rb'))


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def my_padding(X,maxlen,dim=300):
    if len(X)>=maxlen:
        X = X[-maxlen:]
    else:
        X = [np.zeros(dim)]*(maxlen-len(X))+X
    return X

def batch_generator(X,q1,q2,y,batch_size=128,shuffle=True,maxlen=238,dim=300):
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

                qqq3,qqq4=calc_w2v_sim(qq1,qq2,embedder=trmodel,maxlen=maxlen,dim=dim)
                # 
                X_batch_3.append(qqq1)
                X_batch_4.append(qqq2)

            X_batch_1 = np.array(X_batch_1)
            X_batch_2 = np.array(X_batch_2)

            X_batch_3 = np.array(X_batch_3)
            X_batch_4 = np.array(X_batch_4)

            X_batch = [X_batch_1,X_batch_2,X_batch_3,X_batch_4]

            y_batch = y[batch_ids]
            
            yield X_batch,y_batch


def calc_w2v_sim(q1,q2,embedder=w2vmodel,idf_dict=idf_dict,maxlen=50,dim=300):
    '''
    Calc w2v similarities and diff of centers of query\title
    '''
    a2 = [x for x in q1.lower().split() if x in embedder]
    b2 = [x for x in q2.lower().split() if x in embedder]


    vectorAs = []
    for w in a2:
        if w in idf_dict:
            coef = idf_dict[w]
        else:
            coef = idf_dict['default_idf']
        coef=1.0
        vectorA = embedder[w]
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

    vectorAs = my_padding(vectorAs,maxlen=maxlen,dim=dim)
    vectorBs = my_padding(vectorBs,maxlen=maxlen,dim=dim)
    vectorAs = np.vstack(vectorAs)

    vectorBs = np.vstack(vectorBs)
    return vectorAs,vectorBs
 
def test_batch_generator(X,q1,q2,y,batch_size=128,shuffle=True,maxlen=238,dim=300):
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

            qqq3,qqq4=calc_w2v_sim(qq1,qq2,embedder=trmodel,maxlen=maxlen,dim=dim)
            # 
            X_batch_3.append(qqq1)
            X_batch_4.append(qqq2)


        X_batch_1 = np.array(X_batch_1)
        X_batch_2 = np.array(X_batch_2)

        X_batch_3 = np.array(X_batch_3)
        X_batch_4 = np.array(X_batch_4)

        X_batch = [X_batch_1,X_batch_2,X_batch_3,X_batch_4]

        y_batch = y[batch_ids]
        
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

    inputs_q1 = Input(shape=(maxlen,dim),name='input_q1')
    inputs.append(inputs_q1)
    inputs_q2 = Input(shape=(maxlen,dim),name='input_q2')
    inputs.append(inputs_q2)

    inputs_q3 = Input(shape=(maxlen,dim),name='input_q3')
    inputs.append(inputs_q3)
    inputs_q4 = Input(shape=(maxlen,dim),name='input_q4')
    inputs.append(inputs_q4)





    emb_q1 = inputs_q1
    emb_q2 = inputs_q2


    emb_q3 = inputs_q3
    emb_q4 = inputs_q4
   
    lstm1 = LSTM(256,dropout=0.1, recurrent_dropout=0.05,return_sequences=True)
    lstm2 = LSTM(256,dropout=0.1, recurrent_dropout=0.05,return_sequences=True)



    
    atten1 = Attention(maxlen)
    atten2 = Attention(maxlen)

    latent_q1 = lstm1(emb_q1)
    latent_q2 = lstm1(emb_q2)
    latent_q1 = atten1(latent_q1)
    latent_q2 = atten1(latent_q2)

    latent_q3 = lstm2(emb_q3)
    latent_q4 = lstm2(emb_q4)

    latent_q3 = atten2(latent_q3)
    latent_q4 = atten2(latent_q4)


    latent = Dense(128,activation='tanh')
    flatten_q1 = merge([latent_q1,latent_q3,],mode='concat')
    flatten_q2 = merge([latent_q2,latent_q4,],mode='concat')

    latent_q1 = latent(flatten_q1)
    latent_q2 = latent(flatten_q2)
    
    outputs_contrastive_loss = Lambda(euclidean_distance,output_shape=(1,),name='contrastive_loss')([
            latent_q1,latent_q2
            ])


    outputs = [outputs_contrastive_loss,]
    

    model = Model(input=inputs, output=outputs)
    
    model.compile(
                optimizer='rmsprop',
                loss = {
                # 'prediction_loss':'binary_crossentropy',
                'contrastive_loss':contrastive_loss,
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


    model = build_model(dim0=dim0,maxlen=maxlen,n=n,dim=300,hidden=128)
    model_name = 'cnn1d.hdf5'
    plot(model, to_file=path+'%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True)

    batch_size = 384
    tr_gen = batch_generator(X_train,X_q1_train,X_q2_train,y_train,batch_size=batch_size,shuffle=True,maxlen=maxlen,dim=300)
    te_gen = batch_generator(X_test,X_q1_test,X_q2_test,y_test,batch_size=batch_size,shuffle=False,maxlen=maxlen,dim=300)
    model.fit_generator(
            tr_gen, 
            steps_per_epoch = int(X_q1_train.shape[0]/batch_size),
            nb_epoch=12, 
            verbose=1, 
            validation_data=te_gen, 
            validation_steps = int(X_q1_test.shape[0]/batch_size),
            max_q_size=10,
            callbacks = [model_checkpoint]
            )
    y_pred = []
    test_gen = test_batch_generator(X_test,X_q1_test,X_q2_test,y_test,batch_size=batch_size*3,shuffle=False,maxlen=maxlen,dim=300)
    for X_batch,y_batch in test_gen:
        y_p = model.predict_on_batch(X_batch).ravel()
        y_pred.append(y_p)
    y_pred = np.concatenate(y_pred).ravel()
    X_mf[ind_te]+=y_pred
    from sklearn.metrics import log_loss
    score = log_loss(y_test, y_pred)
    print score

    y_pred = []
    test_gen = test_batch_generator(X_t_q1,X_t_q1,X_t_q2,np.zeros(X_t_q1.shape[0]),batch_size=batch_size*3,shuffle=False,maxlen=maxlen,dim=300)
    for X_batch,y_batch in test_gen:
        y_p = model.predict_on_batch(X_batch).ravel()
        y_pred.append(y_p)
    y_pred = np.concatenate(y_pred).ravel()
    X_t_mf+=y_pred

X_t_mf /=5.0
pd.to_pickle(X_mf,path+'X_mf_lstm_att_sia.pkl')
pd.to_pickle(X_t_mf,path+'X_t_mf_lstm_att_sia.pkl')
