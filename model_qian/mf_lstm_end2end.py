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
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Nadam
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU
from keras.models import Model
from keras.utils.visualize_util import plot
seed=1024
np.random.seed(seed)
from config import path
ft = ['question1','question2']
train = pd.read_csv(path+"train.csv")[ft].astype(str)
test = pd.read_csv(path+"test.csv")[ft].astype(str)
len_train = train.shape[0]

data_all = pd.concat([train,test])

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def batch_generator(q1,q2,y,batch_size=128,shuffle=True,maxlen=238):
    sample_size = q1.shape[0]
    index_array = np.arange(sample_size)
    
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]

            X_batch_1 = pad_sequences(q1[batch_ids],padding='pre',maxlen=maxlen)
            X_batch_2 = pad_sequences(q2[batch_ids],padding='pre',maxlen=maxlen)

            X_batch = [X_batch_1,X_batch_2]
            y_batch = y[batch_ids]
            # y_batch = [y_batch,y_batch]
            yield X_batch,y_batch
            
            # y_batch = y[batch_ids].reshape(-1,1)
            # X_batch.append(y_batch)
            # yield X_batch,np.zeros(y_batch.shape[0])


def test_batch_generator(q1,q2,y,batch_size=128,maxlen=238):
    sample_size = q1.shape[0]
    index_array = np.arange(sample_size)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        X_batch_1 = pad_sequences(q1[batch_ids],padding='pre',maxlen=maxlen)
        X_batch_2 = pad_sequences(q2[batch_ids],padding='pre',maxlen=maxlen)
        X_batch = [X_batch_1,X_batch_2]
        y_batch = np.zeros(X_batch_1.shape[0]).reshape(-1,1)
        # X_batch.append(y_batch)
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

def build_model(maxlen=238,n=1e5,dim=128,hidden=512):
    inputs = []
    inputs_q1 = Input(shape=(maxlen,),name='input_q1')
    inputs.append(inputs_q1)
    inputs_q2 = Input(shape=(maxlen,),name='input_q2')
    inputs.append(inputs_q2)

    # inputs_y = Input(shape=(1,),name='input_y')
    # inputs.append(inputs_y)

    shared_emb1 = Embedding(
                n,
                dim,
                input_length=maxlen,
                )
    # conv1 = Convolution1D(256, 5, border_mode='same',activation='relu')
    # pool1 = MaxPooling1D(pool_length=2)

    # conv2 = Convolution1D(128, 3, border_mode='same',activation='relu')
    # pool2 = MaxPooling1D(pool_length=2)

    # conv3 = Convolution1D(64, 3, border_mode='same',activation='relu')
    # pool3 = MaxPooling1D(pool_length=2)
    lstm1 = LSTM(256)
    # lstm2 = LSTM(256)


    emb_q1 = shared_emb1(inputs_q1)
    emb_q1 = Dropout(0.2)(emb_q1)
    emb_q2 = shared_emb1(inputs_q2)
    emb_q2 = Dropout(0.2)(emb_q2)

    latent_q1 = lstm1(emb_q1)
    latent_q2 = lstm1(emb_q2)
    
    # latent = Dense(128,activation='tanh')
    # latent_q1 = latent(flatten_q1)
    # latent_q2 = latent(flatten_q2)

    
    outputs_contrastive_loss = Lambda(euclidean_distance,output_shape=(1,),name='contrastive_loss')([
            latent_q1,latent_q2
            ])

    merge_layer = merge([latent_q1,latent_q2],mode='concat')
    
    fc = Dense(hidden)(merge_layer)
    fc = PReLU()(fc)
    fc = Dropout(0.5)(fc)
    output_logloss = Dense(1,activation='sigmoid',name='prediction_loss')(fc)
    
    # outputs = [output_logloss,outputs_contrastive_loss,]
    outputs = [output_logloss]
    model = Model(input=inputs, output=outputs)
    
    model.compile(
                optimizer='nadam',
                loss = {
                'prediction_loss':'binary_crossentropy',
                # 'contrastive_loss':contrastive_loss,s
                }
              )
    
    return model

maxlen = 40
n = 2**18
q1 = data_all['question1'].apply(lambda x:one_hot(x,n=n, lower=True, split=" ")).values#.tolist()
q2 = data_all['question2'].apply(lambda x:one_hot(x,n=n, lower=True, split=" ")).values#.tolist()
# texts  = data_all['question1'].values.tolist()+data_all['question2'].values.tolist()
# token = Tokenizer()
# token.fit_on_texts(texts)
# q1 = token.texts_to_sequences(data_all['question1'].values.tolist())
# q2 = token.texts_to_sequences(data_all['question2'].values.tolist())
print q1[:10]


X_q1 = q1[:len_train]
X_t_q1 = q1[len_train:]

X_q2 = q2[:len_train]
X_t_q2 = q2[len_train:]

# X_q1 = np.array(X_q1)
# X_t_q1 = np.array(X_t_q1)
# X_q2 = np.array(X_q2)
# X_t_q2 = np.array(X_t_q2)

y = pd.read_csv(path+"train.csv")['is_duplicate'].values
# y[y==0]=-1

# skf = KFold(n_splits=5, shuffle=True, random_state=seed).split(X_q1)
X_mf = np.zeros(X_q1.shape[0])
X_t_mf = np.zeros(X_t_q1.shape[0])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_q1,y)
for ind_tr, ind_te in skf:
    X_q1_train = X_q1[ind_tr]
    X_q2_train = X_q2[ind_tr]
    X_q1_test = X_q1[ind_te]
    X_q2_test = X_q2[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]




    model = build_model(maxlen=maxlen,n=n,dim=256,hidden=256)
    model_name = 'lstm_emb.hdf5'
    plot(model, to_file=path+'%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True)

    batch_size = 384
    tr_gen = batch_generator(X_q1_train,X_q2_train,y_train,batch_size=batch_size,shuffle=True,maxlen=maxlen)
    te_gen = batch_generator(X_q1_test,X_q2_test,y_test,batch_size=batch_size,shuffle=False,maxlen=maxlen)
    model.fit_generator(
            tr_gen, 
            # samples_per_epoch=X_q1_train.shape[0], 
            steps_per_epoch = int(X_q1_train.shape[0]/batch_size),
            nb_epoch=2, 
            verbose=1, 
            validation_data=te_gen, 
            # nb_val_samples=X_q1_test.shape[0], 
            validation_steps = int(X_q1_test.shape[0]/batch_size),
            max_q_size=10,
            callbacks = [model_checkpoint]
            )
    
    y_pred = []
    test_gen = test_batch_generator(X_q1_test,X_q2_test,y_test,batch_size=batch_size*3,maxlen=maxlen)
    for X_batch,y_batch in test_gen:
        y_p = model.predict_on_batch(X_batch).ravel()
        y_pred.append(y_p)
    y_pred = np.concatenate(y_pred).ravel()
    X_mf[ind_te]+=y_pred
    from sklearn.metrics import log_loss
    score = log_loss(y_test, y_pred)
    print score
    
    y_pred = []
    test_gen = test_batch_generator(X_t_q1,X_t_q2,np.zeros(len(X_t_q1)),batch_size=batch_size*3,maxlen=maxlen)
    for X_batch,y_batch in test_gen:
        y_p = model.predict_on_batch(X_batch).ravel()
        y_pred.append(y_p)
    y_pred = np.concatenate(y_pred).ravel()
    X_t_mf+=y_pred

X_t_mf /=5.0
pd.to_pickle(X_mf,path+'X_mf_lstm_end2end.pkl')
pd.to_pickle(X_t_mf,path+'X_t_mf_lstm_end2end.pkl')
