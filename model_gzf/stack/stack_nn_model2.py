import os
import sys
import tqdm
import h5py 
import numpy as np
import pandas as pd
import scipy.sparse as ssp
from sklearn.cross_validation import StratifiedKFold

np.random.seed(1123)
reload(sys)
sys.setdefaultencoding('utf-8')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Merge,Flatten,Input,merge
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D,MaxPooling1D
from keras.optimizers import Nadam,Adam,SGD
from keras.layers.advanced_activations import PReLU
from keras import regularizers
from pre_data import get_tfidf_train,get_other_train,get_tfidf_test,get_other_test

train_tfidf=get_tfidf_train()
X_raw_train=get_other_train()
test_tfidf=get_tfidf_test()
X_raw_test=get_other_test()

y=pd.read_csv('data/train.csv')['is_duplicate'].values

from sklearn.preprocessing import StandardScaler,MinMaxScaler
X_raw_train=pd.DataFrame(X_raw_train).fillna(0.0).values

mm=MinMaxScaler()
mm.fit(X_raw_train)
X_raw_train=mm.transform(X_raw_train)

X_raw_train=np.hstack([X_raw_train,
			train_tfidf])
print X_raw_train.shape

LEN_RAW_INPUT=X_raw_train.shape[1]
X_raw_train=X_raw_train.astype('float32')

def MLP(opt='nadam'):

    X_raw=Input(shape=(LEN_RAW_INPUT,),name='input_raw')
    
    fc1=BatchNormalization()(X_raw)
    fc1=Dense(512)(fc1)
    fc1=PReLU()(fc1)
    fc1=Dropout(0.25)(fc1)

    fc1=BatchNormalization()(fc1)
    fc1=Dense(256)(fc1)
    fc1=PReLU()(fc1)
    fc1=Dropout(0.15)(fc1)

    fc1=BatchNormalization()(fc1)
    auxiliary_output_dense = Dense(1, activation='sigmoid', name='aux_output_dense')(fc1)


    output_all = Dense(1,activation='sigmoid',name='output')(fc1)
    model=Model(input=X_raw,output=output_all)
    model.compile(
                optimizer=opt,
                loss = 'binary_crossentropy')
    return model


fold=0
te_pred=np.zeros(X_raw_train.shape[0])
skf = StratifiedKFold(y,n_folds=5, shuffle=True, random_state=1024)
for ind_tr, ind_te in skf:
    #break
    X_tr_raw=X_raw_train[ind_tr]
    y_tr=y[ind_tr]
    X_te_raw=X_raw_train[ind_te]
    y_te=y[ind_te]
    print X_tr_raw.shape,X_te_raw.shape,y_tr.mean(),y_te.mean()
    model=MLP('adam')
    model.fit(X_tr_raw ,y_tr,   
                     validation_data=(X_te_raw ,y_te),      
                     nb_epoch=6, batch_size=128, shuffle=True) 
    te_pred[ind_te]=model.predict(X_te_raw)
    print('end fold:{}'.format(fold))    
    fold+=1


pd.to_pickle(te_pred,'stack/mf_3/nn_model2.train')
           
model=MLP('adam')
model.fit(X_raw_train,y,nb_epoch=6,batch_size=128,shuffle=True)


X_raw_test=pd.DataFrame(X_raw_test).fillna(0.0).values
X_raw_test=mm.transform(X_raw_test)

X_raw_test=np.hstack([X_raw_test,
                        test_tfidf])
print X_raw_test.shape
#LEN_RAW_INPUT=X_raw_test.shape[1]
#LEN_SPARSE_INPUT=X_sparse_test.shape[1]
X_raw_test=X_raw_test.astype('float32')

ans=model.predict(X_raw_test)
pd.to_pickle(ans,'stack/mf_3/nn_model2.test')



