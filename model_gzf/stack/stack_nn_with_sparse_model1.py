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

tf_idf_co_train=pd.read_pickle('data/train_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_dis_co_train=pd.read_pickle('data/train_distinct_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_q1_unigram_train=pd.read_pickle('data/train_question1_unigram_tfidf.pkl')
tf_idf_q2_unigram_train=pd.read_pickle('data/train_question2_unigram_tfidf.pkl')
tf_idf_q1_bigram_train=pd.read_pickle('data/train_question1_bigram_tfidf.pkl')
tf_idf_q2_bigram_train=pd.read_pickle('data/train_question2_bigram_tfidf.pkl')
train_pattern=pd.read_pickle('data/bowen/train.pattern.onehot.pkl')

X_sparse_train=ssp.hstack([
			    tf_idf_co_train,
                            tf_idf_dis_co_train,
                            tf_idf_q1_unigram_train,
                            tf_idf_q2_unigram_train,
                            tf_idf_q1_bigram_train,
                            tf_idf_q2_bigram_train,
                            train_pattern]).tocsr()


print X_sparse_train.shape,X_raw_train.shape
LEN_RAW_INPUT=X_raw_train.shape[1]
LEN_SPARSE_INPUT=X_sparse_train.shape[1]
X_sparse_train=X_sparse_train.astype('float32')
X_raw_train=X_raw_train.astype('float32')

def MLP(opt='nadam'):

    X_raw=Input(shape=(LEN_RAW_INPUT,),name='input_raw')
    X_sparse=Input(shape=(LEN_SPARSE_INPUT,),name='input_sparse',sparse=True)
    
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
    auxiliary_output_sparse = Dense(1, activation='sigmoid',W_regularizer=regularizers.l2(1e-6),name='aux_output_sparse')(X_sparse)

    fc=merge([fc1,auxiliary_output_sparse],mode='concat')

    output_all = Dense(1,activation='sigmoid',name='output')(fc)
    model=Model(input=[X_raw,X_sparse],output=[output_all,auxiliary_output_dense,auxiliary_output_sparse])
    model.compile(
                optimizer=opt,
                loss = 'binary_crossentropy',
		loss_weights=[1.0,.40,0.60])
    return model


fold=0
te_pred=np.zeros(X_raw_train.shape[0])
skf = StratifiedKFold(y,n_folds=5, shuffle=True, random_state=1024)
for ind_tr, ind_te in skf:
    break
    X_tr_raw=X_raw_train[ind_tr]
    X_tr_sparse=X_sparse_train[ind_tr]
    y_tr=y[ind_tr]
    X_te_raw=X_raw_train[ind_te]
    X_te_sparse=X_sparse_train[ind_te]
    y_te=y[ind_te]
    print X_tr_raw.shape,X_te_raw.shape,X_te_sparse.shape,X_tr_sparse.shape,y_tr.mean(),y_te.mean()
    model=MLP('adam')
    model.fit([X_tr_raw,X_tr_sparse] ,[y_tr]*3,   
                     validation_data=([X_te_raw,X_te_sparse] ,[y_te]*3),      
                     nb_epoch=5, batch_size=128, shuffle=True) 
    te_pred[ind_te]=model.predict([X_te_raw,X_te_sparse])[0]
    print('end fold:{}'.format(fold))    
    fold+=1


#pd.to_pickle(te_pred,'stack/mf_3/nn_model1.train')
           
model=MLP('adam')
model.fit([X_raw_train,X_sparse_train],[y]*3,nb_epoch=5,batch_size=128,shuffle=True)


X_raw_test=pd.DataFrame(X_raw_test).fillna(0.0).values
X_raw_test=mm.transform(X_raw_test)

X_raw_test=np.hstack([X_raw_test,
                        test_tfidf])
print X_raw_test.shape

tf_idf_co_test=pd.read_pickle('data/test_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_dis_co_test=pd.read_pickle('data/test_distinct_question1_unigram_question2_unigram_tfidf.pkl')
tf_idf_q1_unigram_test=pd.read_pickle('data/test_question1_unigram_tfidf.pkl')
tf_idf_q2_unigram_test=pd.read_pickle('data/test_question2_unigram_tfidf.pkl')
tf_idf_q1_bigram_test=pd.read_pickle('data/test_question1_bigram_tfidf.pkl')
tf_idf_q2_bigram_test=pd.read_pickle('data/test_question2_bigram_tfidf.pkl')
test_pattern=pd.read_pickle('data/bowen/test.pattern.onehot.pkl')



X_sparse_test=ssp.hstack([
                            tf_idf_co_test,
                            tf_idf_dis_co_test,
                            tf_idf_q1_unigram_test,
                            tf_idf_q2_unigram_test,
                            tf_idf_q1_bigram_test,
                            tf_idf_q2_bigram_test,
                            test_pattern]).tocsr()


print X_sparse_test.shape,X_raw_test.shape
#LEN_RAW_INPUT=X_raw_test.shape[1]
#LEN_SPARSE_INPUT=X_sparse_test.shape[1]
X_sparse_test=X_sparse_test.astype('float32')
X_raw_test=X_raw_test.astype('float32')

ans=model.predict([X_raw_test,X_sparse_test])[0]
pd.to_pickle(ans,'stack/mf_3/nn_model1.test')



