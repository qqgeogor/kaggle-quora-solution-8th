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
from pre_data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


X_tfidf=get_tfidf_train()
X_other=get_other_train()

X_test_tfidf=get_tfidf_test()
X_test_other=get_other_test()

X_extra=get_extra_train()
X_test_extra=get_extra_test()


train_mf=get_mf_train()
test_mf=get_mf_test()

y=pd.read_csv('data/train.csv')['is_duplicate'].values

train_coo_bigram=pd.read_pickle('data/train_cooccurence_distinct_bigram_encoding_by_label.pkl')
train_coo_unigram=pd.read_pickle('data/train_cooccurence_distinct_encoding_by_label.pkl')
train_neigh_pos_v2=pd.read_pickle('data/train_neigh_pos_v2.pkl')
train_hashed_clique_stats_sep=pd.read_csv('data/train_hashed_clique_stats_sep.csv')

test_coo_bigram=pd.read_pickle('data/test_cooccurence_distinct_bigram_encoding_by_label.pkl')
test_coo_unigram=pd.read_pickle('data/test_cooccurence_distinct_encoding_by_label.pkl')
test_neigh_pos_v2=pd.read_pickle('data/test_neigh_pos_v2.pkl')
test_hashed_clique_stats_sep=pd.read_csv('data/test_hashed_clique_stats_sep.csv')


X_train=np.hstack([X_other,X_extra,
                train_coo_bigram,
                train_coo_unigram,
                train_neigh_pos_v2,
                train_hashed_clique_stats_sep])


print X_train.shape

X_test=np.hstack([X_test_other,X_test_extra,
                  test_coo_bigram,
                  test_coo_unigram,
                  test_neigh_pos_v2,
                  test_hashed_clique_stats_sep])

print X_test.shape

from sklearn.preprocessing import StandardScaler,MinMaxScaler
X_train=pd.DataFrame(X_train).fillna(0.0).values

mm=MinMaxScaler()
mm.fit(X_train)
X_train=mm.transform(X_train)

X_train=np.hstack([X_train,
		       X_tfidf,
                       train_mf])
print X_train.shape

LEN_RAW_INPUT=X_train.shape[1]
X_train=X_train.astype('float32')

X_test=pd.DataFrame(X_test).fillna(0.0).values
X_test=mm.transform(X_test)

X_test=np.hstack([X_test,
                  X_test_tfidf,
		test_mf])
print X_test.shape


def MLP(opt='nadam'):

    X_raw=Input(shape=(LEN_RAW_INPUT,),name='input_raw')
    
    fc1=BatchNormalization()(X_raw)
    fc1=Dense(256)(fc1)
    fc1=PReLU()(fc1)
    fc1=Dropout(0.2)(fc1)

    fc1=BatchNormalization()(fc1)
    fc1=Dense(256)(fc1)
    fc1=PReLU()(fc1)
    #fc1=Dropout(0.2)(fc1)

    fc1=BatchNormalization()(fc1)
    auxiliary_output_dense = Dense(1, activation='sigmoid', name='aux_output_dense')(fc1)


    output_all = Dense(1,activation='sigmoid',name='output')(fc1)
    model=Model(input=X_raw,output=output_all)
    model.compile(
                optimizer=opt,
                loss = 'binary_crossentropy')
    return model


#nadam=Nadam(lr=0.000)
fold=0
te_pred=np.zeros(X_train.shape[0])
test_pred=np.zeros(X_test.shape[0])
skf = StratifiedKFold(y,n_folds=5, shuffle=True, random_state=1024)
for ind_tr, ind_te in skf:
    #break
    X_tr_raw=X_train[ind_tr]
    y_tr=y[ind_tr]
    X_te_raw=X_train[ind_te]
    y_te=y[ind_te]
    print X_tr_raw.shape,X_te_raw.shape,y_tr.mean(),y_te.mean()
    model=MLP('nadam')
    model.fit(X_tr_raw ,y_tr,   
                     validation_data=(X_te_raw ,y_te),      
                     nb_epoch=3, batch_size=32, shuffle=True) 
    #te_pred[ind_te]=model.predict(X_te_raw)
    test_pred+=model.predict(X_test,batch_size=4096).reshape(-1,)
    print('end fold:{}'.format(fold))    
    fold+=1

test_pred/=5
           
#model=MLP('adam')
#model.fit(X_raw_train,y,nb_epoch=6,batch_size=128,shuffle=True)



#ans=model.predict(X_raw_test)
#pd.to_pickle(ans,'stack/mf_3/nn_model2.test')
res=pd.DataFrame()
res['test_id']=range(len(test_pred))
res['is_duplicate']=test_pred


res.to_csv('res/mlp_6_7_meta_v2.csv',index=False)



def adj(x,te=0.173,tr=0.369):
    a=te/tr
    b=(1-te)/(1-tr)
    return a*x/(a*x+b*(1-x))



res.is_duplicate=res.is_duplicate.apply(adj)
res.to_csv('res/mlp_6_7_meta_v2_adj.csv',index=False)



