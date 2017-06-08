'''
Example of an LSTM model with GloVe embeddings along with magic features

Tested under Keras 2.0 with Tensorflow 1.0 backend

Single model may achieve LB scores at around 0.18+, average ensembles can get 0.17+
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from string import punctuation
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D


from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

########################################
## set directories and parameters
########################################
BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'glove.840B.300d.txt'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

#num_lstm = np.random.randint(175, 275)
#num_dense = np.random.randint(100, 150)
#rate_drop_lstm = 0.15 + np.random.rand() * 0.25
#rate_drop_dense = 0.15 + np.random.rand() * 0.25
num_lstm = 200
num_dense = 128
rate_drop_lstm = 0.2
rate_drop_dense = 0.2

act = 'relu'
re_weight = False # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

########################################
## index word vectors
########################################
print('Indexing word vectors')

#embeddings_index = {}
#f = open(EMBEDDING_FILE)
#count = 0
#for line in f:
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()

#print('Found %d word vectors of glove.' % len(embeddings_index))

########################################
## process texts in datasets
########################################
#print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

#texts_1 = [] 
#texts_2 = []
#labels = []
#with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
#    reader = csv.reader(f, delimiter=',')
#    header = next(reader)
#    for values in reader:
#        texts_1.append(text_to_wordlist(values[3]))
#        texts_2.append(text_to_wordlist(values[4]))
#        labels.append(int(values[5]))
#print('Found %s texts in train.csv' % len(texts_1))

#test_texts_1 = []
#test_texts_2 = []
#test_ids = []
#with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
#    reader = csv.reader(f, delimiter=',')
#    header = next(reader)
#    for values in reader:
#        test_texts_1.append(text_to_wordlist(values[1]))
#        test_texts_2.append(text_to_wordlist(values[2]))
#        test_ids.append(values[0])
#print('Found %s texts in test.csv' % len(test_texts_1))

#tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

#sequences_1 = tokenizer.texts_to_sequences(texts_1)
#sequences_2 = tokenizer.texts_to_sequences(texts_2)
#test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
#test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

#word_index = tokenizer.word_index
#print('Found %s unique tokens' % len(word_index))

#data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
#data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
#labels = np.array(labels)
#print('Shape of data tensor:', data_1.shape)
#print('Shape of label tensor:', labels.shape)

#test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
#test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
#test_ids = np.array(test_ids)

## prepare embeddings
########################################
#print('Preparing embedding matrix')

#nb_words = min(MAX_NB_WORDS, len(word_index))+1

#embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
#for word, i in word_index.items():
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        embedding_matrix[i] = embedding_vector
#print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#pd.to_pickle(embedding_matrix,'data/embedding_matrix.pkl')
#pd.to_pickle(data_1,'data/train_nn_q1.pkl')
#pd.to_pickle(data_2,'data/train_nn_q2.pkl')
#pd.to_pickle(test_data_1,'data/test_nn_q1.pkl')
#pd.to_pickle(test_data_2,'data/test_nn_q2.pkl')
#pd.to_pickle(labels,'data/train_nn_labels.pkl')

########################################
## sample train/validation data
########################################
#np.random.seed(1234)
embedding_matrix=pd.read_pickle('data/embedding_matrix.pkl')
data_1=pd.read_pickle('data/train_nn_q1.pkl')
data_2=pd.read_pickle('data/train_nn_q2.pkl')
test_data_1=pd.read_pickle('data/test_nn_q1.pkl')
test_data_2=pd.read_pickle('data/test_nn_q2.pkl')
labels=pd.read_pickle('data/train_nn_labels.pkl')
nb_words=embedding_matrix.shape[0]

y=pd.read_csv('data/train.csv')['is_duplicate'].values


te_pred=np.zeros(data_1.shape[0])
cnt=0
skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=1024)
for idx_train, idx_val in skf:


    data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
    data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
    ##leaks_train = np.vstack((leaks[idx_train], leaks[idx_train]))
    labels_train = np.concatenate((labels[idx_train], labels[idx_train]))
    data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
    data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
    ##leaks_val = np.vstack((leaks[idx_val], leaks[idx_val]))
    labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels_val==0] = 1.309028344

########################################
## define the model structure
########################################

    NUM_FILTER=64
    NGRAM_FILTERS=[1,3,5]
    embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedded_sequences_1 = embedding_layer(sequence_1_input)
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    convolutions=[]
    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        cur_conv_title = Conv1D(
                                 name="conv_1" + str(n_gram) + '_' + str(i),
                                 filters=NUM_FILTER,
                                 kernel_size=n_gram,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)(embedded_sequences_1)
        one_max = GlobalMaxPooling1D()(cur_conv_title)
        one_avg = GlobalAveragePooling1D()(cur_conv_title)
        convolutions.append(one_max)
        convolutions.append(one_avg)

    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        cur_conv_des = Conv1D(
                                 name="conv_2" + str(n_gram) + '_' + str(i),
                                 filters=NUM_FILTER,
                                 kernel_size=n_gram,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)(embedded_sequences_2)
        # pool


        one_max = GlobalMaxPooling1D()(cur_conv_title)
        one_avg = GlobalAveragePooling1D()(cur_conv_title)
        convolutions.append(one_max)
        convolutions.append(one_avg)

    merged = concatenate(convolutions, name="sentence_vector")
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

########################################
## train the model
########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
    model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
    ##break            ###############################
#model.summary()
#print(STAMP)

#early_stopping =EarlyStopping(monitor='val_loss', patience=3)
#bst_model_path = STAMP + '.h5'
#model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val), \
        epochs=100, batch_size=2048, shuffle=True)

#model.load_weights(bst_model_path)
#bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
    print('Start making the submission before fine-tuning')

    preds = model.predict([data_1[idx_val], data_2[idx_val]], batch_size=8192, verbose=1)
    preds += model.predict([data_2[idx_val], data_1[idx_val]], batch_size=8192, verbose=1)
    preds /= 2
    te_pred[idx_val]=preds
    print('end fold:{}'.format(cnt))
    cnt+=1

#pd.to_pickle(te_pred,'stack/cnn_model1.train')

print 'test'
#bst_model_path = STAMP + '.h5'
#model.load_weights(bst_model_path)
model.fit([np.vstack([data_1,data_2]),np.vstack([data_2,data_1])],np.concatenate([labels,labels]),epochs=16,batch_size=4096,shuffle=True)
preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
preds /= 2

pd.to_pickle(preds,'stack/cnn_model1.test')
