import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,LSTM,Embedding,Dropout,Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import merge


#embedd path
home = os.path.expanduser('~')
glove_dir = os.path.join(home, 'data', 'glove')
glove_corpus = '42B'
sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
#glove 参数
args = {'glove_dir':os.path.join(home, 'data', 'glove'),'glove_corpus':glove_corpus,'glove_vec_size':300}

BASE_DIR = '.'
EMBEDDING_FILE = os.path.join(args['glove_dir'],"glove.{}.{}d.txt".format(args['glove_corpus'], args['glove_vec_size']))
TRAIN_DATA_FILE = BASE_DIR + '/data/train.csv'
TEST_DATA_FILE = BASE_DIR + '/data/test.csv'

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = args['glove_vec_size']
VALIDATION_SPLIT = 0.0

num_lstm = 100#np.random.randint(175,275)
num_dense = 100#np.random.randint(100,150)
rate_drop_lstm = 0.15 #+ np.random.rand() * 0.25
rate_drop_dense = 0.15 #+ np.random.rand() * 0.25


act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

#index word vectors
embedding_Index = {}
with codecs.open(os.path.join(args['glove_dir'],
                              "glove.{}.{}d.txt".format(args['glove_corpus'],
                              args['glove_vec_size'])), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        embedding_Index[word] = coefs
    f.close()
print('Found %s word vectors.'%len(embedding_Index))


########################################
## process texts in datasets
########################################

print('Processing text dataset')
def text_to_wordlist(text,remove_stopwords=False,stem_words=False):

    text = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        text = [w for w in text if not w in stops]
    text = " ".join(text)#to str
    #clean the text
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
    #punction replace
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)#change to  3 words
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    #text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
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
    return (text)


train_data = pd.read_csv(TRAIN_DATA_FILE)
test_data = pd.read_csv(TEST_DATA_FILE)

#clean
print('clean text dataset')
train_data['clean_question1'] = train_data['question1'].astype('str').apply(lambda x:text_to_wordlist(x))
train_data['clean_question2'] = train_data['question2'].astype('str').apply(lambda x:text_to_wordlist(x))
test_data['clean_question1'] = test_data['question1'].astype('str').apply(lambda x:text_to_wordlist(x))
test_data['clean_question2'] = test_data['question2'].astype('str').apply(lambda x:text_to_wordlist(x))

# pd.to_pickle(train_data,'./data/train_clean.pkl')
# pd.to_pickle(test_data,'./data/test_clean.pkl')
train_data = pd.read_pickle('./data/train_clean.pkl')
test_data = pd.read_pickle('./data/test_clean.pkl')

pd.DataFrame().drop_duplicates
print('tokenizer and build vocab')
corpus = []
labels = []
test_ids = []
#save for latter embedd
train_text_1 = []
train_text_2 = []
test_text_1 = []
test_text_2 = []
train_text_1 += train_data['clean_question1'].values.tolist()
train_text_2 += train_data['clean_question2'].values.tolist()
labels = train_data['is_duplicate'].values.tolist()
test_text_1+=test_data['clean_question1'].values.tolist()
test_text_2+=test_data['clean_question2'].values.tolist()
test_ids = test_data['test_id'].values.tolist()

corpus+=(train_text_1+train_text_2+test_text_1+test_text_2)
print('corpus size ',len(corpus))

#create vocab and split
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(corpus)

#split the sequence's token
train_sequences_1 = tokenizer.texts_to_sequences(train_text_1)#split sentence and replace word index
train_sequences_2 = tokenizer.texts_to_sequences(train_text_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_text_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_text_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

#save sequence by token id
#save vocab index
pd.to_pickle(train_sequences_1,'train_data/train_q1_tokenizer_id')
pd.to_pickle(train_sequences_2,'train_data/train_q2_tokenizer_id')
pd.to_pickle(test_sequences_1,'test_data/test_q1_tokenizer_id')
pd.to_pickle(test_sequences_2,'test_data/test_q2_tokenizer_id')


#padding used for train
print('padding samples to the same length')
data_1 = pad_sequences(train_sequences_1,maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(train_sequences_2,maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)


########################################
## prepare embeddings
########################################
#key point null word
print('generate embedding matrix')
nb_words = min(MAX_NB_WORDS,len(word_index))+1

embedding_matrix = np.zeros((nb_words,EMBEDDING_DIM))

for word,i in word_index.items():
    embedding_vector = embedding_Index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
#Null word embeddings: 6B 40000 20414  42B

pd.to_pickle(embedding_matrix,'add_feature/word_id_embedd_matrix.pkl')

########################################
## sample train/validation data
########################################
np.random.seed(1024)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
#idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]


data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))


########################################
## define the model graph
########################################
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)#first not change weights
lstm_layer = LSTM(num_lstm, dropout_W=rate_drop_lstm,dropout_U=rate_drop_lstm)#memory dropout

#create graph from input to embedd-layer and  lstm-layer
sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32')
embedded_sequence_1 = embedding_layer(sequence_1_input)
x1_forward = LSTM(num_lstm,dropout_W=rate_drop_lstm,dropout_U=rate_drop_lstm)(embedded_sequence_1)
x1_back = LSTM(num_lstm,dropout_W=rate_drop_lstm,dropout_U=rate_drop_lstm,go_backwards=True)(embedded_sequence_1)
x1 = merge([x1_forward,x1_back],mode='concat')

#x1 = lstm_layer(embedded_sequence_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)

#bi
y1_forward = LSTM(num_lstm,dropout_W=rate_drop_lstm,dropout_U=rate_drop_lstm)(embedded_sequences_2)
y1_back = LSTM(num_lstm,dropout_W=rate_drop_lstm,dropout_U=rate_drop_lstm,go_backwards=True)(embedded_sequences_2)
y1 = merge([y1_forward,y1_back],mode='concat')

#concate
embedd_out = merge([x1,y1], mode='concat')

merged = Dropout(rate_drop_dense)(embedd_out)
merged = BatchNormalization()(merged)

merged_phrase = Dense(num_dense,activation=act)(merged)#关键特征


merged = Dropout(rate_drop_dense)(merged_phrase)
merged = BatchNormalization()(merged)


preds = Dense(1,activation='sigmoid')(merged)

#add class weight

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
## median LSTM feature
########################################
model = Model([sequence_1_input,sequence_2_input],preds)
q1_embedd = Model([sequence_1_input,sequence_2_input],x1)
q2_embedd = Model([sequence_1_input,sequence_2_input],y1)
q1_q2_all_embedd = Model([sequence_1_input,sequence_2_input],merged_phrase)

model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])

#model.summary()
print(STAMP)

print('begining Training-----------')
early_stopping = EarlyStopping(monitor='loss',patience=3)#3次　val没有下降就退出
bst_model_path = STAMP+'100_40_60'+'.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=False, save_weights_only=True)

hist = model.fit([data_1_train,data_2_train],labels_train,
validation_data=([data_1_train[0:100],data_2_train[0:100]],labels_train[0:100]),
                 nb_epoch=100,batch_size=2048,shuffle=True,
                 class_weight=class_weight,callbacks=[early_stopping,model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])
########################################
## get the lstm embedd
########################################

#q1-phrase and q2 embedd for train
q1_q2_train = q1_q2_all_embedd.predict([data_1,data_2],batch_size=2048,verbose=1)
q1_q2_test = q1_q2_all_embedd.predict([test_data_1,test_data_2],batch_size=2048,verbose=1)
pd.to_pickle(q1_q2_train,'add_feature/lstm_phrase_train.pkl')
pd.to_pickle(q1_q2_test,'add_feature/lstm_phrase_test.pkl')

q1_lstm_train = q1_embedd.predict([data_1,data_2],batch_size=2048,verbose=1)
q2_lstm_train = q2_embedd.predict([data_1,data_2],batch_size=2048,verbose=1)

q1_lstm_test = q1_embedd.predict([test_data_1,test_data_2],batch_size=1024,verbose=1)
q2_lstm_test = q2_embedd.predict([test_data_1,test_data_2],batch_size=1024,verbose=1)

pd.to_pickle(q1_lstm_train,'train_data/train_lstm_embedd_q1.pkl')
pd.to_pickle(q2_lstm_train,'train_data/train_lstm_embedd_q2.pkl')
pd.to_pickle(q1_lstm_test,'test_data/test_lstm_embedd_q1.pkl')
pd.to_pickle(q2_lstm_test,'test_data/test_lstm_embedd_q2.pkl')
