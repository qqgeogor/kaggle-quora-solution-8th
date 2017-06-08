import logging
import numpy as np
from itertools import islice
import os
import re
import csv
import codecs
from  tqdm import tqdm
from collections import defaultdict
from  collections import Counter
import pandas as pd
import numpy as np

seed = 1024

np.random.seed(seed)

path = '../data/'


class DataManager():

    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0
        self._padding_token = "@@PADDING@@"
        self._oov_token = "@@UNKOWN@@"
        self.char_index = {}
        self.index_char = {}
        self.char_vocab_size = 0
        self.mode='word'


    def set_vocab_mode(self,mode):
        self.mode = mode

    def get_vocab_size(self):
        return self.vocab_size


    def get_char_vocab_size(self):
        return self.char_vocab_size


    def fit_vocab(self,train_filename,test_filename,min_count=1,load=False):
        # train_dataset = pd.read_pickle(train_filename)[['question1', 'question2']]
        # test_dataset = pd.read_pickle(test_filename)[['question1','question2']]
        train_dataset = pd.read_pickle('../data/train_final_clean.pkl')[['question1', 'question2']]
        test_dataset = pd.read_pickle('../data/test_final_clean.pkl')[['question1', 'question2']]
        train_test = pd.concat([train_dataset,test_dataset]).values
        self.fit_word_dictionary(train_test, min_count=min_count)
        self.save_dict()


    def get_train_data_from_file(self, filenames,
                                 max_lengths=None, pad=True,VALIDATION_SPLIT=0.2,max_word_chars=5):
        if self.mode=='word':
            train = pd.read_pickle(filenames)
            train_dataset = train[['question1','question2']].values
            labels = train['is_duplicate'].values
            #labels = pd.read_csv('../data/train.csv')['is_duplicate'].values
            inp_q1,inp_q2 = self.get_index_input(train_dataset)
            if pad:
                inp_q1 = self.pad_sequences(inp_q1,max_lengths)
                inp_q2 = self.pad_sequences(inp_q2,max_lengths)

            inp_q1 = inp_q1.tolist()
            inp_q2 = inp_q2.tolist()
            labels = labels.tolist()

            samples = []
            for it in tqdm(np.arange(train_dataset.shape[0])):
                tmp = []
                tmp.append(inp_q1[it])
                tmp.append(inp_q2[it])
                tmp.append(labels[it])
                samples.append(tmp)

            perm = np.random.permutation(len(samples))
            idx_train = perm[:int(len(samples) * (1 - VALIDATION_SPLIT))]
            idx_val = perm[int(len(samples)*(1-VALIDATION_SPLIT)):]

            samples_train = np.array(samples)[idx_train]
            samples_val = np.array(samples)[idx_val]
            return samples_train.tolist(),samples_val.tolist()
        else:
            train = pd.read_pickle(filenames)
            train_dataset = train[['question1', 'question2']].values
            labels = train['is_duplicate'].values
            # labels = pd.read_csv('../data/train.csv')['is_duplicate'].values
            inp_q1, inp_q2,inp_c_q1,inp_c_q2 = self.get_index_input(train_dataset)
            if pad:
                inp_q1 = self.pad_sequences(inp_q1, max_lengths)
                inp_q2 = self.pad_sequences(inp_q2, max_lengths)
            for wl in tqdm(np.arange(len(inp_c_q1))):
                inp_c_q1[wl] = self.pad_sequences(inp_c_q1[wl],max_word_chars).tolist()
            for wl in tqdm(np.arange(len(inp_c_q1))):
                inp_c_q2[wl] = self.pad_sequences(inp_c_q2[wl],max_word_chars).tolist()
            inp_c_q1 = self.pad_sequences(inp_c_q1,max_lengths)
            inp_c_q2 = self.pad_sequences(inp_c_q2,max_lengths)

            inp_q1 = inp_q1.tolist()
            inp_q2 = inp_q2.tolist()
            inp_c_q1 = inp_c_q1.tolist()
            inp_c_q2 = inp_c_q2.tolist()
            labels = labels.tolist()

            samples = []
            for it in tqdm(np.arange(train_dataset.shape[0])):
                tmp = []
                tmp.append(inp_q1[it])
                tmp.append(inp_q2[it])
                tmp.append(inp_c_q1[it])
                tmp.append(inp_c_q2[it])
                tmp.append(labels[it])
                samples.append(tmp)

            perm = np.random.permutation(len(samples))
            idx_train = perm[:int(len(samples) * (1 - VALIDATION_SPLIT))]
            idx_val = perm[int(len(samples) * (1 - VALIDATION_SPLIT)):]

            samples_train = np.array(samples)[idx_train]
            samples_val = np.array(samples)[idx_val]
            return samples_train.tolist(), samples_val.tolist()

    def get_test_data_from_file(self, filenames,
                                 max_lengths=None, pad=True,max_word_chars=5):
        if self.mode=='word':
            test_dataset = pd.read_pickle(filenames)[['question1', 'question2']].values

            inp_q1, inp_q2 = self.get_index_input(test_dataset)
            if pad:
                inp_q1 = self.pad_sequences(inp_q1, max_lengths)
                inp_q2 = self.pad_sequences(inp_q2, max_lengths)

            inp_q1 = inp_q1.tolist()
            inp_q2 = inp_q2.tolist()

            samples = []
            for it in tqdm(np.arange(test_dataset.shape[0])):
                tmp = []
                tmp.append(inp_q1[it])
                tmp.append(inp_q2[it])
                samples.append(tmp)
        else:
            test_dataset = pd.read_pickle(filenames)[['question1', 'question2']].values
            # labels = pd.read_csv('../data/train.csv')['is_duplicate'].values
            inp_q1, inp_q2,inp_c_q1,inp_c_q2 = self.get_index_input(test_dataset)
            if pad:
                inp_q1 = self.pad_sequences(inp_q1, max_lengths)
                inp_q2 = self.pad_sequences(inp_q2, max_lengths)
            for wl in tqdm(np.arange(len(inp_c_q1))):
                inp_c_q1[wl] = self.pad_sequences(inp_c_q1[wl],max_word_chars).tolist()
            for wl in tqdm(np.arange(len(inp_c_q1))):
                inp_c_q2[wl] = self.pad_sequences(inp_c_q2[wl],max_word_chars).tolist()

            inp_c_q1 = self.pad_sequences(inp_c_q1,max_lengths)
            inp_c_q2 = self.pad_sequences(inp_c_q2,max_lengths)

            inp_q1 = inp_q1.tolist()
            inp_q2 = inp_q2.tolist()
            inp_c_q1 = inp_c_q1.tolist()
            inp_c_q2 = inp_c_q2.tolist()

            samples = []
            for it in tqdm(np.arange(test_dataset.shape[0])):
                tmp = []
                tmp.append(inp_q1[it])
                tmp.append(inp_q2[it])
                tmp.append(inp_c_q1[it])
                tmp.append(inp_c_q2[it])
                samples.append(tmp)

        return samples


    def fit_word_dictionary(self,train_dataset,min_count=0):
        if self.mode=='word':
            _word_counts = defaultdict(Counter)

            for it in tqdm(np.arange(train_dataset.shape[0])):
                q1 = train_dataset[it][0]
                q2 = train_dataset[it][1]
                w_q1 = q1.lower().split()
                w_q2 = q2.lower().split()
                for w in w_q1:
                    if w in _word_counts:
                        _word_counts[w] += 1
                    else:
                        _word_counts[w] = 0
                for w in w_q2:
                    if w in _word_counts:
                        _word_counts[w] += 1
                    else:
                        _word_counts[w] = 0

            sorted_word_counts = sorted(_word_counts.items(),
                                        key=lambda pair: (-pair[1],
                                                          pair[0]))
            # 0(padding token) and 1 (OOV token)

            for word, count in sorted_word_counts:
                if count >=min_count:
                    index = len(self.word_index)+2 #fill 0
                    self.word_index[word] =  index
                    self.index_word[index] = word
            vocab_size = len(self.word_index) + 2
            self.vocab_size = vocab_size
        else:
            _word_counts = defaultdict(Counter)
            _char_counts = defaultdict(Counter)
            for it in tqdm(np.arange(train_dataset.shape[0])):
                q1 = train_dataset[it][0]
                q2 = train_dataset[it][1]
                w_q1 = q1.lower().split()
                w_q2 = q2.lower().split()
                # c_w_q1 = [list(c_w) for c_w in w_q1]
                # c_w_q2 = [list(c_w) for c_w in w_q2]
                for w in w_q1:
                    if w in _word_counts:
                        _word_counts[w] += 1
                        for i,ci in enumerate(list(w)):
                            if ci in _char_counts:
                                _char_counts[ci]+=1
                            else:
                                _char_counts[ci] = 0
                    else:
                        _word_counts[w] = 0
                        for i,ci in enumerate(list(w)):
                            if ci in _char_counts:
                                _char_counts[ci] += 1
                            else:
                                _char_counts[ci] = 0
                for w in w_q2:
                    if w in _word_counts:
                        _word_counts[w] += 1
                        for i,ci in enumerate(list(w)):
                            if ci in _char_counts:
                                _char_counts[ci] += 1
                            else:
                                _char_counts[ci] = 0
                    else:
                        _word_counts[w] = 0
                        for i,ci in enumerate(list(w)):
                            if ci in _char_counts:
                                _char_counts[ci] += 1
                            else:
                                _char_counts[ci] = 0

            sorted_word_counts = sorted(_word_counts.items(),
                                        key=lambda pair: (-pair[1],
                                                          pair[0]))

            sorted_char_counts = sorted(_char_counts.items(),
                                        key=lambda pair: (-pair[1],
                                                          pair[0]))
            # 0(padding token) and 1 (OOV token)

            for word, count in sorted_word_counts:
                if count >= min_count:
                    index = len(self.word_index) + 2  # fill 0
                    self.word_index[word] = index
                    self.index_word[index] = word
            vocab_size = len(self.word_index) + 2
            self.vocab_size = vocab_size

            for ch, count in sorted_char_counts:
                index = len(self.char_index) + 2  # fill 0
                self.char_index[ch] = index
                self.index_char[index] = ch
            char_size = len(self.char_index) + 2
            self.char_vocab_size = char_size

    def save_dict(self,outpath='./data/dictionary/'):
        pd.to_pickle(self.word_index, outpath + 'word_index.pkl')
        pd.to_pickle(self.index_word, outpath + 'index_word.pkl')
        if self.mode!='word':
            pd.to_pickle(self.char_index,outpath+'char_index.pkl')
            pd.to_pickle(self.index_char,outpath+'index_char.pkl')

    def load_word_dictionary(self,word_dict={}):
        self.word_index = pd.read_pickle(word_dict['word_index'])
        self.index_word = pd.read_pickle(word_dict['index_word'])
        self.vocab_size = len(self.word_index)+2
        if self.mode!='word':
            self.char_index = pd.read_pickle(word_dict['char_index'])
            self.index_char = pd.read_pickle(word_dict['index_char'])
            self.char_vocab_size = len(self.char_index)+2

    def get_index_input(self,train_dataset):
        if self.mode=='word':
            ind_q1 = []
            ind_q2 = []
            for it in tqdm(np.arange(train_dataset.shape[0])):
                q1 = train_dataset[it][0]
                q2 = train_dataset[it][1]
                w_q1 = q1.lower().split()
                w_q2 = q2.lower().split()

                id_q1 = [self.word_index[w] for w in w_q1 if w in self.word_index]
                id_q2 = [self.word_index[w] for w in w_q2 if w in self.word_index]
                ind_q1.append(id_q1)
                ind_q2.append(id_q2)

            return ind_q1,ind_q2
        else:
            ind_q1 = []
            ind_q2 = []
            ind_c_q1 = []
            ind_c_q2 = []
            for it in tqdm(np.arange(train_dataset.shape[0])):
                q1 = train_dataset[it][0]
                q2 = train_dataset[it][1]
                w_q1 = q1.lower().split()
                w_q2 = q2.lower().split()

                id_q1 = [self.word_index[w] for w in w_q1 if w in self.word_index]
                id_q2 = [self.word_index[w] for w in w_q2 if w in self.word_index]
                cx_q1 = [[self.char_index[cw] for cw in list(w) if cw in self.char_index] for w in w_q1 if
                         w in self.word_index]
                cx_q2 = [[self.char_index[cw] for cw in list(w) if cw in self.char_index] for w in w_q2 if
                         w in self.word_index]
                ind_q1.append(id_q1)
                ind_q2.append(id_q2)
                ind_c_q1.append(cx_q1)
                ind_c_q2.append(cx_q2)

            return ind_q1, ind_q2,ind_c_q1,ind_c_q2



    def pad_sequences(self,sequences, maxlen=None, dtype='int32',
                      padding='post', truncating='post', value=0.):
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break

        x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == 'pre':  # 是从前截断还是从后面截断
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))

            # 如果少了如何进行Padding
            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x

    def get_next_batch(self,samples,batch_size=128,batch_index=0):
        q1 = []
        q2 = []
        q1_c = []
        q2_c = []
        y = []
        samples_c = samples[batch_size*batch_index:(batch_index+1)*batch_size]

        for line in samples_c:
            q1.append(line[0])
            q2.append(line[1])
            if self.mode!='word':
                q1_c.append(line[2])
                q2_c.append(line[3])
            if line[2]==0:
                y.append([1,0])
            else:
                y.append([0,1])
        if self.mode!='word':
            return q1,q2,q1_c,q2_c,y
        return q1, q2, y


    def get_test_next_batch(self,samples,batch_size=128,batch_index=0):
        q1 = []
        q2 = []
        q1_c = []
        q2_c = []
        samples_c = samples[batch_size*batch_index:(batch_index+1)*batch_size]

        for line in samples_c:
            q1.append(line[0])
            q2.append(line[1])
            if self.mode!='word':
                q1_c.append(line[2])
                q2_c.append(line[3])
        if self.mode!='word':
            return q1,q2,q1_c,q2_c
        return q1, q2



    def get_embedd_matrix(self,pretrained_embeddings_dict=True):
        if pretrained_embeddings_dict:
            vector_size = 100
            glove_dir = '../data/glove.6B.{0}d.txt'.format(vector_size)
            embeddings_from_file = {}
            with codecs.open(os.path.join(glove_dir), encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    try:
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                    except:
                        continue
                    embeddings_from_file[word] = coefs
                f.close()

        vocab_size = self.vocab_size
        char_vocab_size = self.char_vocab_size
        # Build the embedding matrix
        numpy_rng = np.random.RandomState(seed)
        scale = 0.05

        char_embedd_matrix = numpy_rng.uniform(low=-scale,high=scale,size=(char_vocab_size,vector_size))
        embedding_matrix = numpy_rng.uniform(low=-scale, high=scale, size=(vocab_size,vector_size))
        # The 2 here because there is no point in setting vectors
        # for 0 (padding token) and 1 (OOV token)
        for i in range(2, vocab_size):
            # Get the word corresponding to the index
            word = self.index_word[i]
            if embeddings_from_file and word in embeddings_from_file:
                embedding_matrix[i] = embeddings_from_file[word]

        if self.mode!='word':

            for i in range(2, char_vocab_size):
                # Get the char corresponding to the index
                ch = self.index_char[i]
                if embeddings_from_file and ch in embeddings_from_file:
                    char_embedd_matrix[i] = embeddings_from_file[ch]

        if self.mode!='word':
            return embedding_matrix, char_embedd_matrix
        else:
            return embedding_matrix


#have a test for fun
if __name__ == '__main__':

    # train_dataset = pd.read_pickle('../../data/train_final_clean.pkl')[['question1', 'question2']]
    # test_dataset = pd.read_pickle('../../data/test_final_clean.pkl')[['question1', 'question2']]
    # train_test = pd.concat([train_dataset, test_dataset]).values
    # dm.fit_word_dictionary(train_test, min_count=1)
    # dm.set_vocab_mode('word+char')
    # dm.fit_word_dictionary(train_test, min_count=1)
    # dm.save_dict()

    dm = DataManager()
    dm.set_vocab_mode('word+char')
    word_dict_base = '../data/dictionary/'
    word_dict_path = {'word_index':word_dict_base+'word_index.pkl','index_word':word_dict_base+'index_word.pkl',
                      'char_index':word_dict_base+'char_index.pkl','index_char':word_dict_base+'index_char.pkl'}
    dm.load_word_dictionary(word_dict=word_dict_path)

    samples,vals = dm.get_train_data_from_file('../../data/train_final_clean.pkl', max_lengths=30)
    # word_dict_path = {'word_index':word_dict_base+'word_index.pkl','index_word':word_dict_base+'index_word.pkl',
    #                   'char_index':word_dict_base+'char_index.pkl','index_char':word_dict_base+'index_char.pkl'}
    # dm.load_word_dictionary(word_dict=word_dict_path)
    # print(dm.get_vocab_size(),dm.get_char_vocab_size())

    q1,q2,q1_c,q2_c = dm.get_next_batch(samples,batch_size=128,batch_index=0)

    embedd_matrix,embedd_char_matrix = dm.get_embedd_matrix()

    # embedd_matrix = dm.get_embedd_matrix()
    #
    # batch_q1,batch_q2,batch_y = dm.get_next_batch(samples)
