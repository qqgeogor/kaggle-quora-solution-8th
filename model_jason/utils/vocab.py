# -*- coding:utf-8 -*-
__author__ = 'bowenwu'

from collections import Counter
import numpy as np


"""
Vocab:
    word2index:
    index2word:
Each Language Model have a vocabulary, Vocab is used to manage the vocabulary.
"""


class Vocab:

    __slots__ = ["word2index", "index2word", "unknown"]

    def __init__(self, index2word=None):
        """init the vocab

        By default, **unknown** is 1, <eos> is 2. 0 is left for padding.
        Note about the denotation of unknown and eos.

        Args:
            index2word: add the existing mapping to it.
        """
        self.word2index = {}
        self.index2word = []

        # add unknown word:
        self.add_words(["*PAD*", "*UNK*"])
        self.unknown = self.word2index["*UNK*"]
        self.mask = self.word2index["*PAD*"]

        if index2word is not None:
            self.add_words(index2word)

    @property
    def size(self):
        return len(self.word2index)

    def __len__(self):
        return len(self.word2index)

    def __call__(self, line):
        """
        (1) Convert from numerical representation to words and vice-versa.
        (2) Convert words list to index list
        """
        assert type(line) in [str, unicode, list, np.ndarray]
        if type(line) in [str, unicode]:
            line = line.strip().split(" ")
        return [self.word2index.get(word, self.unknown) for word in line]

    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                if isinstance(word, str):
                    word = word.decode("utf8")
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)

    # thread: only append words with certan frequency.
    def add_words_with_thes(self, words_counts, thes=1):
        for word, count in words_counts.items():
            count = int(count)
            if count < thes:
                continue

            if word not in self.word2index:
                if isinstance(word, str):
                    word = word.decode("utf8")
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)
        # end vocab building

    def save_vocab(self, vocab_file):
        fvocab = open(vocab_file, 'w')
        for windx, word in enumerate(self.index2word):
            fvocab.write('%d\t%s\n' % (windx, word.encode("utf8")))
            fvocab.flush()
        fvocab.close()

    def load_vocab(self, vocab_file):
        self.word2index = {}
        self.index2word = []

        fin = open(vocab_file)
        for line in fin:
            items = line.strip().decode("utf8").split('\t')
            self.word2index[items[1]] = int(items[0])
        fin.close()
        word_num = len(self.word2index)
        self.index2word = [''] * word_num
        for word, idx in self.word2index.items():
            self.index2word[idx] = word
# end Vocab


def buildVocab(sentences, thes=None):
    """
    input: sentences, list of sentences separated by space
    return: Vocab
    """
    stat_num = 10000
    words = []
    wcounter = Counter([])
    line_no = 0
    max_len = 0
    for line in sentences:
        t_words = line.strip().split()
        max_len = max(max_len, len(t_words))
        words.extend(t_words)
        line_no += 1
        if line_no % stat_num == 1:
            wcounter.update(Counter(words))
            words = []  # generate a processing log
    wcounter.update(Counter(words))

    vocab = Vocab()  # create vocab
    if thes:
        vocab.add_words_with_thes(dict(wcounter), thes=thes)
    else:
        vocab.add_words(list(wcounter))
    return vocab, max_len, len(wcounter)
