# -*- coding:utf-8 -*-
__author__ = 'bowenwu'

import numpy as np

chars = [chr(i) for i in xrange(ord('A') - 32, ord('A'))] + [chr(i) for i in xrange(ord('a'), ord('z') + 5)]
chars_to_idx = {}
for i, c in enumerate(chars):
    chars_to_idx[c] = i + 1
chars_to_idx['<unk>'] = len(chars)
char_default_idx = len(chars)


def pad_sequences_char(seq, maxlen, word_maxlen, dtype):
    new_seq = np.zeros((len(seq), maxlen, word_maxlen), dtype=dtype)
    for i, sen in enumerate(seq):
        words = sen.split(' ')[:maxlen]
        for j, word in enumerate(words):
            for k, c in enumerate(word):
                if k == word_maxlen:
                    break
                idx = chars_to_idx.get(c, char_default_idx)
                new_seq[i][j][k] = idx
    return new_seq
