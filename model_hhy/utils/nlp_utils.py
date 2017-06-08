import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from .dist_utils import _calc_similarity

def _get_embedd_Index(glove_dir):
    embedding_Index = {}
    with codecs.open(os.path.join(glove_dir), encoding='utf-8') as f:
        for line in f:#按照行存储的
            values = line.split()
            try:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                continue
            embedding_Index[word] = coefs
        f.close()
    return embedding_Index


def _tokenize(text, token_pattern=" "):
    # token_pattern = r"(?u)\b\w\w+\b"
    # token_pattern = r"\w{1,}"
    # token_pattern = r"\w+"
    # token_pattern = r"[\w']+"
    if token_pattern == " ":
        # just split the text into tokens
        return text.split(" ")
    else:
        token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
        group = token_pattern.findall(text)
        return group

# after spacy not become !


class Embedd_generator:
    def __init__(self,glove_dir=''):
        self.dir = glove_dir
        self.embedd_Index = _get_embedd_Index(glove_dir)

    def _warpper_token_embedd_cos(self,q1,q2):
        embedd_Index = self.embedd_Index
        if (len(q1)==0) & (len(q2)==0):
            return 1
        if (len(q1)==0) | ((len(q2)==0)):
            return 0
        centroid_q1 = np.zeros(100)
        k = 0
        for w in q1:
            w_l = str(w).lower().split()
            for _w in w_l:
                if _w in embedd_Index:
                    centroid_q1 += embedd_Index[_w]
                    k += 1
        if k==0: return 0
        centroid_q1 /= float(k)

        centroid_q2 = np.zeros(100)
        k = 0
        for w in q2:
            w_l = str(w).lower().split()
            for _w in w_l:
                if _w in embedd_Index:
                    centroid_q2 += embedd_Index[_w]
                    k += 1
        if k==0: return 0
        centroid_q2 /= float(k)
        return _calc_similarity(centroid_q1, centroid_q2)

    def _wrapper_sent_embedd_cos(self,q1,q2):
        embedd_Index = self.embedd_Index
        if (len(q1)==0) & (len(q2)==0):
            return 1
        if (len(q1)==0) | ((len(q2)==0)):
            return 0
        centroid_q1 = np.zeros(100)
        k = 0
        for l1 in q1:
            for w in l1:
                if w[0] == '!':
                    centroid_q1+=embedd_Index['not']
                    k+=1
                    w = w[1:]
                if w in embedd_Index:
                    centroid_q1 += embedd_Index[w]
                    k+=1
        if k==0:return 0
        centroid_q1 /= float(k)

        centroid_q2 = np.zeros(100)
        k = 0
        for l in q2:
            for w in l:
                if w[0]=='!':
                    centroid_q2+=embedd_Index['not']
                    k+=1
                    w = w[1:]
                if w in embedd_Index:
                    centroid_q2 += embedd_Index[w]
                    k+=1
        if k==0:return 0
        centroid_q2 /= float(k)

        return _calc_similarity(centroid_q1,centroid_q2)

    def _wrapper_sent_subject_cos(self,q1,q2):
        embedd_Index = self.embedd_Index
        if (len(q1)==0) & (len(q2)==0):
            return 1
        if (len(q1)==0) | ((len(q2)==0)):
            return 0
        centroid_q1 = np.zeros(100)
        k = 0
        for l1 in q1:
            w = l1[0]
            if w in embedd_Index:
                centroid_q1 += embedd_Index[w]
                k+=1
        if k==0:return 0
        centroid_q1 /= float(k)

        centroid_q2 = np.zeros(100)
        k = 0
        for l in q2:
            w = l[0]
            if w in embedd_Index:
                centroid_q2 += embedd_Index[w]
                k+=1
        if k==0:return 0
        centroid_q2 /= float(k)
        return _calc_similarity(centroid_q1,centroid_q2)

    def _wrapper_sent_verb_cos(self,q1,q2):
        embedd_Index = self.embedd_Index
        if (len(q1)==0) & (len(q2)==0):
            return 1
        if (len(q1)==0) | ((len(q2)==0)):
            return 0
        centroid_q1 = np.zeros(100)
        k = 0
        for l1 in q1:
            w = l1[1]
            if w[0]=='!':
                k+=1
                centroid_q1+=embedd_Index['not']
                w = w[1:]
            if w in embedd_Index:
                centroid_q1 += embedd_Index[w]
                k+=1
        if k==0:return 0
        centroid_q1 /= float(k)
        centroid_q2 = np.zeros(100)
        k = 0
        for l in q2:
            w = l[1]
            if w[0]=='!':
                k+=1
                centroid_q1+=embedd_Index['not']
                w = w[1:]
            if w in embedd_Index:
                centroid_q2 += embedd_Index[w]
                k+=1
        if k==0:return 0
        centroid_q2 /= float(k)
        return _calc_similarity(centroid_q1,centroid_q2)

    def _wrapper_sent_object_cos(self,q1,q2):
        embedd_Index = self.embedd_Index
        if (len(q1)==0) & (len(q2)==0):
            return 1
        if (len(q1)==0) | ((len(q2)==0)):
            return 0
        centroid_q1 = np.zeros(100)
        k = 0
        for l1 in q1:
            w = l1[2]
            if w in embedd_Index:
                centroid_q1 += embedd_Index[w]
                k+=1
        if k==0:return 0
        centroid_q1 /= float(k)
        centroid_q2 = np.zeros(100)
        k = 0
        for l in q2:
            w = l[2]
            if w in embedd_Index:
                centroid_q2 += embedd_Index[w]
                k+=1
        if k==0:return 0
        centroid_q2 /= float(k)
        return _calc_similarity(centroid_q1,centroid_q2)
