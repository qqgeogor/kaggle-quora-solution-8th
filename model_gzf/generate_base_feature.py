# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 03:51:55 2017

@author: t-zhga
"""
import pandas as pd
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

import warnings
warnings.filterwarnings(action='ignore')

path = "../input/"

train = pd.read_csv(path+"train_porter.csv")
test = pd.read_csv(path+"test_porter.csv")
test['is_duplicated']=[-1]*test.shape[0]

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stop_words:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stop_words:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))
    return R


def simple_feature(data):
    res=pd.DataFrame()
    print('begin simple features...')
    res['len_q1'] = data.question1.apply(lambda x: len(str(x)))
    res['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    res['diff_len'] = res.len_q1 - res.len_q2
    print('begin simple feature 2...')
    res['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    res['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    res['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    res['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    print('begin simple feature 3...')
    res['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    res['R']=data.apply(word_match_share, axis=1, raw=True)
    
    print('end simple features...')
    
    return res
    
if  __name__ == '__main__':
    feature_train=simple_feature(train)
    feature_test=simple_feature(test)
    feature_train.to_pickle(path+'simple_feature_train.pkl')
    feature_test.to_pickle(path+'simple_feature_test.pkl')
    feature_train.to_csv(path+'simple_feature_train.csv')
    feature_test.to_csv(path+'simple_feature_test.csv')
    
