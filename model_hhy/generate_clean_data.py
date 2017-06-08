import pandas as pd
import numpy as np
import nltk
import scipy.stats as sps
from .utils import ngram_utils,split_data,nlp_utils,dist_utils
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import string
import re
seed = 1024
np.random.seed(seed)

path = '../data/'

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
test['is_duplicated']=[-1]*test.shape[0]
y_train = train['is_duplicate']
feats= ['question1','question2']
train_value = train[feats].values

data_all = pd.concat([train,test])[feats].values


abbr_dict = {
    "what's": "what is",
    "what're": "what are",
    "who's": "who is",
    "who're": "who are",
    "where's": "where is",
    "where're": "where are",
    "when's": "when is",
    "when're": "when are",
    "how's": "how is",
    "how're": "how are",
    "why's": "why is",
    "why're": "why are",

    "i'm": "i am",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",

    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "who've": "who have",
    "would've": "would have",
    "not've": "not have",

    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'll": "they will",

    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "can't": "can not",
    "couldn't": "could not",
    "don't": "do not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "mustn't": "must not",

    "e-mail": "email",
    "imrovement": 'improvement',
    "intial": "initial",
    "motorolla": "motorola",
    "programing": "programming",
    "quikly": "quickly",
    "demonitization": "demonetization",
    "60k": "60000",
    " 9 11 ":"911",
    " u s ":'american',
    "b g":"bg",
    "e g":'eg',
    ####add.....
    # r'[^\x00-\xff]+':'NOENGLISH'
}


def preprocessing(question):
    #print question
    question=str(question).lower()
    for item in abbr_dict.items():
        question=question.replace(item[0],item[1])
    #print question
    question=question.translate(string.punctuation)
    #question=re.sub(r'[^\x00-\xff]+','non-english',question.decode('utf-8'))
    return question

def text_to_wordlist(text):
    text = text.lower().split()
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
    # Return a list of words
    return (text)


clean_q1 = []
clean_q2 = []
for i in tqdm(np.arange(data_all.shape[0])):
    clean_q1.append(preprocessing(data_all[i][0]))
    clean_q2.append(preprocessing(data_all[i][1]))

for i in tqdm(np.arange(data_all.shape[0])):
    clean_q1[i] = text_to_wordlist(clean_q1[i])
    clean_q2[i] = text_to_wordlist(clean_q2[i])

train_clean = pd.DataFrame()
test_clean = pd.DataFrame()
train_clean['question1'] = clean_q1[:train.shape[0]]
train_clean['question2'] = clean_q2[:train.shape[0]]

test_clean['question1'] = clean_q1[train.shape[0]:]
test_clean['question2'] = clean_q2[train.shape[0]:]

pd.to_pickle(train_clean,'../X_v2/train_final_clean.pkl')
pd.to_pickle(test_clean,'../X_v2/test_final_clean.pkl')

