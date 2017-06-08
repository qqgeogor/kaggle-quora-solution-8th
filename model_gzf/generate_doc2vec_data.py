import pandas as pd
import numpy as np
import os
import string
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

train=pd.read_csv('data/train.csv')[['question1','question2']]
test=pd.read_csv('data/test.csv')[['question1','question2']]

abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",
    "why's":"why is",
    "why're":"why are",
    
    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    "mustn't":"must not",
    
    "e-mail":"email",
    "imrovement":'improvement',
    "intial":"initial",
    "motorolla":"motorola",
    "programing":"programming",
    "quikly":"quickly",
    "demonitization":"demonetization"
    ####add.....
    #r'[^\x00-\xff]+':'NOENGLISH' 
}


def preprocessing(question):
    #print question
    question=str(question).lower()
    for item in abbr_dict.items():
        question=question.replace(item[0],item[1])
    #print question
    question=question.translate(None,string.punctuation)
    #question=re.sub(r'[^\x00-\xff]+','non-english',question.decode('utf-8'))
    return question


data=[]
for fe in ['question1','question2']:
    data.extend(train[fe].astype(str).tolist())
    data.extend(test[fe].astype(str).tolist())

f = open('data/all_sentences.txt','w')
for i,line in enumerate(data):
    line=preprocessing(line)
    f.write('_*'+str(i)+' '+line+'\n')
f.close()


