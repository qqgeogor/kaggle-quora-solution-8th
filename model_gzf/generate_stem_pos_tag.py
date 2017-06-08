import re
import sys
import nltk
import string
import pandas as pd
from bs4 import BeautifulSoup
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings(action='ignore')

path='../input/'
train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')
stopwords=nltk.corpus.stopwords.words('english')
english_stemmer = nltk.stem.SnowballStemmer('english')
#english_stemmer_2=nltk.stem.PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

token_pattern = r"(?u)\b\w\w+\b"

def pos_tag_text(line,
                 token_pattern=token_pattern,
                 exclude_stopword=stopwords,
                 encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    for name in ["question1", "question2"]:
        l = line[name]
        ## tokenize
        tokens = [x.lower() for x in token_pattern.findall(l)]
        ## stem
        #tokens=l.lower().split()
        #print tokens
        tokens = stem_tokens(tokens, english_stemmer)
        line[name+'_stem']=' '.join(tokens)
        #print tokens
        if exclude_stopword:
            tokens = [x for x in tokens if x not in stopwords]
        tags = pos_tag(tokens)
        tags_list = [t for w,t in tags]
        tags_str = " ".join(tags_list)
        #print tags_str
        line[name+'_pos_tag'] = tags_str
    return line[[ u'question1_stem', u'question1_pos_tag', u'question2_stem',
       u'question2_pos_tag']]

train=train.apply(pos_tag_text,axis=1)
test=test.apply(pos_tag_text,axis=1)

train.to_csv(path+'train_stem_pos_tag.csv')
test.to_csv(path+'test_stem_pos_tag.csv')


