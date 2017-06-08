from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
from feat_utils import get_jaccard
from feat_utils import get_dice
from feat_utils import get_count_q1_in_q2
from feat_utils import get_ratio_q1_in_q2
from feat_utils import get_count_of_question
from feat_utils import get_count_of_unique_question
from feat_utils import get_ratio_of_unique_question
from feat_utils import get_count_of_digit
from feat_utils import get_ratio_of_digit
from config import path
from simhash import Simhash
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize

def tokenize(sequence):
    words = word_tokenize(sequence)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words

def get_word_ngrams(sequence, n=3):
    tokens = tokenize(sequence)
    return [' '.join(ngram) for ngram in ngrams(tokens, n)]

def get_character_ngrams(sequence, n=3):
    sequence = clean_sequence(sequence)
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

def caluclate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))

def get_word_distance( q1, q2):
    q1, q2 = tokenize(q1), tokenize(q2)
    return caluclate_simhash_distance(q1, q2)
def get_word_2gram_distance( q1, q2):
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)
def get_word_2gram_distance( q1, q2):
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)

# stops = set(["http","www","img","border","home","body","a","about","above","after","again","against","all","am","an",
# "and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't",
# "cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from",
# "further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
# "herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
# "itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought",
# "our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
# "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're",
# "they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
# "weren't","what","what's","when","when's""where","where's","which","while","who","who's","whom","why","why's","with","won't","would",
# "wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves" ])

def prepare_ngram_interaction(path,out,ngram='unigram'):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('unigram_sim_hash,bigram_sim_hash\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1_ngram = str(row['question1_%s'%ngram])#.split()
            q2_ngram = str(row['question2_%s'%ngram])#.split()

            unigram_sim_hash = get_word_distance(q1_ngram,q2_ngram)
            bigram_sim_hash = get_word_2gram_distance(q1_ngram,q2_ngram)
            # trigram_sim_hash = get_word_2gram_distance(q1_ngram,q2_ngram)
            # if unigram_sim_hash<10:
            #     print q1_ngram,q2_ngram
            #     print(unigram_sim_hash,bigram_sim_hash)
            
            outfile.write('%s,%s\n' % (
                unigram_sim_hash,
                bigram_sim_hash,
                # trigram_sim_hash,
                ))
            c+=1
        end = datetime.now()

    print 'times:',end-start

prepare_ngram_interaction(path+'train_unigram.csv',path+'train_simhash_features.csv',ngram='unigram')
prepare_ngram_interaction(path+'test_unigram.csv',path+'test_simhash_features.csv',ngram='unigram')
