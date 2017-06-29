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
# path = '../input/'


def prepare_ngram_interaction(path,path_o,out,ngram='distinct_unigram',ngram_o='unigram'):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('count_of_question1,count_of_question2\n')
        for row,row_o in zip(DictReader(open(path), delimiter=','),DictReader(open(path_o), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1_ngram = str(row['question1_%s'%ngram]).split()
            q2_ngram = str(row['question2_%s'%ngram]).split()

            q1_o = str(row_o['question1_%s'%ngram_o]).split()
            q2_o = str(row_o['question2_%s'%ngram_o]).split()

            count_of_question1 = get_count_of_question(q1_ngram)
            count_of_question2 = get_count_of_question(q2_ngram)

            # ratio_of_question1 = count_of_question1/(len(q1_o)+1.0)
            # ratio_of_question2 = count_of_question2/(len(q2_o)+1.0)
            
            # ratio_q1_of_q2 = count_of_question1/(count_of_question2+1.0)
            # abs_diff_q1_of_q2 = abs(count_of_question1-count_of_question2)
            
            outfile.write('%s,%s\n' % (
                count_of_question1,
                count_of_question2,
                # ratio_of_question1,
                # ratio_of_question2,
                # ratio_q1_of_q2,
                # abs_diff_q1_of_q2,
                ))
            c+=1
        end = datetime.now()

    print 'times:',end-start

prepare_ngram_interaction(path+'train_distinct_unigram.csv',path+'train_unigram.csv',path+'train_distinct_unigram_features.csv',ngram='distinct_unigram',ngram_o='unigram')
prepare_ngram_interaction(path+'test_distinct_unigram.csv',path+'test_unigram.csv',path+'test_distinct_unigram_features.csv',ngram='distinct_unigram',ngram_o='unigram')

prepare_ngram_interaction(path+'train_distinct_bigram.csv',path+'train_bigram.csv',path+'train_distinct_bigram_features.csv',ngram='distinct_bigram',ngram_o='bigram')
prepare_ngram_interaction(path+'test_distinct_bigram.csv',path+'test_bigram.csv',path+'test_distinct_bigram_features.csv',ngram='distinct_bigram',ngram_o='bigram')
