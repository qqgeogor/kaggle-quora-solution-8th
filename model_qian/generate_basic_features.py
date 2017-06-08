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
        outfile.write('jaccard,dice,count_q1_in_q2,ratio_q1_in_q2,count_of_question1,count_of_question2,count_of_unique_question1,count_of_unique_question2,ratio_of_unique_question1,ratio_of_unique_question2,count_of_digit_question1,count_of_digit_question2,ratio_of_digit_question1,ratio_of_digit_question2\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1_ngram = str(row['question1_%s'%ngram]).split()
            q2_ngram = str(row['question2_%s'%ngram]).split()

            jaccard = get_jaccard(q1_ngram,q2_ngram)
            dice = get_dice(q1_ngram,q2_ngram)

            count_q1_in_q2 = get_count_q1_in_q2(q1_ngram,q2_ngram)
            ratio_q1_in_q2 = get_ratio_q1_in_q2(q1_ngram,q2_ngram)

            count_of_question1 = get_count_of_question(q1_ngram)
            count_of_question2 = get_count_of_question(q2_ngram)

            count_of_question_min = min(count_of_question1,count_of_question2)
            count_of_question_max = max(count_of_question1,count_of_unique_question2)
            
            count_of_unique_question1 = get_count_of_unique_question(q1_ngram)
            count_of_unique_question2 = get_count_of_unique_question(q2_ngram)
            
            count_of_unique_question_min = min(count_of_unique_question1,count_of_unique_question2)
            count_of_unique_question_max = max(count_of_unique_question1,count_of_unique_question2)
            
            ratio_of_unique_question1 = get_ratio_of_unique_question(q1_ngram)
            ratio_of_unique_question2 = get_ratio_of_unique_question(q2_ngram)
            
            ratio_of_unique_question_min = min(ratio_of_unique_question1,ratio_of_unique_question2)
            ratio_of_unique_question_max = max(ratio_of_unique_question1,ratio_of_unique_question2)
            
            count_of_digit_question1 = get_count_of_digit(q1_ngram)
            count_of_digit_question2 = get_count_of_digit(q2_ngram)
                        
            count_of_digit_question_min = min(count_of_digit_question1,count_of_digit_question2)
            count_of_digit_question_max = max(count_of_digit_question1,count_of_digit_question2)
            
            ratio_of_digit_question1 = get_ratio_of_digit(q1_ngram)
            ratio_of_digit_question2 = get_ratio_of_digit(q2_ngram)
                        
            ratio_of_digit_question_min = min(ratio_of_digit_question1,ratio_of_digit_question2)
            ratio_of_digit_question_max = max(ratio_of_digit_question1,ratio_of_digit_question2)
            
            
            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                jaccard, 
                dice,
                count_q1_in_q2,
                ratio_q1_in_q2,
                count_of_question1,
                count_of_question2,
                count_of_unique_question1,
                count_of_unique_question2,
                ratio_of_unique_question1,
                ratio_of_unique_question2,
                count_of_digit_question1,
                count_of_digit_question2,
                ratio_of_digit_question1,
                ratio_of_digit_question2,
                ))
            c+=1
        end = datetime.now()

    print 'times:',end-start

prepare_ngram_interaction(path+'train_unigram.csv',path+'train_unigram_features.csv',ngram='unigram')
prepare_ngram_interaction(path+'test_unigram.csv',path+'test_unigram_features.csv',ngram='unigram')

prepare_ngram_interaction(path+'train_bigram.csv',path+'train_bigram_features.csv',ngram='bigram')
prepare_ngram_interaction(path+'test_bigram.csv',path+'test_bigram_features.csv',ngram='bigram')
