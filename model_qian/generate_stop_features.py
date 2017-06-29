from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
from feat_utils import get_count_q1_in_q2
from feat_utils import get_ratio_q1_in_q2
from config import path
# path = '../input/'

stops = ["http","www","img","border","home","body","a","about","above","after","again","against","all","am","an",
"and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't",
"cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from",
"further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
"herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
"itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought",
"our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
"than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're",
"they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
"weren't","what","what's","when","when's""where","where's","which","while","who","who's","whom","why","why's","with","won't","would",
"wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves" ]


def prepare_ngram_interaction(path,out):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('count_of_stop_question1,ratio_of_stop_question1,count_of_stop_question2,ratio_of_stop_question2\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1_ngram = str(row['question1'].lower()).split()
            q2_ngram = str(row['question2'].lower()).split()

            count_of_stop_question1 = get_count_q1_in_q2(q1_ngram,stops)
            ratio_of_stop_question1 = get_ratio_q1_in_q2(q1_ngram,stops)

            count_of_stop_question2 = get_count_q1_in_q2(q2_ngram,stops)
            ratio_of_stop_question2 = get_ratio_q1_in_q2(q2_ngram,stops)


            outfile.write('%s,%s,%s,%s\n' % (
                count_of_stop_question1,
                ratio_of_stop_question1,
                count_of_stop_question2,
                ratio_of_stop_question2,
                ))
            c+=1
        end = datetime.now()

    print 'times:',end-start

prepare_ngram_interaction(path+'train_porter.csv',path+'train_porter_stop_features.csv')
prepare_ngram_interaction(path+'test_porter.csv',path+'test_porter_stop_features.csv')
