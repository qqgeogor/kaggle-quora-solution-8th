from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
from ngram import getUnigram



path = "F:\\Quora\\"


def try_divide(x, y, val=0.0):
    """ 
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val

def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


def mean(x):
    return sum(x)/float(len(x))

def median(x):
    len_2 = len(x)/2
    return x[len_2]

def std(x):
    mean_x = mean(x)
    s = 0.0
    for xx in x:
        s+=(xx-mean_x)**2
    s/=len(x)
    s = sqrt(s)
    return s



def prepare_position_index(path,out):
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('min_pos_q1_in_q2,max_pos_q1_in_q2,mean_pos_q1_in_q2,median_pos_q1_in_q2,std_pos_q1_in_q2,min_pos_q2_in_q1,max_pos_q2_in_q1,mean_pos_q2_in_q1,median_pos_q2_in_q1,std_pos_q2_in_q1\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_unigram']).lower().split(' ')
            q2 = str(row['question2_unigram']).lower().split(' ')
            pos_list = get_position_list(q1,q2)
            min_pos_q1_in_q2 = min(pos_list)
            max_pos_q1_in_q2 = max(pos_list)
            mean_pos_q1_in_q2 = mean(pos_list)
            median_pos_q1_in_q2 = median(pos_list)
            std_pos_q1_in_q2 = std(pos_list)

            pos_list = get_position_list(q2,q1)
            min_pos_q2_in_q1 = min(pos_list)
            max_pos_q2_in_q1 = max(pos_list)
            mean_pos_q2_in_q1 = mean(pos_list)
            median_pos_q2_in_q1 = median(pos_list)
            std_pos_q2_in_q1 = std(pos_list)


            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                min_pos_q1_in_q2,
                max_pos_q1_in_q2,
                mean_pos_q1_in_q2,
                median_pos_q1_in_q2,
                std_pos_q1_in_q2,
                min_pos_q2_in_q1,
                max_pos_q2_in_q1,
                mean_pos_q2_in_q1,
                median_pos_q2_in_q1,
                std_pos_q2_in_q1
                ))


            c+=1
        end = datetime.now()


    print 'times:',end-start

prepare_position_index(path+'train_unigram.csv',path+'train_position_index.csv')
prepare_position_index(path+'test_unigram.csv',path+'test_position_index.csv')
