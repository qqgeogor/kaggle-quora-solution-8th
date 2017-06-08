import pandas as pd
import numpy as np
import string
import re

path = "../input/"
ft = ['question1','question2']
train = pd.read_csv(path+"train_porter.csv")[ft]
test = pd.read_csv(path+"test_porter.csv")[ft]
ft = ['question1','question2']
train = pd.read_csv(path+"train_porter.csv")[ft]
test = pd.read_csv(path+"test_porter.csv")[ft]


def hasNumber(str1):
    return int(any(ch.isdigit() for ch in str1))


def getDiffNumber(str1, str2):
    num1 = re.findall(r'\d+', str1)
    num2 = re.findall(r'\d+', str1)
    return int(num1 == num2)

def getNumber(str1):
    return re.findall(r'\d+', str(str1))

train['number1']=train.question1.apply(getNumber)
train['number2']=train.question2.apply(getNumber)

test['number1']=test.question1.apply(getNumber)
test['number2']=test.question2.apply(getNumber)

train['q1_number_len']=train.number1.map(len)
train['q2_number_len']=train.number2.map(len)

test['q1_number_len']=test.number1.map(len)
test['q2_number_len']=test.number2.map(len)

train['number1']=train.question1.apply(getNumber)
train['number2']=train.question2.apply(getNumber)

train['q1_q2_number_diff']=(train.number1==train.number2).astype(float)
test['q1_q2_number_diff']=(test.number1==test.number2).astype(float)

pd.to_pickle(train[['q1_number_len','q2_number_len','q1_q2_number_diff']].values,path+'train_number_diff.pkl')
pd.to_pickle(test[['q1_number_len','q2_number_len','q1_q2_number_diff']].values,path+'test_number_diff.pkl')

