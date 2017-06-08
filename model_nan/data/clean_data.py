import sys
import cPickle as pickle
import csv




def build_substitute_word(table_path,train_path,test_path):
    vocab = set()
    reader = csv.reader(open(train_path))
    next(reader)
    for id, qid1, qid2, q1, q2, dup in reader:
        q1 = q1.split()
        for w in q1:
            if len(w) == 0: continue
            vocab.add(w)
        q2 = q2.split()
        for w in q2:
            if len(w) == 0: continue
            vocab.add(w)
    reader = csv.reader(open(test_path))
    next(reader)
    for id, q1, q2 in reader:
        q1 = q1.split()
        for w in q1:
            if len(w) == 0: continue
            vocab.add(w)
        q2 = q2.split()
        for w in q2:
            if len(w) == 0: continue
            vocab.add(w)

    data=open(table_path,'r').read().split('\n')
    sub_table=dict()
    missed=list()
    for line in data:
        it=line.split('->')
        wrong=it[0]

        correct=it[1]
        if correct.find('.')>=0:
            correct=it[1].split(',')[0]
        if correct.find(wrong)>=0:
            print correct,wrong
            continue
        if correct.endswith('ed') or wrong.endswith('ed'):
            continue
        if wrong not in vocab:
            print wrong
            continue
        sub_table[wrong]=correct
        missed.append(wrong)
    fw=open('mis_spelling.rule','w')
    for k in sub_table:
        fw.write(k+'->'+sub_table[k]+'\n')
    fw.close()

    with open('sub_table.pkl','w')as f:
        pickle.dump(sub_table,f)
    with open('missed.pkl','w')as f:
        pickle.dump(missed,f)


def build_accent_word(table_path):
    data=open(table_path,'r').read().split('\n')
    sub_table=dict()
    missed=list()
    for line in data:
        it=line.split('->')
        wrong=it[0]

        correct=it[1]
        sub_table[wrong]=correct
        missed.append(wrong)
    with open('accent_sub_table.pkl','w')as f:
        pickle.dump(sub_table,f)
    with open('accent_missed.pkl','w')as f:
        pickle.dump(missed,f)


import unicodedata


def clean_accents(train_filepath,test_filepath):
    with open('accent_sub_table.pkl','r')as f:
        sub_table=pickle.load(f)
    with open('accent_missed.pkl','r')as f:
        missed=pickle.load(f)
    '''
    reader = open(train_filepath).read().split('\n')[:-1]
    fw=open('raw_train.txt2','w')
    idxs=0
    for line in reader:
        idxs+=1
        if idxs%10000==0:
            print idxs

        q1, q2=line.strip().split('\t')

        for w in missed:
            for idx in range(len(q1)):
                if w in q1:
                    q1 = q1.replace(w, sub_table[w])

                if w in q2:
                    q2 = q2.replace(w, sub_table[w])
        fw.write(q1+'\t'+q2+'\n')
    fw.close()
    '''
    reader = open(test_filepath)

    fw=open('raw_test.txt2','w')
    idxs = 1
    for line in reader:
        idxs += 1
        if idxs % 10000 == 0:
            print idxs
        lined = line.split('\t')
        if len(lined)!=2:
            print idxs,lined
        q1,q2=lined
        #print q1
        #print q2
        for w in missed:
            if w in q1:
                q1 = q1.replace(w, sub_table[w])
                    # print q1
            if w in q2:
                q2 = q2.replace(w, sub_table[w])
        fw.write(q1 + '\t' + q2 )

    fw.close()


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
}
def reconstruct_abbrev(train_filepath,test_filepath):
    '''
    reader = open(train_filepath).read().split('\n')[:-1]
    fw=open('raw_train.txt3','w')
    idxs=0
    for line in reader:
        idxs+=1
        if idxs%10000==0:
            print idxs

        q1, q2=line.split('\t')

        q1 = str(q1).lower()
        for item in abbr_dict.items():
            q1 = q1.replace(item[0], item[1])

        q2 = str(q2).lower()
        for item in abbr_dict.items():
            q2 = q2.replace(item[0], item[1])

        fw.write(q1+'\t'+q2+'\n')
    fw.close()
    '''
    reader = open(test_filepath)

    fw=open('raw_test.txt3','w')
    idxs = 1
    for line in reader:
        idxs += 1
        if idxs % 10000 == 0:
            print idxs
        lined = line.split('\t')


        if len(lined) != 2:
            print idxs, lined

        q1, q2 = lined

        q1 = str(q1).lower()
        for item in abbr_dict.items():
            q1 = q1.replace(item[0], item[1])

        q2 = str(q2).lower()
        for item in abbr_dict.items():
            q2 = q2.replace(item[0], item[1])

        fw.write(q1 + '\t' + q2 )

    fw.close()

import nltk

def word_tokenize(train_filepath,test_filepath):

    reader = open(train_filepath).read().split('\n')[:-1]
    fw = open('raw_train.txt4', 'w')
    idxs = 0
    for line in reader:
        idxs += 1
        if idxs % 10000 == 0:
            print idxs

        q1, q2 = line.split('\t')

        q1=nltk.word_tokenize(q1.decode('utf-8'))
        q2=nltk.word_tokenize(q2.decode('utf-8'))
        fw.write(' '.join(q1).encode('utf-8') + '\t' + " ".join(q2).encode('utf-8') + '\n')
    fw.close()
    '''
    reader = open(test_filepath)

    fw = open('raw_test.txt4', 'w')
    idxs = 1
    for line in reader:
        idxs += 1
        if idxs % 10000 == 0:
            print idxs

        lined = line.split('\t')
        if len(lined) != 2:
            print idxs, lined

        q1, q2 = lined

        q1 = nltk.word_tokenize(q1.decode('utf-8'))
        q2 = nltk.word_tokenize(q2.decode('utf-8'))
        fw.write(' '.join(q1).encode('utf-8') + '\t' + " ".join(q2).encode('utf-8') + '\n')

    fw.close()
    '''



def reform_upper(train_filepath,test_filepath):
    with open('accent_sub_table.pkl','r')as f:
        sub_table=pickle.load(f)
    with open('accent_missed.pkl','r')as f:
        missed=pickle.load(f)
    reader = open(train_filepath).read().split('\n')[:-1]
    fw = open('raw_train.txt4', 'w')
    idxs = 0
    for line in reader:
        idxs += 1
        if idxs % 10000 == 0:
            print idxs

        q1, q2 = line.split('\t')

        for w in missed:
            for idx in range(len(q1)):
                if w in q1:
                    q1 = q1.replace(w, sub_table[w])

                if w in q2:
                    q2 = q2.replace(w, sub_table[w])
        fw.write(q1 + '\t' + q2 + '\n')
    fw.close()
    print  'loading another'
    reader = open(test_filepath)

    fw = open('raw_test.txt4', 'w')
    idxs = 1
    for line in reader:
        idxs += 1
        if idxs % 10000 == 0:
            print idxs
        q1, q2 = line.split('\t')

        for w in missed:
            if w in q1:
                q1 = q1.replace(w, sub_table[w])
                # print q1
            if w in q2:
                q2 = q2.replace(w, sub_table[w])
        fw.write(q1 + '\t' + q2 + '\n')

    fw.close()


def clean_mispelling_word(train_filepath, test_filepath):
    dat=open('mis_spelling.rule','r').read().split('\n')[:-1]
    wongset=set()
    subst=dict()
    for it in dat:
        wrong,wrte=it.split('->')
        wrong=' '+wrong.lower()+' '
        wrte=' '+wrte.lower()+' '
        wongset.add(wrong)
        subst[wrong]=wrte

    text = open(train_filepath).read()
    fw = open('raw_train.txt5', 'w')
    for w in wongset:
        if w in text:
            text = text.replace(w, subst[w])
    fw.write(text)
    fw.close()
    print 'finished....'


    print 'loading another'
    text = open(test_filepath).read()
    fw = open('raw_test.txt5', 'w')
    for w in wongset:
        if w in text:
            text = text.replace(w, subst[w])
    fw.write(text)
    fw.close()

import re
def clean_number(train_filepath,test_filepath):
    www_pat=(r'www\.','')
    com_pat=(r'\.com','')
    uk_pat=(r'\.co\.uk','')
    year=(r'[12][0-9]{3}->yEar','yEar')
    float=(r'[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?','fLoat')

    text = open(train_filepath).read()
    fw = open('raw_train.txt6', 'w')

    text = re.sub(www_pat[0], www_pat[1], text)
    text = re.sub(com_pat[0], com_pat[1], text)
    text = re.sub(uk_pat[0], uk_pat[1], text)
    text = re.sub(year[0], year[1], text)
    text = re.sub(float[0], float[1], text)
    fw.write(text)
    fw.close()
    print 'finished....'

    print 'loading another'
    text = open(test_filepath).read()
    fw = open('raw_test.txt6', 'w')
    text = re.sub(www_pat[0],www_pat[1], text)
    text = re.sub(com_pat[0],com_pat[1], text)
    text = re.sub(uk_pat[0],uk_pat[1], text)
    text = re.sub(year[0],year[1], text)
    text = re.sub(float[0],float[1], text)
    fw.write(text)
    fw.close()

def conver_to_csv(train_filepath,train_filepath4,test_filepath,test_filepath4):

    reader=csv.reader(open(train_filepath))

    csvlist=[]
    for id,qid1,qid2,q1,q2,dup in reader:
        csvlist.append([id,qid1,qid2,q1,q2,dup])

    ofile = open('raw_train.csv', "wb")
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    reader=open(train_filepath4,'r').read().split('\n')[:-1]
    for i in range(1,len(reader)):
        lined=reader[i].split('\t')
        if len(lined)!=2:
            print i, reader[i]
        q1,q2=lined
        csvlist[i][3]=q1
        csvlist[i][4]=q2
    writer.writerows(csvlist)



    reader=csv.reader(open( test_filepath))

    csvlist=[]
    for id,q1,q2 in reader:
        csvlist.append([id,q1,q2])

    ofile = open('raw_test.csv', "wb")
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    reader=open(test_filepath4,'r').read().split('\n')[:-1]
    for i in range(len(reader)):
        lined=reader[i].split('\t')
        if len(lined)!=2:
            print i, reader[i]
        q1,q2=lined
        csvlist[i+1][1]=q1
        csvlist[i+1][2]=q2
    writer.writerows(csvlist)





if __name__ == '__main__':
    # 1. substitute mispelling words
    #build_substitute_word('spelling.misc','train.csv','test.csv')
    #clean_substitute_word('raw_train.txt','raw_test.txt')
    #build_accent_word('tokens.rule')
    #clean_accents('raw_train.txt','raw_test.txt')
    #reconstruct_abbrev('raw_train.txt2','raw_test.txt2')
    #word_tokenize('raw_train.txt3','raw_test.txt3')
    #clean_mispelling_word('raw_train.txt4','raw_test.txt4')
    #clean_number('raw_train.txt5','raw_test.txt5')
    conver_to_csv("train.csv", "raw_train.txt6", 'test.csv', 'raw_test.txt6')




