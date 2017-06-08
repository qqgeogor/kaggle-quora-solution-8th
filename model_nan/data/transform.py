import numpy as np
from random import shuffle
import operator
import csv
import cPickle as pickle
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold


VOCABULARY_SIZE=150000

def build_dataset(train_filepath,test_filepath):
    vocab=defaultdict(int)
    reader=csv.reader(open(train_filepath))
    next(reader)
    for id, qid1,qid2,q1,q2,dup in reader:
        q1=q1.split()
        for w in q1:
            if len(w)==0: continue
            vocab[w]+=1
        q2 = q2.split()
        for w in q2:
            if len(w)==0: continue
            vocab[w] += 1
    reader = csv.reader(open(test_filepath))
    next(reader)
    for id,q1,q2 in reader:
        q1=q1.split()
        for w in q1:
            if len(w)==0: continue
            vocab[w]+=1
        q2 = q2.split()
        for w in q2:
            if len(w)==0: continue
            vocab[w] += 1
    print len(vocab)
    sorted_vocab = sorted(vocab.items(), cmp=lambda x, y: cmp(x[1], y[1]), reverse=True)
    #oov = sorted_vocab[VOCABULARY_SIZE:]
    sorted_vocab=sorted_vocab[:VOCABULARY_SIZE]

    #with open('oov.pkl','w')as f:
    #    pickle.dump(oov,f)
    word2idx=dict()
    for widx,(w,_) in enumerate(sorted_vocab):
        word2idx[w]=widx

    with open('word2idx.pkl','w')as f:
        pickle.dump(word2idx,f)




def rewrite_corpus(vocab_path,train_filepath,test_filepath):
    word2idx=pickle.load(open(vocab_path))
    reader=csv.reader(open(train_filepath))
    fw=open('train.txt','w')
    next(reader)
    for id, qid1,qid2,q1,q2,dup in reader:
        q1 = q1.split()
        sent1 = [str(word2idx[w] + 1) if w in word2idx else '0' for w in q1]
        q2 = q2.split()
        sent2 = [str(word2idx[w] + 1) if w in word2idx else '0' for w in q2]

        fw.write(" ".join(sent1) + "\t" + " ".join(sent2)+"\t" +str(dup)+ "\n")
    fw.close()
    reader = csv.reader(open(test_filepath))
    fw = open('test.txt', 'w')
    next(reader)
    for id, q1,q2 in reader:
        q1=q1.split()
        sent1=[str(word2idx[w]+1) if w in word2idx else '0' for w in q1 ]
        q2=q2.split()
        sent2 = [str(word2idx[w] + 1) if w in word2idx else '0' for w in q2]
        fw.write(" ".join(sent1)+"\t"+" ".join(sent2)+"\n")
    fw.close()


def kflod(train_filepath):
    data=open(train_filepath,'r')
    y=[]
    p,q=[],[]
    for line in data:
        lines=line.split('\t')
        p.append(lines[0])
        q.append(lines[1])
        y.append(lines[2])

    skf=StratifiedKFold(y=y,n_folds=5,shuffle=True,random_state=1024)
    idx=0
    for train_idx,test_idx in skf:


        fw = open(train_filepath + '.train.' + str(idx), 'w')
        for id in train_idx:
            pp,qq,yy= p[id],q[id],y[id]
            fw.write( pp+'\t'+qq+'\t'+yy)
        fw.close()

        fw = open(train_filepath+".valid."+str(idx), 'w')
        for id in test_idx:
            pp, qq, yy = p[id], q[id], y[id]
            fw.write(pp + '\t' + qq + '\t' + yy )
        fw.close()
        idx+=1





#build_dataset('raw_train.csv','raw_test.csv')
rewrite_corpus('word2idx.pkl','raw_train.csv','raw_test.csv')
kflod('train.txt')
#clean_traindata('train.csv','test.csv')
