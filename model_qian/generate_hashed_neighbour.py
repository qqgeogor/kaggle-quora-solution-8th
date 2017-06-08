from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
from ngram import getUnigram
import string
import random
seed =1024
random.seed(seed)




def prepare_hash_neighbour(paths):
    neighbour_dict = dict()
    for path in paths:
        print path
        c = 0
        start = datetime.now()
        # with open(out, 'w') as outfile:
            # outfile.write('question1_hash_count,question2_hash_count\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            # q1 = hash(q1)
            # q2 = hash(q2)
            
            l = neighbour_dict.get(q1,[])
            l.append(q2)
            neighbour_dict[q1] = l

            l = neighbour_dict.get(q2,[])
            l.append(q1)
            neighbour_dict[q2] = l

            # outfile.write('%s,%s\n' % (q1_idf, q2_idf))
            
            c+=1
            end = datetime.now()


        print 'times:',end-start
    return neighbour_dict

print("Generate neighbour dict")
neighbour_dict = prepare_hash_neighbour([path+'train_hashed.csv',path+'test_hashed.csv'])
print("Dumping")

out = path+'neighbour.csv'
with open(out, 'w') as outfile:
    outfile.write('question,ids\n')
    for k in neighbour_dict.keys():
        outfile.write('%s,%s\n'%(k,' '.join(neighbour_dict[k])))

print("End")
