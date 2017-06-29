
# coding: utf-8

# In[13]:

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import log
from datetime import datetime
from csv import DictReader
from feat_utils import get_jaccard
#import matplotlib.pyplot as plt

from config import path,large_path


# In[15]:

def prepare_graph(paths):
    G = nx.Graph()

    idf_dict = dict()
    for path in paths:
        print(path)
        c = 0
        start = datetime.now()

        for t, row in enumerate(DictReader(open(path), delimiter=',')):
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            G.add_edge(q1,q2)
            c+=1
    end = datetime.now()
    print('times:',end-start)
    return G
    




def prepare_df_dict(paths,smooth=1.0):
    df_dict = dict()
    for path in paths:
        print path
        c = 0
        start = datetime.now()

        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])
            # q1 = str(hash(q1))
            # q2 = str(hash(q2))
            for key in [q1,q2]:
                df = df_dict.get(key,0)
                df+=1
                df_dict[key]=df

            c+=1
        end = datetime.now()
    print 'times:',end-start
    return df_dict

def prepare_hash_df(path,out,neighbour_dict,df_dict):
    n_qids = float(len(neighbour_dict.keys()))
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('max_entropy,min_entropy,jaccard,intersection,intersection_entropy\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])

            q1_df = neighbour_dict.get(q1,[])
            q2_df = neighbour_dict.get(q2,[])
            HA = 0.0
            for q in q1_df:
                q_df = df_dict.get(q,1)
                HA +=-(q_df/n_qids)*log(q_df/n_qids)

            HB = 0.0
            for q in q2_df:
                q_df = df_dict.get(q,1)
                HB +=-(q_df/n_qids)*log(q_df/n_qids)
            
            qmax = max(HA,HB)
            qmin = min(HA,HB)
            
            intersection = set(q1_df).intersection(set(q2_df))

            H_intersection = 0.0
            for q in intersection:
                q_df = df_dict.get(q,1)
                H_intersection +=-(q_df/n_qids)*log(q_df/n_qids)
            

            jaccard = get_jaccard(q1_df,q2_df) 


            outfile.write('%s,%s,%s,%s,%s\n' % (qmax, qmin,jaccard,len(intersection),H_intersection))
            
            c+=1
            end = datetime.now()


    print 'times:',end-start


print ("Start reading emb")
def read_emb(path):
    count=0
    f = open(path,'r')
    emb_dict = dict()
    for line in f:
        if count==0:
            count+=1
            continue
        line = line.split(' ')
        id = int(line[0])
        
        weights = line[1:]
        weights = np.array([float(i) for i in weights])
        count+=1
        emb_dict[id] = weights
    return emb_dict


def prepare_hash_emb(path,out,neighbour_dict,df_dict):
    n_qids = float(len(neighbour_dict.keys()))
    print path
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        columns = [
            'deepwalk_dist_mean',
            'deepwalk_dist_max',
            'deepwalk_dist_min',
            'deepwalk_dist_std',
        ]
        columns = ','.join(columns)
        outfile.write(columns+'\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1_hash'])
            q2 = str(row['question2_hash'])

            a2 = neighbour_dict.get(q1,[])
            b2 = neighbour_dict.get(q2,[])
            
            
            sims = []
            dists = []
            if len(a2)==0 or len(b2)==0:
                sims=[999.0]
                dists=[999.0]
            else:
                for i in range(len(a2)):
                    for j in range(len(b2)):
                        try:
                            worda = a2[i]
                            wordb = b2[j]
                            if worda=="" or wordb=="":
                                continue
                            # print worda,wordb
                            
                            # sim = embedder.n_similarity(worda, wordb)
                            worda = int(worda)
                            wordb = int(wordb)

                            va =df_dict[worda]
                            vb =df_dict[wordb]
                            try:
                                # sim = cosine(df_dict.transform_paragraph(worda),df_dict.transform_paragraph(wordb))
                                sim = cosine(va,vb)
                            except:
                                sim=999.0
                            # vector_diff = df_dict.transform_paragraph(worda)-df_dict.transform_paragraph(wordb)
                            vector_diff = va-vb
                            dist = np.sqrt(np.sum(vector_diff**2))

                            sims.append(sim)
                            dists.append(dist)
                        except Exception,e:
                            # print e
                            continue
            if len(sims)==0 or len(dists)==0:
                sims=[999.0]
                dists=[999.0]


            deepwalk_dist_mean = np.mean(dists)
            deepwalk_dist_max = np.max(dists)
            deepwalk_dist_min = np.min(dists)
            deepwalk_dist_std = np.std(dists)
            features = (
                deepwalk_dist_mean,
                deepwalk_dist_max,
                deepwalk_dist_min,
                deepwalk_dist_std,
            )
            outfile.write('%s,%s,%s,%s\n' % features)
            c+=1
            end = datetime.now()


    print 'times:',end-start


G = prepare_graph([path+'train_hashed.csv',path+'test_hashed.csv'])
cnt=0
max_clique=dict()
for clique in nx.find_cliques(G):
    if cnt%100000==0:
        print("deal cnts %d" %cnt)
    len_clique=len(clique)
    for item in clique:
        c = max_clique.get(item,[])
        if len(c)<len_clique:
            max_clique[item]=clique  
    cnt+=1


df_dict = prepare_df_dict([path+'train_hashed.csv',path+'test_hashed.csv'])

prepare_hash_df(path=path+'train_hashed.csv',out=path+'train_max_clique_entropy_features.csv',df_dict=df_dict,neighbour_dict=max_clique)
prepare_hash_df(path=path+'test_hashed.csv',out=path+'test_max_clique_entropy_features.csv',df_dict=df_dict,neighbour_dict=max_clique)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv(path+"train_hashed.csv")
test = pd.read_csv(path+"test_hashed.csv")
len_train = train.shape[0]
data_all = pd.concat([train,test])

le = LabelEncoder()
corpus = data_all['question1_hash'].values.tolist()+data_all['question2_hash'].values.tolist()
le.fit(corpus)
data_all['question1_hash'] = le.transform(data_all['question1_hash'])
data_all['question2_hash'] = le.transform(data_all['question2_hash'])

train['question1_hash'] = le.transform(train['question1_hash'])
train['question2_hash'] = le.transform(train['question2_hash'])

test['question1_hash'] = le.transform(test['question1_hash'])
test['question2_hash'] = le.transform(test['question2_hash'])
train.to_csv(path+'train_hashed_le.csv',index=False)
test.to_csv(path+'test_hashed_le.csv',index=False)

G = prepare_graph([path+'train_hashed_le.csv',path+'test_hashed_le.csv'])
cnt=0
max_clique=dict()
for clique in nx.find_cliques(G):
    if cnt%100000==0:
        print("deal cnts %d" %cnt)
    len_clique=len(clique)
    for item in clique:
        c = max_clique.get(item,[])
        if len(c)<len_clique:
            max_clique[item]=clique  
    cnt+=1

q_dict = read_emb(large_path+'question.emb')

prepare_hash_emb(path=path+'train_hashed_le.csv',out=path+'train_max_clique_neighbour_stats.csv',df_dict=q_dict,neighbour_dict=max_clique)
prepare_hash_emb(path=path+'test_hashed_le.csv',out=path+'test_max_clique_neighbour_stats.csv',df_dict=q_dict,neighbour_dict=max_clique)
