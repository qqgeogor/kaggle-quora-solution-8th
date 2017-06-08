import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import path
import tqdm

train = pd.read_csv(path+"train_hashed.csv")
test = pd.read_csv(path+"test_hashed.csv")
len_train = train.shape[0]
data_all = pd.concat([train,test])

le = LabelEncoder()
corpus = data_all['question1_hash'].values.tolist()+data_all['question2_hash'].values.tolist()
le.fit(corpus)
data_all['question1_hash'] = le.transform(data_all['question1_hash'])
data_all['question2_hash'] = le.transform(data_all['question2_hash'])

def generate_doc(df,name,concat_name):
    res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
    res.columns = [name,'%s_doc'%concat_name]
    return res

question_doc = generate_doc(data_all,name='question1_hash',concat_name='question2_hash')

# question_doc['question2_hash_doc'] = question_doc.astype(str).apply(lambda x:x['question2_hash_doc']+" "+x['question1_hash'],axis=1)
X = []
for row in question_doc['question2_hash_doc'].astype(str).values.tolist():
    if len(row.split(' '))>1:
        X.append(row)

question_doc = generate_doc(data_all,name='question2_hash',concat_name='question1_hash')
# question_doc['question1_hash_doc'] = question_doc.astype(str).apply(lambda x:x['question1_hash_doc']+" "+x['question2_hash'],axis=1)

for row in question_doc['question1_hash_doc'].astype(str).values.tolist():
    if len(row.split(' '))>1:
        X.append(row)

question_doc = pd.DataFrame()
question_doc['question2_hash_doc'] = X
question_doc['question2_hash_doc'].astype(str).to_csv('question_doc.adjlist',index=False)

import commands
res = commands.getoutput("bash train_deepwalk.sh")
print res
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

q_dict = read_emb('../large_input/question.emb')

data_emb = []
for d in data_all['question1_hash'].values:
    q=d
    if q in q_dict:
        q_emb = q_dict[q].tolist()
    else:
        q_emb = [0]*64
    data_emb.append(q_emb)
data_emb = np.array(data_emb)
train_question1_deepwalk = data_emb[:len_train]
test_question1_deepwalk = data_emb[len_train:]

data_emb = []
for d in data_all['question2_hash'].values:
    q=d
    if q in q_dict:
        q_emb = q_dict[q].tolist()
    else:
        q_emb = [0]*64
    data_emb.append(q_emb)
data_emb = np.array(data_emb)
train_question2_deepwalk = data_emb[:len_train]
test_question2_deepwalk = data_emb[len_train:]

pd.to_pickle(train_question1_deepwalk,path+'train_question1_deepwalk.pkl')
pd.to_pickle(test_question1_deepwalk,path+'test_question1_deepwalk.pkl')
pd.to_pickle(train_question2_deepwalk,path+'train_question2_deepwalk.pkl')
pd.to_pickle(test_question2_deepwalk,path+'test_question2_deepwalk.pkl')
