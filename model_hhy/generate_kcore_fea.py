import pandas as pd
import numpy as np
import networkx as nx
import scipy.stats as sps
seed = 1024
np.random.seed(seed)

path = '../data/'

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
data_all = pd.concat([train, test])[['question1','question2']]

#generate id
q_all = pd.DataFrame(np.hstack([train['question1'], test['question1'],
                   train['question2'], test['question2']]), columns=['question'])
q_all = pd.DataFrame(q_all.question.value_counts()).reset_index()

q_num = dict(q_all.values)
q_index = {}
for i,key in enumerate(q_num.keys()):
    q_index[key] = i
data_all['qid1'] = data_all['question1'].map(q_index)
data_all['qid2'] = data_all['question2'].map(q_index)

print("df_all.shape:", data_all.shape) # df_all.shape: (2750086, 2)


#build graph
df = data_all
g = nx.Graph()
g.add_nodes_from(df.qid1)
edges = list(df[['qid1', 'qid2']].to_records(index=False))
g.add_edges_from(edges)
g.remove_edges_from(g.selfloop_edges())

print(len(set(df.qid1)), g.number_of_nodes()) # 4789604
print(len(df), g.number_of_edges()) # 2743365 (after self-edges)


#generate kcore feature
df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
print("df_output.shape:", df_output.shape)
NB_CORES = 20

for k in range(2, NB_CORES + 1):
    fieldname = "kcore{}".format(k)
    print("fieldname = ", fieldname)
    ck = nx.k_core(g, k=k).nodes()
    print("len(ck) = ", len(ck))
    df_output[fieldname] = 0
    df_output.ix[df_output.qid.isin(ck), fieldname] = k


df_core = df_output.iloc[:,1:]
df_core['max_kcore'] = df_core.apply(lambda row: max(row), axis=1)
df_core['qid'] = df_output['qid']
df_core = df_core.drop_duplicates()

q1_fea = pd.merge(data_all[['qid1']],df_core,how='left',left_on='qid1',right_on='qid')
q2_fea = pd.merge(data_all[['qid2']],df_core,how='left',left_on='qid2',right_on='qid')
q1_fea.drop(['qid1','qid'],inplace=1,axis=1)
q2_fea.drop(['qid2','qid'],inplace=1,axis=1)


all_fea = np.hstack([q1_fea,q2_fea])
train_fea = all_fea[:train.shape[0]]
test_fea = all_fea[train.shape[0]:]


pd.to_pickle(train_fea,'../X_v2/train_kcore.pkl')
pd.to_pickle(test_fea,'../X_v2/test_kcore.pkl')

