import numpy as np
import pandas as pd
import scipy.stats as sps

seed = 1024
np.random.seed(seed)
path = 'data/'


train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
data_all = pd.concat([train, test])
#question concat
#question concat
q_all = pd.DataFrame(np.hstack([train['question1'], test['question1'],
                   train['question2'], test['question2']]), columns=['question'])
q_all = pd.DataFrame(q_all.question.value_counts()).reset_index()
q_all.columns = ['question', 'time']
q_avg = q_all.time.mean()
q_max = q_all.time.max()


data_all = pd.merge(data_all,q_all,how='left', left_on='question1', right_on='question')
data_all = pd.merge(data_all,q_all,how='left', left_on='question2', right_on='question')
data_all = data_all.rename(columns = {'time_x':'q1_dup','time_y':'q2_dup'})
data_all = data_all[['id','qid1','qid2','q1_dup','q2_dup','test_id']]

data_all['q1_dup_ratio_q2_dup'] = data_all['q1_dup']/data_all['q2_dup']
data_all['q1_dup+q2_dup'] = data_all['q1_dup'] + data_all['q2_dup']
data_all['q1_q2_dup_same'] = (data_all['q1_dup']==data_all['q2_dup']).astype(int)
#overfitting?
data_all['dup_max']=data_all.apply(lambda x:max(x['q1_dup'],x['q2_dup']),axis=1)
data_all['dup_min']=data_all.apply(lambda x:min(x['q1_dup'],x['q2_dup']),axis=1)
data_all['dup_dis']=data_all['dup_max']-data_all['dup_min']
data_all['dup_dis_ratio']=data_all['dup_dis']/data_all['dup_max']
data_all['q1_dup-avg_dup'] = data_all['q1_dup']-q_avg
#data_all['q1_dup-max_dup'] = q_max-data_all['q1_dup']
data_all['q2_dup-avg_dup'] = data_all['q2_dup']-q_avg
#data_all['q2_dup-max_dup'] = q_max-data_all['q2_dup']
#data_all['dup_dis_ratio_min']=data_all['dup_dis']/train_x['dup_min']

test_x = data_all[data_all.test_id.notnull()]
test_x = test_x.reset_index(drop=True)
train_x = data_all[data_all.test_id.isnull()]
column = [col for col in train_x.columns if col not in ['id','qid1','qid2','test_id']]
train_x = train_x[column]
test_x = test_x[column]

#relate
for fe in train_x.columns:
    print(fe,sps.spearmanr(train_x[fe],train.is_duplicate)[0])


train_x.to_pickle(path+'train_dup_stats.pkl')
test_x.to_pickle(path+'test_dup_stats.pkl')