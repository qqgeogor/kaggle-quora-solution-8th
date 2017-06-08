import numpy as np
import pandas as pd


seed = 1024
np.random.seed(seed)


train_x = pd.read_pickle('X_v2/train_dup_stats.pkl')
test_x = pd.read_pickle('X_v2/test_dup_stats.pkl')

numeric_feature = ['q1_dup','q2_dup','q1_dup+q2_dup']


for feature in numeric_feature:
    train_x['r_'+feature] = train_x[feature].rank(method='max')
    test_x['r_'+feature] = test_x[feature].rank(method='max')

test_x = test_x.reset_index(drop=True)
train_x.to_pickle('X_v2/train_dup_stats_rank.pkl')
test_x.to_pickle('X_v2/test_dup_stats_rank.pkl')