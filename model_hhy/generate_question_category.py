import pandas as pd
import numpy as np
import nltk
import scipy.stats as sps
from .utils import ngram_utils,split_data,nlp_utils,dist_utils
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA,NMF,SparsePCA,TruncatedSVD

seed = 1024
np.random.seed(seed)

path = '../X_v2/'

train = pd.read_pickle(path+'train_final_clean.pkl')
test = pd.read_pickle(path+'test_final_clean.pkl')
test['is_duplicated']=[-1]*test.shape[0]
feats= ['question1','question2']
train_value = train[feats].values
y = pd.read_csv('../data/train.csv')['is_duplicate']
data_all = pd.concat([train,test])[feats].values

#embedd
vector_size = 300
glove_dir =  'D:/glove/glove.840B.{0}d.txt'.format(vector_size)
Embedd_model = nlp_utils._get_embedd_Index(glove_dir)


def _find_begin_pattern(q):
    s = str(q).lower()
    if s.find('what')==0:
        return 'what'
    elif s.find('how')==0:
        return 'how'
    elif s.find('who')==0:
        return 'who'
    elif s.find('when')==0:
        return 'when'
    elif s.find('which')==0:
        return 'which'
    elif s.find('where')==0:
        return 'where'
    elif s.find('why')==0:
        return 'why'
    elif s.find('be')==0:
        return 'be'
    elif s.find('is')==0:
        return 'be'
    elif s.find('am')==0:
        return 'be'
    elif s.find('are')==0:
        return 'be'
    elif s.find('other')==0:
        return 'other'
    else:
        return 'unknown'

def category_same(q1,q2):
    return int(q1==q2)

q1_category = []
q2_category = []
q1_category_embedd = []
q2_category_embedd = []
for i in tqdm(range(data_all.shape[0])):
    q1_c = _find_begin_pattern(data_all[i][0])
    q2_c = _find_begin_pattern(data_all[i][1])
    q1_category.append(q1_c)
    q2_category.append(q2_c)
    q1_category_embedd.append(Embedd_model[q1_c])
    q2_category_embedd.append(Embedd_model[q2_c])

category_same_fea = list(map(category_same,q1_category,q2_category))
q1_embedd = np.array(q1_category_embedd)
q2_embedd = np.array(q2_category_embedd)

pca = TruncatedSVD(n_components=12,random_state=1123)#(samples,features)
pca.fit(q1_embedd.T)
pca_fea_q1 = pca.components_.T

pca = TruncatedSVD(n_components=12,random_state=1123)#(samples,features)
pca.fit(q2_embedd.T)
pca_fea_q2 = pca.components_.T

fea_all = np.hstack([np.array(category_same_fea).reshape(-1,1),pca_fea_q1,pca_fea_q2])

# #stats feature
# data_all = pd.DataFrame()
# data_all['category_q1'] = q1_category
# data_all['category_q2'] = q2_category
# cate_all = pd.DataFrame()
# cate_all['category'] = q1_category + q2_category
# cate_all = pd.DataFrame(pd.DataFrame(cate_all.category.value_counts()).reset_index().values,columns=['category','count'])
# data_all = pd.merge(data_all,cate_all,how='left', left_on='category_q1', right_on='category')
# data_all = pd.merge(data_all,cate_all,how='left', left_on='category_q2', right_on='category')
#
# data_all['category_min']=data_all.apply(lambda x:min(x['count_x'],x['count_y']),axis=1)
# data_all['category_max'] = data_all.apply(lambda x:max(x['count_x'],x['count_y']),axis=1)
# data_all['category_dis'] = data_all['category_max'] - data_all['category_min']
#
# stat_fea = data_all[['category_min','category_dis']].values
# fea_all = np.hstack([fea_all,stat_fea])

train_fea = fea_all[:train.shape[0]]
test_fea = fea_all[train.shape[0]:]
test_x = split_data.split_test(test_fea)

pd.to_pickle(train_fea,'../X_v2/train_question_type.pkl')
for i in range(6):
    pd.to_pickle(test_x[i], '../X_v2/test_question_type{0}.pkl'.format(i))




