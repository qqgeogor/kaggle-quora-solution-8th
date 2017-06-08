import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .utils import np_utils,nlp_utils,dist_utils,split_data

#embedd path
path = '../data/'
vector_size = 100
glove_dir = path+'glove.6B.{0}d.txt'.format(vector_size)
EMBEDDING_FILE = os.path.join(glove_dir)

TRAIN_DATA_FILE = path + 'train_clean.pkl'
TEST_DATA_FILE = path + 'test_clean.pkl'
ft = ['clean_question1','clean_question2']

MISSING_VALUE_NUMERIC = -1

train_data = pd.read_pickle(TRAIN_DATA_FILE)[ft]
test_data = pd.read_pickle(TEST_DATA_FILE)[ft]
data_all = np.vstack([train_data,test_data])


#feature base class
class Glove_BaseEstimator():
    def __init__(self,model,vector_size=100,not_aggregator=1):
        self.aggregation_mode_prev = ["mean", "max",'min','median']
        self.aggregation_mode = ["mean", "std", "max", "median"]
        self.aggregator = [None if m == "" else getattr(np, m) for m in self.aggregation_mode]
        self.aggregator_prev = [None if m == "" else getattr(np, m) for m in self.aggregation_mode_prev]
        self.not_aggregator = not_aggregator
        self.vector_size = vector_size
        self.model = model

    def _get_valid_word_list(self,text):
        return [w for w in text.lower().split(" ") if w in self.model]

    #not oov ratio
    def _get_importance(self,text1,text2):
        len_prev_1 = len(text1.split(" "))
        len_prev_2 = len(text2.split(" "))
        len1 = len(self._get_valid_word_list(text1))
        len2 = len(self._get_valid_word_list(text2))
        imp = np_utils._try_divide(len1+len2,len_prev_1+len_prev_2)
        return  imp

    def _get_importance_each(self,text1):
        len_prev_1 = len(text1.split(" "))
        len1 = len(self._get_valid_word_list(text1))
        imp = np_utils._try_divide(len1,len_prev_1)
        return  imp

    #sent  mean vector
    def _get_centroid_vector(self, text):
        lst = self._get_valid_word_list(text)
        centroid = np.zeros(self.vector_size)
        for w in lst:
            centroid += self.model[w]
        if len(lst) > 0:
            centroid /= float(len(lst))
        return centroid

    def _get_n_similarity(self,text1,text2):
        lst1 = self._get_centroid_vector(text1)
        lst2 = self._get_centroid_vector(text2)
        if len(lst1)>0 and len(lst2)>0:
            return dist_utils._calc_similarity(lst1,lst2)

    def _get_n_similarity_imp(self,text1,text2):
        sim = self._calc_similarity(text1,text2)
        imp = self._get_importance(text1,text2)
        return sim*imp

    # v1 - v2
    def _get_centroid_vdiff(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return dist_utils._vdiff(centroid1, centroid2)

    #(v1-v2)^2
    def _get_centroid_rmse(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return dist_utils._rmse(centroid1, centroid2)

    def _get_centroid_rmse_imp(self, text1, text2):
        rmse = self._get_centroid_rmse(text1, text2)
        imp = self._get_importance(text1, text2)
        return rmse * imp

    def fit_transform(self,data_all):
        score = list(map(self.transform_one, data_all[:, 0], data_all[:, 1]))
        if self.not_aggregator:
            return score
        self.N = data_all.shape[0]
        res = np.zeros((self.N, len(self.aggregator_prev) * len(self.aggregator)), dtype=float)
        for m, aggregator_prev in enumerate(self.aggregator_prev):
            for n, aggregator in enumerate(self.aggregator):
                idx = m * len(self.aggregator) + n
                for i in range(self.N):
                    # process in a safer way
                    try:
                        tmp = []
                        for l in score[i]:
                            try:
                                s = aggregator_prev(l)
                            except:
                                s = MISSING_VALUE_NUMERIC
                            tmp.append(s)
                    except:
                        tmp = [MISSING_VALUE_NUMERIC]
                    try:
                        s = aggregator(tmp)
                    except:
                        s = MISSING_VALUE_NUMERIC
                    res[i, idx] = s
        return res
#feature class
class Glove_Centroid_Vector(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=1):
        super().__init__(model,vector_size,not_aggregator)
    def __name__(self):
        return "Word2Vec_glove%d_Centroid_Vector"%( self.vector_size)
    def transform_one(self, obs):
        return self._get_centroid_vector(obs)
    def fit_transform(self,data_all):
        fea1 = list(map(self.transform_one, data_all[:, 0]))
        fea2 = list(map(self.transform_one,data_all[:,1]))
        return fea1,fea2

class Glove_Importance_each(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=1):
        super().__init__(model,vector_size,not_aggregator)

    def __name__(self):
        return "Word2Vec_%d_Importance_each" % (self.vector_size)

    def transform_one(self, obs):
        return self._get_importance_each(obs)

    def fit_transform(self,data_all):
        fea1 = list(map(self.transform_one, data_all[:, 0]))
        fea2 = list(map(self.transform_one,data_all[:,1]))
        return fea1,fea2
# not oov ratio in two sentences
class Glove_Importance(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=1):
        super().__init__(model,vector_size,not_aggregator)
    def __name__(self):
        return "Word2Vec_%d_Importance"%( self.vector_size)
    def transform_one(self, obs,target):
        return self._get_importance(obs,target)

class Glove_N_Similarity(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=1):
        super().__init__(model,vector_size,not_aggregator)
    def __name__(self):
        return "Word2Vec_%d_Similarity" % (self.vector_size)

    def transform_one(self, obs,target):
        return self._get_n_similarity(obs,target)

    def set_centroid(self,lst1,lst2):
        self.lst1 = lst1
        self.lst2 = lst2
        self.has_centroid = True

    def fit_transform(self,data_all):
        if self.has_centroid:
            return list(map(dist_utils._calc_similarity,self.lst1,self.lst2))
        else:
            return  super().fit_transform(data_all)

class Glove_N_Similarity_imp(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=1):
        super().__init__(model,vector_size,not_aggregator)
    def __name__(self):
        return "Word2Vec_%d_Similarity_imp" % (self.vector_size)

    def transform_one(self, obs, target):
        return self._get_n_similarity_imp(obs, target)

    def set_centroid(self,lst1,lst2):
        self.lst1 = lst1
        self.lst2 = lst2
        self.has_centroid = True
    def set_imp(self,imp):
        self.imp = imp
        self.has_imp = True
    def fit_transform(self,data_all):
        if self.has_centroid:
            if self.has_imp:
                imp = self.imp
            else:
                imp = list(map(self._get_importance,data_all[:,0],data_all[:,1]))
            sim = list(map(dist_utils._calc_similarity,self.lst1,self.lst2))
            im_sim = np.array(imp)*np.array(sim)
            return im_sim.tolist()
        else:
            return  super().fit_transform(data_all)

class Glove_Centroid_RMSE(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=1):
        super().__init__(model,vector_size,not_aggregator)
    def __name__(self):
        return "Word2Vec_%d_RMSE" % (self.vector_size)

    def transform_one(self, obs, target):
        return self._get_centroid_rmse(obs, target)

    def set_centroid(self, lst1, lst2):
        self.lst1 = lst1
        self.lst2 = lst2
        self.has_centroid = True
    def fit_transform(self,data_all):
        if self.has_centroid:
            return list(map(dist_utils._rmse,self.lst1,self.lst2))
        else:
            return  super().fit_transform(data_all)

class Glove_Centroid_RMSE_IMP(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=1):
        super().__init__(model,vector_size,not_aggregator)
    def __name__(self):
        return "Word2Vec_%d_RMSE_IMP" % (self.vector_size)

    def set_centroid(self, lst1, lst2):
        self.lst1 = lst1
        self.lst2 = lst2
        self.has_centroid = True
    def set_imp(self,imp):
        self.imp = imp
        self.has_imp = True
    def transform_one(self, obs, target):
        return self._get_centroid_rmse_imp(obs, target)
    def fit_transform(self,data_all):
        if self.has_centroid:
            if self.has_imp:
                imp = self.imp
            else:
                imp = list(map(self._get_importance,data_all[:,0],data_all[:,1]))
            rms = list(map(dist_utils._rmse,self.lst1,self.lst2))
            im_rms = np.array(imp)*np.array(rms)
            return im_rms.tolist()
        else:
            return  super().fit_transform(data_all)

class Glove_Centroid_Vdiff(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=1):
        super().__init__(model,vector_size,not_aggregator)
    def __name__(self):
        return "Word2Vec_%d_Centroid_Vdiff" % (self.vector_size)

    def transform_one(self, obs, target):
        return self._get_centroid_vdiff(obs, target)
    def set_centroid(self, lst1, lst2):
        self.lst1 = lst1
        self.lst2 = lst2
        self.has_centroid = True
    def fit_transform(self,data_all):
        if self.has_centroid:
            return list(map(dist_utils._vdiff,self.lst1,self.lst2))
        else:
            return  super().fit_transform(data_all)
#each word sim
class Word2Vec_Sim(Glove_BaseEstimator):
    def __init__(self,model,vector_size=100,not_aggregator=0):
        super().__init__(model,vector_size,not_aggregator)
    def transform_one(self, obs, target):
        val_list = []
        obs_tokens = nlp_utils._tokenize(obs)
        target_tokens = nlp_utils._tokenize(target)
        for obs_token in obs_tokens:
            _val_list = []
            if obs_token in self.model:
                for target_token in target_tokens:
                    if target_token in self.model:
                        sim = dist_utils._calc_similarity(self.model[obs_token], self.model[target_token])
                        _val_list.append(sim)
            if len(_val_list) == 0:
                _val_list = [MISSING_VALUE_NUMERIC]
            val_list.append(_val_list)
        if len(val_list) == 0:
            val_list = [[MISSING_VALUE_NUMERIC]]
        return val_list

def get_Glove_Model(embedd_file):
    return  nlp_utils._get_embedd_Index(embedd_file)


if __name__ == '__main__':
    Glove_model = get_Glove_Model(EMBEDDING_FILE)

    print('generate CenterVec,Center_IMP,Vdiff')
    gc = Glove_Centroid_Vector(Glove_model)
    cv1,cv2 = gc.fit_transform(data_all)

    gi = Glove_Importance_each(Glove_model)
    im1,im2 = gi.fit_transform(data_all)
    gi2 = Glove_Importance(Glove_model)
    im = gi2.fit_transform(data_all)

    gcv = Glove_Centroid_Vdiff(Glove_model)
    gcv.set_centroid(cv1,cv2)
    vdif = np.array(gcv.fit_transform(data_all))

    fea_all = np.vstack([np.array(im1),np.array(im2),np.array(im)])
    print('feature shape ',fea_all.T.shape)
    # not calc oov
    ft_generators_1 = [
        Glove_N_Similarity,
        Glove_Centroid_RMSE,
    ]
    print('generate N_sim,Center_RMSE')
    for i,ft_c in enumerate (ft_generators_1):
        ft = ft_c(Glove_model)
        ft.set_centroid(cv1,cv2)
        fea = ft.fit_transform(data_all)
        fea_all = np.vstack([fea_all,np.array(fea)])
    print('feature shape ', fea_all.T.shape)

    print('generate N_sim,Center_RMSE+IMP')
    ft_generators_2 = [
        Glove_N_Similarity_imp,
        Glove_Centroid_RMSE_IMP,
    ]
    for i,ft_c in enumerate (ft_generators_2):
        ft = ft_c(Glove_model)
        ft.set_centroid(cv1,cv2)
        ft.set_imp(im)
        fea = ft.fit_transform(data_all)
        fea_all = np.vstack([fea_all,np.array(fea)])
    print('feature shape ', fea_all.T.shape)

    fea_all = fea_all.T


    cv1 = np.array(cv1)
    cv2 = np.array(cv2)
    train_vec_1 = cv1[:train_data.shape[0]]
    train_vec_2 = cv2[:train_data.shape[0]]
    train_diff = vdif[:train_data.shape[0]]
    train_vec = np.hstack([train_vec_1,train_vec_2,train_diff])
    pd.to_pickle(train_vec, path + 'train_glove_vec.pkl')

    test_vec_1 = cv1[train_data.shape[0]:]
    test_vec_2 = cv2[train_data.shape[0]:]
    test_diff =  vdif[train_data.shape[0]:]
    test_vec = np.hstack([test_vec_1,test_vec_2,test_diff])
    test_v_x = split_data.split_test(test_vec)
    for i in range(6):
        pd.to_pickle(test_v_x[i],path+'test_glove_vec{0}.pkl'.format(i))

    train_fea = fea_all[:train_data.shape[0]]
    test_fea = fea_all[train_data.shape[0]:]


    #word2word sim feature approximatly need  9 hours
    ws = Word2Vec_Sim(Glove_model,not_aggregator=0)
    print('generate w2w features')
    w2w_sim = ws.fit_transform(data_all)
    train_w2w = w2w_sim[:train_data.shape[0]]
    test_w2w = w2w_sim[train_data.shape[0]:]

    path = '../X_v3/'
    pd.to_pickle(train_fea,path+'train_glove_sim_dist.pkl')
    pd.to_pickle(test_fea,path+'test_glove_sim_dist.pkl')
    #
    # pd.to_pickle(test_vec,path+'test_glove_vec.pkl')
    pd.to_pickle(train_w2w,path+'train_w2w.pkl')
    pd.to_pickle(test_w2w,path+'test_w2w.pkl')

