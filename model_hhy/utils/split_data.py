

def split_test(test_data,stride=6,size=400000):
    test = []
    for i in range(stride):
        idx = i*size
        idx2 = (i+1)*size
        if i==(stride-1):
            test.append(test_data[idx:])
        else:
            test.append(test_data[idx:idx2])
    return test

def split_data(data,size=400000):
    data_x = []
    N = data.shape[0]
    stride = int(N/size)+1
    for i in range(stride):
        idx = i*size
        idx2 = (i+1)*size
        if i==(stride-1):
            data_x.append(data[idx:])
        else:
            data_x.append(data[idx:idx2])
    return data_x

def get_feature_importance(feature):
    import scipy.stats as sps
    import pandas as pd
    y_train = pd.read_csv('../data/train.csv')['is_duplicate']
    return  sps.spearmanr(feature,y_train)[0]

# import pickle
# pickle.dump(X_train,open("data_train.pkl", 'wb'), protocol=2)
#
# data_file=['test_deptree','test_glove_sim_dist','test_pca_glove',
#            'test_pca_pattern','test_w2w','test_pos','test_pca_char']
#
# path='../test/'
# for it in range(6):
#     tmp=[]
#     flist=[item+str(it) for item in data_file]
#     test=np.empty((400000,0))
#     if it==5:
#         test=np.empty((345796,0))
#     for f in flist:
#         test=np.hstack([test,pd.read_pickle(path+f+'.pkl')])
#     pickle.dump(test,open('data_test{0}.pkl'.format(it),'wb'),protocol=2)