import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from config import path
if __name__ == '__main__':
    
    seed = 1024
    np.random.seed(seed)

    ft = ['question1','question2']
    train = pd.read_csv(path+"train.csv")[ft]
    test = pd.read_csv(path+"test.csv")[ft]
    
    len_train = train.shape[0]
    lda = LatentDirichletAllocation(n_topics=10, doc_topic_prior=None, topic_word_prior=None, learning_method='batch', learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=8, verbose=1, random_state=seed)
    bow = CountVectorizer(ngram_range=(1,1),max_df=0.95,min_df=3,stop_words='english')
    vect_orig = make_pipeline(bow,lda)
    
    corpus = []
    for f in ft:
        train[f] = train[f].astype(str)
        test[f] = test[f].astype(str)
        corpus+=train[f].values.tolist()

    vect_orig.fit(
        corpus
        )

    for f in ft:
        train_lda = vect_orig.transform(train[f].values.tolist())
        test_lda = vect_orig.transform(test[f].values.tolist())

        pd.to_pickle(train_lda,path+'train_%s_lda.pkl'%f)
        pd.to_pickle(test_lda,path+'test_%s_lda.pkl'%f)