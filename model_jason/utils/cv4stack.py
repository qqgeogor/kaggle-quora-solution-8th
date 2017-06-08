# -*- coding:utf-8 -*-

from sklearn.cross_validation import StratifiedKFold
import numpy as np
import pickle
import pandas as pd


def get_idxes_of_cv(indexes, folderk):
    skf = StratifiedKFold(indexes, n_folds=5, shuffle=True, random_state=1024)
    for i, (ind_tr, ind_te) in enumerate(skf):
        if i == folderk:
            break
    return ind_tr, ind_te


def test():
    d = np.random.randint(0, 1, size=(1000,))
    ind_tr, ind_te = get_idxes_of_cv(d, 2)
    print ind_tr
    print ind_te
    print ind_tr.shape
    print ind_te.shape


def merge_valid(pred_f, subname=''):
    path = "../../data/"
    train = pd.read_csv(path + "train.csv")
    y = train.is_duplicate.apply(lambda x: int(x)).values.tolist()
    print len(y)
    # y = np.ones((404290, ))
    preds = np.zeros((len(y), ))
    for i in xrange(5):
        print 'folder', i
        ind_tr, ind_te = get_idxes_of_cv(y, i)
        print len(ind_te)
        with open(pred_f + '.%d%s.val%d' % (i, subname, i), 'r') as fp:
            t_preds = pickle.load(fp)
            print t_preds.shape
            for idx, ori_idx in enumerate(ind_te):
                preds[ori_idx] = t_preds[idx]
    with open('../cv/stack/%s%s.train.csv' % (pred_f.split('/')[-1], subname), 'w') as fo:
        fo.write('id,is_duplicate\n')
        for i, prob in enumerate(preds):
            fo.write('%d,%f\n' % (i, prob))


def merge_test(pred_f, subname=''):
    preds = []
    for i in xrange(5):
        print 'folder', i
        with open(pred_f + '.%d%s.test%d' % (i, subname, i), 'r') as fp:
            t_preds = pickle.load(fp)
            preds.append(t_preds.reshape(t_preds.shape[0], 1))
    preds = np.concatenate(preds, axis=1)
    preds = np.mean(preds, axis=1)
    with open('../cv/stack/%s%s.test.csv' % (pred_f.split('/')[-1], subname), 'w') as fo:
        fo.write('id,is_duplicate\n')
        for i, prob in enumerate(preds):
            fo.write('%d,%f\n' % (i, prob))


if __name__ == '__main__':
    # test()
    model = '../cv/model/jasonnetwork.bs300.h512.att_concat.drop25.watt1.b0.l0'
    subname = '.match'
    merge_valid(model, subname)
    merge_test(model, subname)
    model = '../cv/model/jasonnetwork.bs300.h512.att_concat.drop20.watt1.b0.l0'
    subname = '.naive'
    merge_valid(model, subname)
    merge_test(model, subname)
    model = '../cv/model/jasonnetwork.bs300.h512.att_concat.drop20.watt1.b0.l0'
    subname = '.diff'
    merge_valid(model, subname)
    merge_test(model, subname)
