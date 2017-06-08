#-*-coding:utf-8-*-
import pandas as pd

from stanford_parser import parser
from nltk.tree import Tree
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import warnings
warnings.filterwarnings(action='ignore')

path = './data/'
out_path = './train_data/'
standford_parser = parser.Parser()


#get features from the structure of the deptree
def getDepTree(x):
    # 构建语法分析工具
    tokens,tree = standford_parser.parse(unicode(x))
    posTag = standford_parser.getPosTag(tree)
    return str(tree),posTag

def getDepTreeHeight(x):
    #_, ret = standford_parser.parse(unicode(x))
    t = Tree.fromstring(str(x))
    return t.height()

#leaves
def getDepTreeLeaves(x):
    #_, ret = standford_parser.parse(unicode(x))
    t = Tree.fromstring(str(x))
    return len(t.leaves())

#root
def getTreeRootPos(x):
    #_, ret = standford_parser.parse(unicode(x))
    t = Tree.fromstring(str(x))
    return t.pos()[0][1]

#wh- question tag
def getHasWH(x):
    #set1 = set(standford_parser.parseToStanfordDependencies(unicode(x)).posTags)
    set1 = set(x)
    if 'WDT' in set1:
        return 'WDT'
    elif 'WP' in set1:
        return 'WP'
    elif 'WP$'in set1:
        return 'WP$'
    elif 'WRB' in set1:
        return 'WRB'
    else:
        return 'NO'

#compare pos
def getSamePos(x):
    pos1 = x['q1PosTag']
    pos2 = x['q2PosTag']
    set1 = set(pos1)
    set2 = set(pos2)
    return len(set1.intersection(set2))

def getSamePosR(x):
    l1 = len(x['q1PosTag'])
    l2 = len(x['q2PosTag'])
    return (x['same_pos']*2.0)/(l1+l2)

def generate_some_feautre(df):
    df['hei_diff'] = 0
    df['leaves_diff'] = 0
    df['Root_is_same'] = 0
    df['same_pos'] = 0
    df['same_pos_R'] = 0
    df['WH_same'] = 0
    return df


def generateTreeFeature(df):
    print('generate---features!')
    df['q1hei'] = df['q1tree'].apply(getDepTreeHeight)
    df['q1leaves'] = df['q1tree'].apply(getDepTreeLeaves)
    df['q1RootPos'] = df['q1tree'].apply(getTreeRootPos)
    df['q2hei'] = df['q2tree'].apply(getDepTreeHeight)
    df['q2leaves'] = df['q2tree'].apply(getDepTreeLeaves)
    df['q2RootPos'] = df['q2tree'].apply(getTreeRootPos)
    df['hei_diff'] = abs(df['q1hei'] - df['q2hei'])
    df['leaves_diff'] = abs(df['q1leaves']-df['q2leaves'])
    df['Root_is_same'] = (df['q1RootPos']==df['q2RootPos']).astype(int)
    df['q1WHPos'] = df['q1PosTag'].apply(getHasWH)
    df['q2WHPos'] = df['q2PosTag'].apply(getHasWH)
    df['same_pos'] = df.apply(lambda x:getSamePos(x),axis=1,raw=True)
    df['same_pos_R']= df.apply(lambda x:getSamePosR(x),axis=1,raw=True)
    df['WH_same'] = (df['q1WHPos']==df['q2WHPos']).astype(int)
    print('end---generate---TreePos')
    return df





def generateTreeAndPos(st,ed,df):
    q1_tree = []
    q1_pos = []
    q2_tree = []
    q2_pos = []
    for i in range(st,ed,1):
        t1,p1 = getDepTree(df['question1'][i])
        t2,p2 = getDepTree(df['question2'][i])
        q1_tree.append(t1)
        q1_pos.append(p1)
        q2_tree.append(t2)
        q2_pos.append(p2)
    cur_df = df.iloc[st:ed, :]
    cur_df['q1tree'] = q1_tree
    cur_df['q2tree'] = q2_tree
    cur_df['q1PosTag'] = q1_pos
    cur_df['q2PosTag'] = q2_pos
    return cur_df


if __name__ == '__main__':

    train = pd.read_csv(path+'train.csv')
    train = train[['id','question1','question2']]

    #first create tree
    #tree_features = []
    batch_size = 5000
    epoch = int(train.shape[0]*1.0/batch_size)+1
    for i in range(0,epoch):
        print('generate batch---{0}---'.format(i+1))
        st = i*batch_size
        et = (i+1)*batch_size
        if et>train.shape[0]:et = train.shape[0]
        #cur_feature = generateTreeFeature(train,st,et)
        cur_batch = generateTreeAndPos(st,et,train)
        cur_batch = generateTreeFeature(cur_batch)
        pd.to_pickle(cur_batch, out_path + "train_DepTree{0}.pkl".format(i))
        print('batch---{0}----end'.format(i+1))

