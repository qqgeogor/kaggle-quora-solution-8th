import numpy as np
import cPickle as pickle

def save_model(f,model):
    ps={}
    for p in model.params:
        ps[p.name]=p.get_value()
    pickle.dump(ps,open(f,'wb'))

def load_model(f,model):
    ps=pickle.load(open(f,'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model

class TextIterator(object):
    def __init__(self,source,n_batch,maxlen=None,mode=0):

        self.source=open(source,'r')
        self.n_batch=n_batch
        self.maxlen=maxlen

        self.end_of_data=False
        self.mode=mode

    def __iter__(self):
        return self


    def reset(self):
        self.source.seek(0)

    def goto_line(self, line_index):
        for _ in range(line_index):
            self.source.readline()

    def next(self):
        if self.end_of_data:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        batch_p=[]
        batch_q = []
        batch_label = []
        try:
            while True:
                s=self.source.readline()
                if s=="":
                    raise IOError
                s=s.strip().split('\t')
                if len(s)!=3 and len(s)!=2:
                    raise IOError


                p=[int(w) for w in s[0].split(' ') if len(w)>0]
                q = [int(w) for w in s[1].split(' ') if len(w)>0]
                if len(s)==3:
                    label=int(s[2])
                if self.maxlen and len(p)>self.maxlen:
                    p=p[:self.maxlen]
                if self.maxlen and len(q)>self.maxlen:
                    q=q[:self.maxlen]

                batch_p.append(p)
                batch_q.append(q)
                if len(s)==3:
                    batch_label.append(label)
                if len(batch_p)>=self.n_batch:
                    break
        except IOError:
            self.end_of_data=True

        if len(batch_p)<=0 or len(batch_q)<=0:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        if self.mode==2:
            return prepare_data(batch_p,self.maxlen),prepare_data(batch_q,self.maxlen)
        else:
            return prepare_data(batch_p, self.maxlen), prepare_data(batch_q, self.maxlen),np.asarray(batch_label,dtype='int32')

def prepare_data(seqs_x,maxlen=10):
    lengths_x=[len(s) for s in seqs_x]
    n_samples=len(seqs_x)


    x=np.zeros((maxlen,n_samples)).astype('int32')
    x_mask=np.zeros((maxlen,n_samples)).astype('float32')

    for idx,s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx]=s_x
        x_mask[:lengths_x[idx],idx]=1.

    return x,x_mask


