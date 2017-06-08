import theano
import theano.tensor as T
import numpy as np

class Similarity(object):
    def __init__(self,x,y,metrics='eucdian'):
        if metrics=='eucdian':
            x=x.dimshuffle(1,0,2)
            y = y.dimshuffle(1,2,0)
            activation=T.batched_dot(x,y)
            #activation=T.tensordot(x,y,axes=-1)

        self.activation=activation.dimshuffle(0,'x',1,2)