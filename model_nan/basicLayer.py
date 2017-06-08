import numpy as np
import theano
import theano.tensor as T


class basicLayer(object):
    def __init__(self,input,input_shape):
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(6. / 200),
                                              high=np.sqrt(6. / 200),
                                              size=(input_shape,1)),
                            dtype=theano.config.floatX)
        W = theano.shared(value=init_W, name='basic_W', borrow=True)
        init_b = np.zeros((1,), dtype=theano.config.floatX)
        b = theano.shared(value=init_b, name='basic_b',borrow=True)
        self.activation=T.nnet.sigmoid(T.dot(input,W)+b).flatten()
        self.params=[W,b]
