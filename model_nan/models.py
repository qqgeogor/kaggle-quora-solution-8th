import theano
import numpy as np
import theano.tensor as T

from gru import GRU
from lstm import LSTM
from convpool import ConvPool
from updates import *
from Similarity import Similarity
from basicLayer import basicLayer

if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.compile.nanguardmode import NanGuardMode
import logging
from logging.config import fileConfig

fileConfig('logging_config.ini')
log = logging.getLogger()


class RCNNModel(object):
    def __init__(self, n_input, n_vocab, n_hidden, cell='gru', optimizer='adam', dropout=0.1, sim='eucdian',maxlen=50,batch_size=20):

        self.x = T.imatrix('batched input query p')
        self.xmask = T.matrix('batched masked query p')
        self.y = T.imatrix('batched input query q')
        self.ymask = T.matrix('batched masked query q')
        self.label = T.ivector('batched similarity label')

        self.n_input = n_input  # input word dimension
        self.n_hidden = n_hidden  # hidden size
        self.batch_size=batch_size

        self.cell = cell
        self.optimizer = optimizer
        self.dropout = dropout
        self.sim = sim
        self.is_train = T.iscalar('is_train')
        self.maxlen=maxlen

        init_Embd = np.asarray(
            np.random.uniform(low=-np.sqrt(6. / (n_vocab + n_input)), high=np.sqrt(1. / (n_vocab + n_hidden)),
                              size=(n_vocab, n_input)),
            dtype=theano.config.floatX)

        self.E = theano.shared(value=init_Embd, name='word_embedding', borrow=True)

        self.rng = RandomStreams(1234)
        self.build()

    def build(self):
        log.info('building rnn cell....')
        if self.cell == 'gru':
            recurent_x = GRU(self.rng,
                             self.n_input,
                             self.n_hidden,
                             self.x, self.E, self.xmask,
                             self.is_train, self.dropout)

            recurent_y = GRU(self.rng,
                             self.n_input,
                             self.n_hidden,
                             self.y, self.E, self.ymask,
                             self.is_train, self.dropout)
        elif self.cell == 'lstm':
            recurent_x = LSTM(self.rng,
                              self.n_input,
                              self.n_hidden,
                              self.x, self.E, self.xmask,
                              self.is_train, self.dropout)

            recurent_y = LSTM(self.rng,
                              self.n_input,
                              self.n_hidden,
                              self.y, self.E, self.ymask,
                              self.is_train, self.dropout)
        log.info('build the sim matrix....')
        sim_layer = Similarity(recurent_x.activation, recurent_y.activation,metrics=self.sim)

        log.info('building convolution pooling layer....')
        conv_pool_layer = ConvPool(input=sim_layer.activation,
                                   filter_shape=(2,1,3,3), # feature_maps, 1, filter_h, filter_w
                                   input_shape=(self.batch_size,1,50,50))#sim_layer.activation.shape)
        projected_layer=basicLayer(conv_pool_layer.activation,input_shape=1152)
        rav_cost=T.nnet.binary_crossentropy(projected_layer.activation, self.label)
        cost = T.mean(rav_cost)
        acc=T.eq(projected_layer.activation>0.5,self.label)
        log.info('cost calculated.....')

        self.params = [self.E, ]
        self.params += recurent_x.params
        self.params += recurent_y.params
        self.params += conv_pool_layer.params
        self.params += projected_layer.params

        lr = T.scalar('lr')
        gparams = [T.clip(T.grad(cost, p), -3, 3) for p in self.params]
        #gparams = [T.grad(cost, p) for p in self.params]

        if self.optimizer == 'sgd':
            updates = sgd(self.params, gparams, lr)
        elif self.optimizer == 'adam':
            updates = adam(self.params, gparams, lr)
        elif self.optimizer == 'rmsprop':
            updates = rmsprop(self.params, gparams, lr)

        log.info('gradient calculated.....')

        self.train = theano.function(inputs=[self.x, self.xmask,self.y,self.ymask,self.label, lr],
                                     outputs=[cost,acc],
                                     updates=updates,
                                     givens={self.is_train: np.cast['int32'](1)})

        self.predict = theano.function(inputs=[self.x, self.xmask, self.y, self.ymask, self.label],
                                    outputs=[rav_cost,acc],
                                    givens={self.is_train: np.cast['int32'](0)})

        self.test = theano.function(inputs=[self.x, self.xmask, self.y, self.ymask],
                                     outputs=projected_layer.activation,
                                     givens={self.is_train: np.cast['int32'](0)})

