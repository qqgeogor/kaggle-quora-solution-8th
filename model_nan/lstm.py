import numpy as np
import theano
import theano.tensor as T

class LSTM(object):
    def __init__(self,rng,
                 n_input,n_hidden,n_batch,
                 x,E,mask,
                 is_train=1,p=0.5):
        self.rng=rng

        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_batch=n_batch

        self.x=x
        self.E=E
        self.mask=mask
        self.is_train=is_train
        self.p=p

        self.f=T.nnet.sigmoid

        # Forget gate params
        init_Wf=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input+n_hidden,n_hidden)),
                           dtype=theano.config.floatX)
        init_bf=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wf=theano.shared(value=init_Wf,name='Wf')
        self.bf=theano.shared(value=init_bf,name='bf')

        # Input gate params
        init_Wi=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input+n_hidden,n_hidden)),
                           dtype=theano.config.floatX)
        init_bi=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wi=theano.shared(value=init_Wi,name='Wi')
        self.bi=theano.shared(value=init_bi,name='bi')

        # Cell gate params
        init_Wc=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input+n_hidden,n_hidden)),
                           dtype=theano.config.floatX)
        init_bc=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wc=theano.shared(value=init_Wc,name='Wc')
        self.bc=theano.shared(value=init_bc,name='bc')

        # Output gate params
        init_Wo=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input+n_hidden,n_hidden)),
                           dtype=theano.config.floatX)
        init_bo=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wo=theano.shared(value=init_Wo,name='Wo')
        self.bo=theano.shared(value=init_bo,name='bo')

        # Params
        self.params=[self.Wi,self.Wf,self.Wc,self.Wo,
                     self.bi,self.bf,self.bc,self.bo];

        self.build()

    def build(self):
        '''
            Compute the hidden state in an LSTM.
            params:
                x_t : Input Vector
                h_tm1: hidden varibles from previous time step.
                c_tm1: cell state from previous time step.
            return [h_t, c_t]
        '''
        def _recurrence(x_t,m,h_tm1,c_tm1):
            x_e=self.E[x_t,:]
            concated=T.concatenate([x_e,h_tm1],axis=-1)

            # Forget gate
            f_t=self.f(T.dot(concated,self.Wf) + self.bf)
            # Input gate
            i_t=self.f(T.dot(concated,self.Wi) + self.bi)

            # Cell update
            c_tilde_t=T.tanh(T.dot(concated,self.Wc) + self.bc)
            c_t=f_t * c_tm1 + i_t * c_tilde_t

            # Output gate
            o_t=self.f(T.dot(concated,self.Wo) + self.bo)

            # hidden state
            h_t= o_t * T.tanh(c_t)

            c_t=c_t * m[:,None]
            h_t=h_t * m[:,None]

            return [h_t,c_t]

        [h,c],_=theano.scan(fn=_recurrence,
                            sequences=[self.x,self.mask],
                            truncate_gradient=-1,
                            outputs_info=[dict(initial=T.zeros((self.n_batch,self.n_hidden))),
                                          dict(initial=T.zeros((self.n_batch,self.n_hidden)))])
        self.activation=h
        '''
        # Dropout
        if self.p>0:
            drop_mask=self.rng.binomial(n=1,p=1-self.p,size=h.shape,dtype=theano.config.floatX)
            self.activation=T.switch(T.eq(self.is_train,1),h*drop_mask,h*(1-self.p))
        else:
            self.activation=T.switch(T.eq(self.is_train,1),h,h)
        '''

        
            
                    
