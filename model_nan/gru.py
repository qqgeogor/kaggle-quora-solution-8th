import numpy as np
import theano
import theano.tensor as T

class GRU:
    def __init__(self,rng,
                 n_input,n_hidden,
                 x,E,mask,
                 is_train=1,p=0.5):
        self.rng=rng

        self.n_input=n_input
        self.n_hidden=n_hidden
        self.f=T.nnet.sigmoid

        self.x=x
        self.E=E
        self.mask=mask
        self.is_train=is_train
        self.p=p

        # Update gate
        init_Wz=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input+n_hidden,n_hidden)),
                           dtype=theano.config.floatX)
        init_bz=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wz=theano.shared(value=init_Wz,name='Wz')
        self.bz=theano.shared(value=init_bz,name='bz')

        # Reset gate
        init_Wr=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input+n_hidden,n_hidden)),
                           dtype=theano.config.floatX)
        init_br=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wr=theano.shared(value=init_Wr,name='Wr')
        self.br=theano.shared(value=init_br,name='br')

        # Cell update
        init_Wxc=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                              high=np.sqrt(1./n_input),
                                              size=(n_input,n_hidden)),
                            dtype=theano.config.floatX)
        init_Whc=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                              high=np.sqrt(1./n_input),
                                              size=(n_hidden,n_hidden)),
                            dtype=theano.config.floatX)
        init_bc=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wxc=theano.shared(value=init_Wxc,name='Wxc')
        self.Whc=theano.shared(value=init_Whc,name='Whx')
        self.bc=theano.shared(value=init_bc,name='bc')

        # Params
        self.params=[self.Wz,self.bz,self.Wr,self.br,self.Wxc,self.Whc,self.bc]

        self.build()

    def build(self):
        state_pre=T.zeros((self.x.shape[-1],self.n_hidden),dtype=theano.config.floatX)
        def _recurrence(x_t,m,h_tm1):
            x_e=self.E[x_t,:]
            concated=T.concatenate([x_e,h_tm1],axis=1)

            # Update gate
            z_t=self.f(T.dot(concated,self.Wz) + self.bz )

            # Input fate
            r_t=self.f(T.dot(concated,self.Wr) + self.br )

            # Cell update
            c_t=T.tanh(T.dot(x_e,self.Wxc)+T.dot(r_t*h_tm1,self.Whc)+self.bc)

            # Hidden state
            h_t=(T.ones_like(z_t)-z_t) * c_t + z_t * h_tm1

            # masking
            h_t=h_t*m[:,None]

            return h_t

        h,_=theano.scan(fn=_recurrence,
                        sequences=[self.x,self.mask],
                        outputs_info=state_pre,
                        truncate_gradient=-1)

        # Dropout
        if self.p>0:
            drop_mask=self.rng.binomial(n=1,p=1-self.p,size=h.shape,dtype=theano.config.floatX)
            self.activation=T.switch(T.eq(self.is_train,1),h*drop_mask,h*(1-self.p))
        else:
            self.activation=T.switch(T.eq(self.is_train,1),h,h)
