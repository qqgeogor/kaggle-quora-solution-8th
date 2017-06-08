import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import pool

class ConvPool(object):
    def __init__(self,input,
               filter_shape,   # 2*3*3
               input_shape,
               pool_size=(2,2)):
        """

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                                      filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                                     image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        #print input_shape
        #print filter_shape
        #assert input_shape[1] == filter_shape[1]

        self.input=input
        fan_in=np.prod(filter_shape[1:])

        fan_out=(filter_shape[0]*np.prod(filter_shape[2:])/
                 np.prod(pool_size))


        init_W=np.asarray(np.random.uniform(low=-np.sqrt(6./(fan_in+fan_out)),
                                      high=np.sqrt(6./(fan_in+fan_out)),
                                      size=filter_shape),
                          dtype=theano.config.floatX)
        self.W=theano.shared(value=init_W,name='W',borrow=True)

        init_b=np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b=theano.shared(value=init_b,borrow=True)

        conv_out=T.nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=input_shape
        )
        pool_out=pool.pool_2d(
            input=conv_out,
            ws=pool_size,
            ignore_border=True)

        self.activation=T.tanh(pool_out+self.b.dimshuffle('x',0,'x','x')).flatten(2)
        self.params=[self.W,self.b]




