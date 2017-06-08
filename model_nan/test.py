import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy
rng=numpy.random.RandomState(1234)

input=T.tensor4(name='input')
w_shp=(2,3,9,9)
W_bound=numpy.sqrt(3*9*9)
W=theano.shared(numpy.asarray(
    rng.uniform(low=-1.0/W_bound,
                high=1.0/W_bound,
                size=w_shp),
    dtype=input.dtype),name='W')

b_shp = (2,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

conv_out=conv2d(input,W)

