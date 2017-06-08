# -*- coding:utf-8 -*-

import numpy

import theano
import theano.tensor as tt

theano.config.compute_test_value = 'raise'

# test max attention hidden
print '------------------------------------------------------'
print 'test max attention hidden'

axis = 1

x_att = tt.fmatrix()
x_att.tag.test_value = numpy.array([[0.2, 0.8], [0.6, 0.4]],
                               dtype=theano.config.floatX)

x = tt.tensor3()
x.tag.test_value = numpy.array([[[3, 2, 6], [5, 1, 4]], [[2, 1, 6], [6, 1, 5]]],
                               dtype=theano.config.floatX)

print tt.batched_dot(x_att, x).tag.test_value
print ''

# Identify the largest value in each row
x_att_argmax = tt.argmax(x_att, axis=axis, keepdims=True)

# Construct a row of indexes to the length of axis
indexes = tt.arange(x_att.shape[axis]).dimshuffle(
    *(['x' for dim1 in xrange(axis)] + [0] + ['x' for dim2 in xrange(x_att.ndim - axis - 1)]))

# Create a binary mask indicating where the maximum values appear
mask = tt.eq(indexes, x_att_argmax)

# Alter the original matrix only at the places where the maximum values appeared
x_prime = tt.set_subtensor(x_att[mask.nonzero()], 0)

print x_att[mask.nonzero()].tag.test_value
print ''
print x_prime.tag.test_value
print ''
print x[mask.nonzero()].tag.test_value

# test word attention

print '------------------------------------------------------'
print 'test word attention'

q1_hs = tt.tensor3()
q1_hs.tag.test_value = numpy.array([[[0., 0., 0.], [0.2, 0.8, 0.3], [0.6, 0.4, 0.3], [0.6, 0.6, 0.3]]],
                               dtype=theano.config.floatX)

q2_hs = tt.tensor3()
q2_hs.tag.test_value = numpy.array([[[0., 0., 0.], [0., 0., 0.], [0.4, 0.9, 0.3], [0.6, 0.9, 0.3]]],
                               dtype=theano.config.floatX)

q1_mask = tt.fmatrix()
q1_mask.tag.test_value = numpy.array([[0., 1., 1., 1.]],
                               dtype=theano.config.floatX)

q2_mask = tt.fmatrix()
q2_mask.tag.test_value = numpy.array([[0., 0., 1., 1.]],
                               dtype=theano.config.floatX)

att_w = tt.fmatrix()
att_w.tag.test_value = numpy.array([[0.2, 0.6, 0.3], [0.2, 0.8, 0.3], [0.6, 0.4, 0.3]],
                               dtype=theano.config.floatX)

new_hid = tt.dot(q1_hs, att_w).dimshuffle(0, 2, 1)
print new_hid.tag.test_value
print ''

e = tt.batched_dot(q2_hs, new_hid)
print e.tag.test_value
print ''

max_attn = tt.max(e, axis=1, keepdims=True)
tmp = e - max_attn
print tmp.tag.test_value
attn_masked = q2_mask.dimshuffle(0, 1, 'x') * tt.exp(e - max_attn)
print attn_masked.tag.test_value
print ''

attn_vec = q1_mask.dimshuffle(0, 'x', 1) * (attn_masked / attn_masked.sum(axis=1, keepdims=True))
print attn_vec.tag.test_value
print ''

attented_express_q1 = tt.batched_dot(attn_vec.dimshuffle(0, 2, 1), q2_hs)
print attented_express_q1.tag.test_value
print ''

max_attn = tt.max(e, axis=2, keepdims=True)
attn_masked = q1_mask.dimshuffle(0, 'x', 1) * tt.exp(e - max_attn)
attn_vec = q2_mask.dimshuffle(0, 1, 'x') * (attn_masked / attn_masked.sum(axis=2, keepdims=True))
attented_express_q2 = tt.batched_dot(attn_vec, q1_hs)
print attented_express_q2.tag.test_value
