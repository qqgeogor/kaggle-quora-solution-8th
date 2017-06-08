# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

# import nltk
from nltk.stem import SnowballStemmer
# from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text

import lasagne
import theano
import theano.tensor as T
from lasagne.regularization import regularize_network_params, l1, l2
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import pickle
import cPickle
import logging
import sys
import time
import warnings

from keras.preprocessing import sequence
from utils.vocab import buildVocab, Vocab
from utils.cv4stack import get_idxes_of_cv

__authors__ = ['bowenwu']


is_train = True
fast_debug = False
balance = False
large_debug_cases = True

cv = 5

batch_size = 300
embedding_size = 200
hidden_size = 512
# lr = 0.0003
lr = 0.001
reg_type = None
reg_rate = 1e-4
drop_rate = 0.2
watt_drop_rate = 0.2
batch_norm = False
hyper_lstm = False

# concat, general
attn_type = 'concat'

# class_weight = {0: 1.309028344, 1: 0.472001959}
class_weight = {0: 1., 1: 1.}


# general
word_level_att = None
# word_level_att = 'general'

leak_f = None
# leak_f = './features/%s_leak_stand.pkl' % ('train' if is_train else 'test')


leak_num = 85 if leak_f else 0


warnings.filterwarnings(action='ignore')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
if is_train and not fast_debug and cv < 0:
    logPath = './log'
    logFile = 'tmp_drop%d_watt%d_b%d_l%d' % (
        int(drop_rate * 100), (1 if word_level_att else 0), (1 if balance else 0), (1 if leak_f else 0))
    fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, logFile))
    logger.addHandler(fileHandler)

snowball_stemmer = SnowballStemmer('english')
# wordnet_lemmatizer = WordNetLemmatizer()


def load_pretrain_emb(fname):
    if not fname:
        return None
    logger.info('load pretrain embedding %s', fname)
    dic = {}
    with open(fname, 'r') as fp:
        for line in fp:
            items = line.strip().split()
            rword = items[0]
            try:
                word = snowball_stemmer.stem(rword)
                # word = wordnet_lemmatizer.lemmatize(word)
                dic[word] = map(float, items[1:])
            except Exception:
                dic[rword] = map(float, items[1:])
    return dic


class JasonNetwork(object):

    def __init__(self, batch_size, max_seq_len, embedding_size, hidden_size, vocab,
                 pretrain_dict=None, backlstm=False, word_level_att=True, hyper_lstm=False, batch_norm=False,
                 lr=0.0001, reg_type=None, reg_rate=1e-4, drop_rate=0, watt_drop_rate=0, attn_type=None, leak_num=0, fast_debug=False):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size
        self.attn_type = attn_type
        self.lr = lr
        self.drop_rate = drop_rate
        self.reg_type = None
        self.reg_rate = reg_rate
        self.reg = None
        self.backlstm = backlstm
        self.hidden_output_size = hidden_size if (not backlstm or hyper_lstm) else (
            hidden_size * 2)
        self.vocab = vocab
        self.pretrain_dict = pretrain_dict
        self.hyper_lstm = hyper_lstm
        self.word_level_att = word_level_att
        self.batch_norm = batch_norm
        self.leak_num = leak_num
        self.fast_debug = fast_debug
        self.watt_drop_rate = watt_drop_rate
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

        if reg_type == "l1":
            self.reg_type = l1
        elif reg_type == "l2":
            self.reg_type = l2

        self.parameters = []

        if attn_type:
            attn_parameters = []
            if self.attn_type == "concat":
                self.V_enc_to_attn = lasagne.utils.create_param(
                    lasagne.init.GlorotUniform(), (self.hidden_output_size, 1), "V_enc_to_attn")
                self.W_enc_to_attn = lasagne.utils.create_param(
                    lasagne.init.GlorotUniform(), (2 * self.hidden_output_size, self.hidden_output_size), "W_enc_to_attn")
                self.b_enc_to_attn = lasagne.utils.create_param(
                    lasagne.init.Constant(0.), (self.hidden_output_size, ), "b_enc_to_attn")
                attn_parameters = [self.V_enc_to_attn, self.W_enc_to_attn]
            elif self.attn_type == "general":
                self.W_enc_to_attn = lasagne.utils.create_param(
                    lasagne.init.GlorotUniform(), (self.hidden_output_size, self.hidden_output_size), "W_enc_to_attn")
                attn_parameters = [self.W_enc_to_attn]
            self.parameters += attn_parameters

        if word_level_att:
            wattn_parameters = []
            if self.word_level_att == "concat":
                self.V_enc_to_wattn = lasagne.utils.create_param(
                    lasagne.init.GlorotUniform(), (self.hidden_output_size, 1), "V_enc_to_wattn")
                self.W_enc_to_wattn = lasagne.utils.create_param(
                    lasagne.init.GlorotUniform(), (2 * self.hidden_output_size, self.hidden_output_size), "W_enc_to_wattn")
                self.b_enc_to_wattn = lasagne.utils.create_param(
                    lasagne.init.Constant(0.), (self.hidden_output_size, ), "b_enc_to_wattn")
                wattn_parameters = [self.V_enc_to_wattn, self.W_enc_to_wattn]
            elif self.word_level_att == "general":
                self.W_enc_to_wattn = lasagne.utils.create_param(
                    lasagne.init.GlorotUniform(), (self.hidden_output_size, self.hidden_output_size), "W_enc_to_wattn")
                wattn_parameters = [self.W_enc_to_wattn]
            self.parameters += wattn_parameters

        if fast_debug:
            theano.config.allow_gc = False
            theano.config.mode = "FAST_COMPILE"
            theano.config.optimizer = 'None'
            theano.config.compute_test_value = 'warn'
            theano.config.exception_verbosity = 'high'
            theano.config.floatX = 'float32'
            # theano.config.warn_float64 = 'raise'
            theano.gof.compilelock.set_lock_status(False)

    def load_model(self, model_path):
        params = cPickle.load(open(model_path))
        for p, v in zip(self.parameters, params):
            if p.get_value().shape != v.shape:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" %
                                 (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

    def save_model(self, model_path):
        fout = open(model_path, 'w')
        params = [p.get_value() for p in self.parameters]
        cPickle.dump(params, fout)
        fout.close()

    def _build_encoder(self):
        logger.debug(">>> Encoder <<<")

        en_x_q1_sym = T.imatrix()
        en_xmask_q1_sym = T.fmatrix()
        en_x_q2_sym = T.imatrix()
        en_xmask_q2_sym = T.fmatrix()

        en_x = np.random.randint(0, 10, size=(
            self.batch_size, self.max_seq_len)).astype("int32")
        en_xmask = np.zeros(
            (self.batch_size, self.max_seq_len)).astype("float32")

        l_en_x = lasagne.layers.InputLayer((self.batch_size, self.max_seq_len))
        l_en_mask = lasagne.layers.InputLayer(
            (self.batch_size, self.max_seq_len))
        l_en_emb = lasagne.layers.EmbeddingLayer(
            l_en_x, self.vocab_size, self.embedding_size, name="EnEmbeddingLayer")
        self.l_emb = l_en_emb
        l_emb_drop = lasagne.layers.DropoutLayer(l_en_emb, p=self.drop_rate)
        l_en_enc = lasagne.layers.LSTMLayer(l_emb_drop, num_units=self.hidden_size, unroll_scan=False, backwards=False,
                                            gradient_steps=-1, name="EnLSTMLayer", mask_input=l_en_mask, only_return_final=False)
        if self.backlstm:
            l_en_enc_back = lasagne.layers.LSTMLayer(l_emb_drop, num_units=self.hidden_size, unroll_scan=False, backwards=True,
                                                     gradient_steps=-1, name="EnLSTMLayerBack", mask_input=l_en_mask, only_return_final=False)
            l_en_enc = lasagne.layers.ConcatLayer(
                [l_en_enc, l_en_enc_back], axis=2)
            if self.hyper_lstm:
                if self.batch_norm:
                    l_en_enc = lasagne.layers.batch_norm(l_en_enc)
                l_en_enc = lasagne.layers.LSTMLayer(l_en_enc, num_units=self.hidden_size, unroll_scan=False, backwards=False,
                                                    gradient_steps=-1, name="EnLSTMLayerHyper", mask_input=l_en_mask, only_return_final=False)
        if self.batch_norm:
            l_en_enc = lasagne.layers.batch_norm(l_en_enc)
        l_en_enc_drop = lasagne.layers.DropoutLayer(l_en_enc, p=self.drop_rate)

        en_enc_q1_sym = lasagne.layers.get_output(
            l_en_enc_drop, inputs={l_en_x: en_x_q1_sym, l_en_mask: en_xmask_q1_sym}, deterministic=False)
        en_enc_q2_sym = lasagne.layers.get_output(
            l_en_enc_drop, inputs={l_en_x: en_x_q2_sym, l_en_mask: en_xmask_q2_sym}, deterministic=False)
        logger.debug("Encoder Layer: %s", en_enc_q1_sym.eval(
            {en_x_q1_sym: en_x, en_xmask_q1_sym: en_xmask}).shape)

        en_enc_q1_sym_test = lasagne.layers.get_output(
            l_en_enc_drop, inputs={l_en_x: en_x_q1_sym, l_en_mask: en_xmask_q1_sym}, deterministic=True)
        en_enc_q2_sym_test = lasagne.layers.get_output(
            l_en_enc_drop, inputs={l_en_x: en_x_q2_sym, l_en_mask: en_xmask_q2_sym}, deterministic=True)

        encoder_parameters = lasagne.layers.get_all_params(
            [l_en_enc_drop], trainable=True)

        if self.reg_type:
            self.reg = regularize_network_params(l_en_enc_drop, self.reg_type)

        if self.pretrain_dict:
            self._load_pretrain_embedding_weights(
                self.vocab, self.pretrain_dict)

        en_out_syms_train = [en_enc_q1_sym, en_enc_q2_sym]
        en_out_syms_test = [en_enc_q1_sym_test, en_enc_q2_sym_test]

        return [en_x_q1_sym, en_xmask_q1_sym, en_x_q2_sym, en_xmask_q2_sym], en_out_syms_train, en_out_syms_test, encoder_parameters

    def attn_func(self, hidden, attn_input, attn_mask):
        """Attention function
        Args:
            hidden (theano symbol): size: [batch * hidden_size]
            attn_input (theano symbol): size: [batch * time_step * hidden_size]
        """
        batch_size, time_step, hidden_size = attn_input.shape
        # attn_mask = 1 - T.eq(attn_input, 0)
        # attn_mask = attn_mask[:, :, 0]  # shape of [batch * time_step]
        if self.attn_type == "dot":
            e = T.batched_dot(attn_input, hidden.dimshuffle(
                0, 1, 'x')).flatten(ndim=2)
        elif self.attn_type == "general":
            new_hid = T.dot(hidden, self.W_enc_to_attn).dimshuffle(0, 1, 'x')
            e = T.batched_dot(attn_input, new_hid).flatten(ndim=2)
        elif self.attn_type == "concat":
            hidden_exp = hidden.dimshuffle(0, "x", 1).repeat(time_step, axis=1)
            # Align with mask in encoder.
            hidden_exp = attn_mask.dimshuffle(0, 1, 'x') * hidden_exp
            concated = T.concatenate((attn_input, hidden_exp), axis=2)
            concated = T.tanh(
                T.dot(concated, self.W_enc_to_attn)) + self.b_enc_to_attn
            e = T.dot(concated, self.V_enc_to_attn).flatten(ndim=2)
        else:
            raise ValueError(
                'Attention type must be either "dot", "concat" or "general"')

        max_attn = T.max(e, axis=1, keepdims=True)
        attn_masked = attn_mask * T.exp(e - max_attn)
        attn_vec = (attn_masked / attn_masked.sum(axis=1,
                                                  keepdims=True)).reshape((batch_size, time_step))

        return attn_vec

    def word_level_attn_func(self, q1_hs, q1_hs_mask, q2_hs, q2_hs_mask):
        """Attention function
        Args:
            q1_hs (theano symbol): size: [batch * time_step * hidden_size]
            q2_hs (theano symbol): size: [batch * time_step * hidden_size]
        """
        _type = self.word_level_att
        batch_size, time_step, hidden_size = q1_hs.shape
        if _type == "general":
            new_hid = T.tanh(T.dot(q1_hs, self.W_enc_to_wattn).dimshuffle(0, 2, 1))
            e = T.batched_dot(q2_hs, new_hid)
        elif _type == "concat":
            q1_hs_exp = q1_hs.dimshuffle(
                0, 1, "x", 2).repeat(time_step, axis=2)
            q2_hs_exp = q2_hs.dimshuffle(
                0, "x", 1, 2).repeat(time_step, axis=1)
            concated = T.concatenate((q1_hs_exp, q2_hs_exp), axis=3)
            concated = T.tanh(
                T.dot(concated, self.W_enc_to_wattn)) + self.b_enc_to_wattn
            e = T.dot(concated, self.V_enc_to_wattn).flatten(ndim=3)
        else:
            raise ValueError(
                'Attention type must be either "dot", "concat" or "general"')

        # express q1 by q2
        max_attn = T.max(e, axis=1, keepdims=True)
        attn_masked = q2_hs_mask.dimshuffle(0, 1, 'x') * T.exp(e - max_attn)
        attn_vec = q1_hs_mask.dimshuffle(
            0, 'x', 1) * (attn_masked / attn_masked.sum(axis=1, keepdims=True))
        attented_express_q1 = T.batched_dot(
            attn_vec.dimshuffle(0, 2, 1), q2_hs)

        # express q2 by q1
        max_attn = T.max(e, axis=2, keepdims=True)
        attn_masked = q1_hs_mask.dimshuffle(0, 'x', 1) * T.exp(e - max_attn)
        attn_vec = q2_hs_mask.dimshuffle(
            0, 1, 'x') * (attn_masked / attn_masked.sum(axis=2, keepdims=True))
        attented_express_q2 = T.batched_dot(attn_vec, q1_hs)

        return T.tanh(e), attented_express_q1, attented_express_q2

    def __diff(self, vec1, vec2, train):
        concated = T.concatenate((vec1, vec2), axis=1)
        diff1 = lasagne.nonlinearities.rectify(T.dot(concated, self.W_diff) + self.b_diff)
        if train:
            retain_prob = 1 - self.drop_rate
            diff1 /= retain_prob
            mask = self._srng.binomial(diff1.shape, p=retain_prob, dtype=diff1.dtype)
            diff1 = diff1 * mask
        diff1 = T.dot(diff1, self.W_diff2) + self.b_diff2
        # diff1 += T.batched_dot(T.dot(vec1.dimshuffle(0, 'x', 1), self.W_tensor_diff),
        # vec2.dimshuffle(0, 1, 'x')).reshape((self.batch_size, 1))
        diff1 = T.tanh(diff1)
        # return [diff1]
        diff2 = T.tanh((vec1 * vec2).sum(axis=-1, keepdims=True))
        return [diff1, diff2]

    def _build_predictor(self, en_out_q1_syms, en_out_q2_syms, en_q1_mask_sym, en_q2_mask_sym,
                         diff_q1_mask_sym, diff_q2_mask_sym, leak_feaes=None, train=True):
        if train:
            logger.debug(">>> Predictor <<<")
        en_q1 = np.zeros((self.batch_size, self.max_seq_len,
                          self.hidden_output_size)).astype("float32")
        en_q2 = np.zeros((self.batch_size, self.max_seq_len,
                          self.hidden_output_size)).astype("float32")
        en_q1_mask = np.ones(
            (self.batch_size, self.max_seq_len)).astype("float32")
        en_q2_mask = np.ones(
            (self.batch_size, self.max_seq_len)).astype("float32")
        eval_list = {}
        eval_list[en_out_q1_syms] = en_q1
        eval_list[en_q1_mask_sym] = en_q1_mask
        # sentence embedding
        if self.hyper_lstm:
            last_en_q1_syms = en_out_q1_syms[:, -1, :]
            last_en_q2_syms = en_out_q2_syms[:, -1, :]
        avg_en_q1_syms = (en_out_q1_syms * en_q1_mask_sym.dimshuffle(0, 1, 'x')
                          ).sum(axis=1) / en_q1_mask_sym.dimshuffle(0, 1, 'x').sum(axis=1)
        avg_en_q2_syms = (en_out_q2_syms * en_q2_mask_sym.dimshuffle(0, 1, 'x')
                          ).sum(axis=1) / en_q2_mask_sym.dimshuffle(0, 1, 'x').sum(axis=1)
        if train:
            logger.debug("Average LSTM hidden of q1: %s",
                         avg_en_q1_syms.eval(eval_list).shape)
        eval_list[en_out_q2_syms] = en_q2
        # attention value
        if self.hyper_lstm:
            att_q1byq2 = self.attn_func(
                last_en_q2_syms, en_out_q1_syms, en_q1_mask_sym)
            att_q2byq1 = self.attn_func(
                last_en_q1_syms, en_out_q2_syms, en_q2_mask_sym)
        else:
            eval_list[en_q2_mask_sym] = en_q2_mask
            att_q1byq2 = self.attn_func(
                avg_en_q2_syms, en_out_q1_syms, en_q1_mask_sym)
            att_q2byq1 = self.attn_func(
                avg_en_q1_syms, en_out_q2_syms, en_q2_mask_sym)
        if train:
            logger.debug("Attention on the q1: %s",
                         att_q1byq2.eval(eval_list).shape)
        atted_q1 = T.batched_dot(att_q1byq2, en_out_q1_syms)
        atted_q2 = T.batched_dot(att_q2byq1, en_out_q2_syms)
        if train:
            logger.debug("Attend q1: %s", atted_q1.eval(eval_list).shape)
        # hidden of max attened
        indexes = T.arange(att_q1byq2.shape[1]).dimshuffle(
            *(['x' for dim1 in xrange(1)] + [0] + ['x' for dim2 in xrange(att_q1byq2.ndim - 1 - 1)]))
        q1_max_choose = T.argmax(att_q1byq2, axis=1, keepdims=True)
        q1_max_mask = T.eq(indexes, q1_max_choose)
        max_atted_hid_q1 = en_out_q1_syms[q1_max_mask.nonzero()]
        q2_max_choose = T.argmax(att_q2byq1, axis=1, keepdims=True)
        q2_max_mask = T.eq(indexes, q2_max_choose)
        max_atted_hid_q2 = en_out_q1_syms[q2_max_mask.nonzero()]
        if train:
            logger.debug("Max attend q1: %s",
                         max_atted_hid_q1.eval(eval_list).shape)
        # hidden diffs
        diff_hid_q1 = (en_out_q1_syms *
                       diff_q1_mask_sym.dimshuffle(0, 1, 'x')).sum(axis=1)
        diff_hid_q2 = (en_out_q2_syms *
                       diff_q2_mask_sym.dimshuffle(0, 1, 'x')).sum(axis=1)
        if train:
            eval_list = {}
            eval_list[en_out_q1_syms] = en_q1
            eval_list[diff_q1_mask_sym] = np.zeros(
                (self.batch_size, self.max_seq_len)).astype("float32")
            logger.debug("Diffs of hidden: %s",
                         diff_hid_q1.eval(eval_list).shape)
        # diffs
        if train:
            # self.W_tensor_diff = lasagne.utils.create_param(
            # lasagne.init.GlorotUniform(), (self.hidden_output_size,
            # self.hidden_output_size), "W_tensor_diff")
            self.W_diff = lasagne.utils.create_param(
                lasagne.init.GlorotUniform(), (2 * self.hidden_output_size, 50), "W_diff")
            self.b_diff = lasagne.utils.create_param(
                lasagne.init.Constant(0.), (50, ), "b_diff")
            self.W_diff2 = lasagne.utils.create_param(
                lasagne.init.GlorotUniform(), (50, 1), "W_diff2")
            self.b_diff2 = lasagne.utils.create_param(
                lasagne.init.Constant(0.), (1, ), "b_diff2")
            # self.parameters += [self.W_tensor_diff, self.W_diff, self.b_diff]
            self.parameters += [self.W_diff, self.b_diff, self.W_diff2, self.b_diff2]
        diffs = []
        diffs += self.__diff(avg_en_q1_syms, avg_en_q2_syms, train)
        diffs += self.__diff(max_atted_hid_q1, max_atted_hid_q2, train)
        diffs += self.__diff(atted_q1, atted_q2, train)
        diffs += self.__diff(diff_hid_q1, diff_hid_q2, train)
        if self.hyper_lstm:
            diffs += self.__diff(last_en_q1_syms, last_en_q2_syms, train)
            diffs += self.__diff(last_en_q1_syms, atted_q2, train)
            diffs += self.__diff(last_en_q2_syms, atted_q1, train)
            diffs += self.__diff(last_en_q1_syms, max_atted_hid_q2, train)
            diffs += self.__diff(last_en_q2_syms, max_atted_hid_q1, train)
        else:
            diffs += self.__diff(avg_en_q1_syms, atted_q2, train)
            diffs += self.__diff(avg_en_q2_syms, atted_q1, train)
            diffs += self.__diff(avg_en_q1_syms, max_atted_hid_q2, train)
            diffs += self.__diff(avg_en_q2_syms, max_atted_hid_q1, train)
        # concate
        concated = T.concatenate(diffs, axis=1)
        if train:
            concated /= 0.5
            mask = self._srng.binomial(concated.shape, p=0.5, dtype=concated.dtype)
            concated = concated * mask

        # join word level attention
        level2_base_len = 10
        level2_len = level2_base_len
        if self.word_level_att:
            word_diff = self._build_word_att_predictor(
                en_out_q1_syms, en_out_q2_syms, en_q1_mask_sym, en_q2_mask_sym, train)
            level2_len += len(word_diff)

        # combine diffs
        if train:
            self.W_final = lasagne.utils.create_param(
                lasagne.init.GlorotUniform(), (len(diffs), level2_base_len), "W_final")
            self.b_final = lasagne.utils.create_param(
                lasagne.init.Constant(0.), (level2_base_len, ), "b_final")
            if self.leak_num:
                self.W_final_leak = lasagne.utils.create_param(
                    lasagne.init.GlorotUniform(), (self.leak_num, 20), "W_final_leak")
                self.b_final_leak = lasagne.utils.create_param(
                    lasagne.init.Constant(0.), (20, ), "b_final_leak")
                level2_len += 20
                self.parameters += [self.W_final_leak, self.b_final_leak]
            self.W_final2 = lasagne.utils.create_param(
                lasagne.init.GlorotUniform(), (level2_len, 1), "W_final2")
            self.b_final2 = lasagne.utils.create_param(
                lasagne.init.Constant(0.), (1, ), "b_final2")
            self.parameters += [self.W_final,
                                self.b_final, self.W_final2, self.b_final2]

        hidden = lasagne.nonlinearities.rectify(
            T.dot(concated, self.W_final) + self.b_final)
        if self.leak_num:
            leak_input = leak_feaes
            if train:
                leak_input = leak_feaes / 0.5
                mask = self._srng.binomial(leak_input.shape, p=0.5, dtype=leak_input.dtype)
                leak_input = leak_input * mask
            hidden_leak = lasagne.nonlinearities.rectify(
                T.dot(leak_input, self.W_final_leak) + self.b_final_leak)
            if train:
                hidden_leak = hidden_leak / 0.8
                mask = self._srng.binomial(hidden_leak.shape, p=0.8, dtype=hidden_leak.dtype)
                hidden_leak = hidden_leak * mask

        t_features = [hidden]
        if self.word_level_att:
            t_features += word_diff
        if self.leak_num:
            t_features += [hidden_leak]
        if len(t_features) > 1:
            hidden = T.concatenate(t_features, axis=1)

        predict = lasagne.nonlinearities.sigmoid(
            T.dot(hidden, self.W_final2) + self.b_final2)
        predict = T.clip(predict, 1e-7, 1.0 - 1e-7)
        return [predict.flatten(), hidden]

    def _build_word_att_predictor(self, en_out_q1_syms, en_out_q2_syms, en_q1_mask_sym, en_q2_mask_sym, train=True):
        if train:
            logger.debug(">>> Word Level Attention Predictor <<<")
        en_q1 = np.zeros((self.batch_size, self.max_seq_len,
                          self.hidden_output_size)).astype("float32")
        en_q2 = np.zeros((self.batch_size, self.max_seq_len,
                          self.hidden_output_size)).astype("float32")
        en_q1_mask = np.ones(
            (self.batch_size, self.max_seq_len)).astype("float32")
        en_q2_mask = np.ones(
            (self.batch_size, self.max_seq_len)).astype("float32")
        eval_list = {}
        eval_list[en_out_q1_syms] = en_q1
        eval_list[en_out_q2_syms] = en_q2
        # att values
        att_matrix, attented_express_q1, attented_express_q2 = self.word_level_attn_func(
            en_out_q1_syms, en_q1_mask_sym, en_out_q2_syms, en_q2_mask_sym)
        if train:
            retain_prob = 1 - self.watt_drop_rate
            attented_express_q1 /= retain_prob
            mask = self._srng.binomial(attented_express_q1.shape, p=retain_prob, dtype=attented_express_q1.dtype)
            attented_express_q1 = attented_express_q1 * mask
            attented_express_q2 /= retain_prob
            mask = self._srng.binomial(attented_express_q2.shape, p=retain_prob, dtype=attented_express_q2.dtype)
            attented_express_q2 = attented_express_q2 * mask
        if train:
            logger.debug("Word Level Attention Matrix: %s",
                         att_matrix.eval(eval_list).shape)
            eval_list[en_q1_mask_sym] = en_q1_mask
            eval_list[en_q2_mask_sym] = en_q2_mask
            logger.debug("Word Level Attented Q1: %s",
                         attented_express_q1.eval(eval_list).shape)
        # concate q1 and q1 expressed by q2
        concated_q1 = T.concatenate(
            (attented_express_q1, en_out_q1_syms), axis=2)
        concated_q2 = T.concatenate(
            (attented_express_q2, en_out_q2_syms), axis=2)
        if train:
            logger.debug("Word Level Attented Concated: %s",
                         concated_q1.eval(eval_list).shape)
        # arch for lstm predictor
        if train:
            self.l_wp_in = lasagne.layers.InputLayer(
                (self.batch_size, self.max_seq_len, 2 * self.hidden_output_size))
            self.l_wp_mask = lasagne.layers.InputLayer(
                (self.batch_size, self.max_seq_len))
            l_wp_enc = lasagne.layers.LSTMLayer(self.l_wp_in, num_units=self.hidden_size, unroll_scan=False, backwards=False,
                                                gradient_steps=20, name="WAttLSTMLayer", mask_input=self.l_wp_mask, only_return_final=False)
            if self.batch_norm:
                l_wp_enc = lasagne.layers.batch_norm(l_wp_enc)
            l_wp_reshape = lasagne.layers.ReshapeLayer(l_wp_enc, (-1, self.hidden_size))
            l_wp_enc_drop = lasagne.layers.DropoutLayer(l_wp_reshape, p=self.watt_drop_rate)
            l_wp_whid = lasagne.layers.DenseLayer(
                l_wp_enc_drop, num_units=self.hidden_size / 2, nonlinearity=lasagne.nonlinearities.rectify)
            if self.batch_norm:
                l_wp_whid = lasagne.layers.batch_norm(l_wp_whid)
            l_wp_hid_drop = lasagne.layers.DropoutLayer(l_wp_whid, p=self.watt_drop_rate)
            l_wp_pred = lasagne.layers.DenseLayer(
                l_wp_hid_drop, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
            self.l_wp_pred_reshape = lasagne.layers.ReshapeLayer(l_wp_pred, (-1, self.max_seq_len))
            params = lasagne.layers.get_all_params(self.l_wp_pred_reshape, trainable=True)
            self.parameters += params
        # input for lstm predictor
        input_list_q1 = {}
        input_list_q1[self.l_wp_in] = concated_q1
        input_list_q1[self.l_wp_mask] = en_q1_mask_sym
        input_list_q2 = {}
        input_list_q2[self.l_wp_in] = concated_q2
        input_list_q2[self.l_wp_mask] = en_q2_mask_sym
        q1_pred = lasagne.layers.get_output(
            self.l_wp_pred_reshape, inputs=input_list_q1, deterministic=not train)
        q1_pred = (q1_pred * en_q1_mask_sym).sum(axis=1, keepdims=True) / en_q1_mask_sym.sum(axis=1, keepdims=True)
        q2_pred = lasagne.layers.get_output(
            self.l_wp_pred_reshape, inputs=input_list_q2, deterministic=not train)
        q2_pred = (q2_pred * en_q2_mask_sym).sum(axis=1, keepdims=True) / en_q2_mask_sym.sum(axis=1, keepdims=True)
        if train:
            en_concat_q1 = np.zeros((self.batch_size, self.max_seq_len,
                                     2 * self.hidden_output_size)).astype("float32")
            eval_lstm_list = {}
            eval_lstm_list[concated_q1] = en_concat_q1
            eval_lstm_list[en_q1_mask_sym] = en_q1_mask
            logger.debug("Word Level Preds on Q1: %s",
                         q1_pred.eval(eval_lstm_list).shape)
        # arch for cnn
        if train:
            self.l_wattp_in = lasagne.layers.InputLayer(
                (self.batch_size, 1, self.max_seq_len, self.max_seq_len))
            l_wattp_cnn1 = lasagne.layers.Conv2DLayer(
                self.l_wattp_in, 3, (3, 3), pad='valid')
            if self.batch_norm:
                l_wattp_cnn1 = lasagne.layers.batch_norm(l_wattp_cnn1)
            l_wattp_cnn1_drop = lasagne.layers.DropoutLayer(
                l_wattp_cnn1, p=self.watt_drop_rate)
            l_wattp_pool1 = lasagne.layers.MaxPool2DLayer(
                l_wattp_cnn1_drop, (3, 3))
            l_wattp_cnn2 = lasagne.layers.Conv2DLayer(
                l_wattp_pool1, 3, (3, 3), pad='valid')
            if self.batch_norm:
                l_wattp_cnn2 = lasagne.layers.batch_norm(l_wattp_cnn2)
            l_wattp_cnn2_drop = lasagne.layers.DropoutLayer(
                l_wattp_cnn2, p=self.watt_drop_rate)
            l_wattp_pool2 = lasagne.layers.MaxPool2DLayer(
                l_wattp_cnn2_drop, (3, 3))
            l_wattp_flatten = lasagne.layers.FlattenLayer(l_wattp_pool2)
            l_wattp_hid = lasagne.layers.DenseLayer(
                l_wattp_flatten, 10, nonlinearity=lasagne.nonlinearities.rectify)
            if self.batch_norm:
                l_wattp_hid = lasagne.layers.batch_norm(l_wattp_hid)
            l_wattp_hid_drop = lasagne.layers.DropoutLayer(
                l_wattp_hid, p=self.watt_drop_rate)
            self.l_wattp_pred = lasagne.layers.DenseLayer(
                l_wattp_hid_drop, 1, nonlinearity=lasagne.nonlinearities.tanh)
            params = lasagne.layers.get_all_params(
                self.l_wattp_pred, trainable=True)
            self.parameters += params
        # input for lstm predictor
        att_matrix = att_matrix.dimshuffle(0, 'x', 1, 2)
        input_list_cnn = {}
        input_list_cnn[self.l_wattp_in] = att_matrix
        cnn_pred = lasagne.layers.get_output(
            self.l_wattp_pred, inputs=input_list_cnn, deterministic=not train)
        if train:
            en_att_matrix = np.zeros((self.batch_size, 1, self.max_seq_len,
                                      self.max_seq_len)).astype("float32")
            eval_cnn_list = {}
            eval_cnn_list[att_matrix] = en_att_matrix
            logger.debug("Word Level Preds by CNN: %s",
                         cnn_pred.eval(eval_cnn_list).shape)
        return [q1_pred, q2_pred, cnn_pred]
        # return [cnn_pred]

    def compute_loss(self, o, y_sym, weights_sym, train=True):
        if train:
            logger.info("Trainable Parameters")
            logger.info("-" * 40)
            for param in self.parameters:
                logger.info("%s %s", param, param.get_value().shape)
            logger.info("-" * 40)

        loss_sym = (T.nnet.binary_crossentropy(o, y_sym) * weights_sym).mean()
        reged_loss_sym = loss_sym
        if self.reg_type and train:
            reged_loss_sym = loss_sym + self.reg_rate * self.reg

        # accuracy function
        probas = T.concatenate(
            [(1 - o).reshape((-1, 1)), o.reshape((-1, 1))], axis=1)
        pred_sym = T.argmax(probas, axis=1)
        acc_sym = T.mean(T.eq(pred_sym, y_sym))
        return reged_loss_sym, loss_sym, acc_sym, pred_sym, probas

    def build(self):
        """General build function."""
        logger.info("Dropout: %.3f", self.drop_rate)
        logger.info("Reg: %.4f + %s", self.reg_rate, self.reg_type)
        logger.info("Bidirection: %s", self.backlstm)
        en_in_syms, en_out_syms_train, en_out_syms_test, en_para_syms = self._build_encoder()
        self.parameters += en_para_syms
        diff_q1_mask_sym = T.fmatrix()
        diff_q2_mask_sym = T.fmatrix()
        en_in_syms += [diff_q1_mask_sym, diff_q2_mask_sym]
        leak_feaes = None
        if self.leak_num:
            leak_feaes = T.fmatrix()
            en_in_syms += [leak_feaes]
        os = self._build_predictor(
            en_out_syms_train[0], en_out_syms_train[1], en_in_syms[1], en_in_syms[3], diff_q1_mask_sym, diff_q2_mask_sym, train=True, leak_feaes=leak_feaes)
        os_test = self._build_predictor(
            en_out_syms_test[0], en_out_syms_test[1], en_in_syms[1], en_in_syms[3], diff_q1_mask_sym, diff_q2_mask_sym, train=False, leak_feaes=leak_feaes)

        o = os[0]
        o_test = os_test[0]
        all_pred_test = os_test[1]

        y_sym = T.fvector()
        in_syms = en_in_syms + [y_sym]

        weights_sym = T.fvector()
        in_syms += [weights_sym]

        train_reged_loss_sym, train_loss_sym, train_acc_sym, train_pred_sym, train_probas = self.compute_loss(
            o, y_sym, weights_sym, train=True)
        _, test_loss_sym, test_acc_sym, test_pred_sym, test_probas = self.compute_loss(
            o_test, y_sym, weights_sym, train=False)

        logger.info('compile params update function ... ... ')
        all_grads = [T.clip(g, -3, 3)
                     for g in T.grad(train_reged_loss_sym, self.parameters)]
        all_grads = lasagne.updates.total_norm_constraint(all_grads, 3)

        # Compile Theano functions.
        updates = lasagne.updates.adam(
            all_grads, self.parameters, learning_rate=self.lr)
        self.train_func = theano.function(
            in_syms, [train_loss_sym, train_acc_sym], updates=updates, on_unused_input='warn')
        # since we don't have any stochasticity in the network we will just use the training graph without any updates given
        # Note that we still apply deterministic=True in test,
        # which is a defect in current implementation structure.
        self.test_func = theano.function(
            in_syms, [test_loss_sym, test_acc_sym, test_pred_sym], on_unused_input='warn')
        self.predict_func = theano.function(
            en_in_syms, [test_pred_sym, test_probas, all_pred_test], on_unused_input='warn')

    def _load_pretrain_embedding_weights(self, vocab, pretrain_dict):
        logger.debug('load pretrain embeddings')
        weights = self.l_emb.get_params()[0].get_value()
        loaded = 0
        for i in xrange(0, vocab.size):
            word = vocab.index2word[i]
            if word in pretrain_dict:
                weights[i] = np.array(pretrain_dict[word])
                loaded += 1
        self.l_emb.get_params()[0].set_value(weights)
        logger.debug('total %d loaded among %d', loaded, weights.shape[0])
# end JasonNetwork


def build_inputs(vocab, max_len, q1, q2, y, fast_debug=False):
    debug_total = 1060
    if large_debug_cases:
        debug_total *= 140
    logger.info('Loading and format inputs')
    q1 = [vocab(q) for q in q1[:(debug_total if fast_debug else len(q1))]]
    q2 = [vocab(q) for q in q2[:(debug_total if fast_debug else len(q2))]]
    # print q1[0]
    # print q2[0]
    x_q1 = sequence.pad_sequences(q1, maxlen=max_len, value=0, truncating='post', padding='pre')
    x_q2 = sequence.pad_sequences(q2, maxlen=max_len, value=0, truncating='post', padding='pre')
    x_q1 = np.array(x_q1, dtype='int32')
    x_q2 = np.array(x_q2, dtype='int32')
    x_mask_q1 = 1 - np.equal(x_q1, 0)
    x_mask_q2 = 1 - np.equal(x_q2, 0)
    t_x_mask_q1 = x_mask_q1 / x_mask_q1.sum(axis=1, keepdims=True)
    assert not np.any(np.isnan(t_x_mask_q1))
    t_x_mask_q2 = x_mask_q2 / x_mask_q2.sum(axis=1, keepdims=True)
    assert not np.any(np.isnan(t_x_mask_q2))
    # diff in q1
    x_mask_diff_q1 = np.copy(x_q1).astype('float32')
    for i in xrange(x_q1.shape[0]):
        for j in xrange(x_q1.shape[1]):
            if x_q1[i, j] != 0:
                if x_q1[i, j] in q2[i]:
                    x_mask_diff_q1[i, j] = 0
                else:
                    x_mask_diff_q1[i, j] = 1
    # print x_mask_diff_q1[0, :]
    x_mask_diff_q1 = x_mask_diff_q1 / x_mask_diff_q1.sum(axis=1, keepdims=True)
    x_mask_diff_q1 = np.nan_to_num(x_mask_diff_q1)
    assert not np.any(np.isnan(x_mask_diff_q1))
    # print x_mask_diff_q1[0, :]
    # diff in q2
    x_mask_diff_q2 = np.copy(x_q2).astype('float32')
    for i in xrange(x_q2.shape[0]):
        for j in xrange(x_q2.shape[1]):
            if x_q2[i, j] != 0:
                if x_q2[i, j] in q1[i]:
                    x_mask_diff_q2[i, j] = 1
                else:
                    x_mask_diff_q2[i, j] = 0
    x_mask_diff_q2 = x_mask_diff_q2 / x_mask_diff_q2.sum(axis=1, keepdims=True)
    x_mask_diff_q2 = np.nan_to_num(x_mask_diff_q2)
    assert not np.any(np.isnan(x_mask_diff_q2))
    if leak_f:
        with open(leak_f, 'r') as fp:
            x_leak = pickle.load(fp)
            if fast_debug:
                x_leak = x_leak[:debug_total]
    weights = y
    if y:
        y = np.array(y[:(debug_total if fast_debug else len(y))], dtype='float32')
        weights = np.array([class_weight[label] for label in y[
                           :(debug_total if fast_debug else len(y))]], dtype='float32')
    if leak_f:
        return [x_q1, x_mask_q1.astype('float32'), x_q2, x_mask_q2.astype('float32'), x_mask_diff_q1, x_mask_diff_q2, x_leak.astype('float32'), y, weights]
    return [x_q1, x_mask_q1.astype('float32'), x_q2, x_mask_q2.astype('float32'), x_mask_diff_q1, x_mask_diff_q2, y, weights]


def _make_batches(size, batch_size):
    # num_batches = int(np.ceil(size / float(batch_size)))
    num_batches = size / batch_size
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]


def _slice_arrays(arrays, ids):
    # for array in arrays:
    #     print array[ids][0]
    return [array[ids] for array in arrays]


def _balancer(arrays, indexes):
    true_counter, counter = 0, 0
    true_indexes = []
    for i in indexes:
        if arrays[-2][i] == 1:
            true_counter += 1
            true_indexes.append(i)
        counter += 1
    logger.info('training data contain %d cases, with %d positive',
                counter, true_counter)
    logger.info('balancing')
    double_cases = counter - 2 * true_counter
    choosed_indexes = np.random.choice(true_indexes, size=double_cases)
    logger.info('oversample %d, and final cases %d',
                double_cases, counter + double_cases)
    return np.append(indexes, choosed_indexes)


def train_model(model, data, iterations, batch_size, save_path, balance=False, cv=-1):
    assert not (balance and cv > 0)
    logger.info('Strat data prepare for training...')
    sys.stdout.flush()
    logger.info('with data shapes:')
    for d in data:
        logger.info(d.shape)
    if cv < 0:
        num_of_data = len(data[0])
        index_array = np.arange(num_of_data)
        np.random.shuffle(index_array)
        index_array_train = index_array[:int(num_of_data * 0.9)]
        if balance:
            index_array_train = _balancer(data, index_array_train)
        index_array_val = index_array[int(num_of_data * 0.9):]
    else:
        index_array_train, index_array_val = get_idxes_of_cv(data[-2], cv)
        np.random.shuffle(index_array_train)
    num_of_train = len(index_array_train)
    batches_val = _make_batches(len(index_array_val), batch_size)
    logger.info('Building model...')
    model.build()
    logger.info('Start training...')
    best_val = 1000
    best_val_epoch = -1
    best_val_past = 0
    for iteration in xrange(iterations):
        logger.info('iteration %d', iteration)
        samples_processed = 0
        time_flag = 0
        start_time = time.time()
        np.random.shuffle(index_array_train)
        batches = _make_batches(num_of_train, batch_size)
        avg_acc = 0.
        last_counts = 0
        last_ten_acc = 0.
        last_ten_loss = 0.
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            in_start_time = time.time()
            batch_ids = index_array_train[batch_start:batch_end]
            ins_batch = _slice_arrays(data, batch_ids)
            batch_loss, batch_acc = model.train_func(*ins_batch)
            avg_acc += batch_acc
            last_counts += 1
            last_ten_acc += batch_acc
            last_ten_loss += batch_loss
            samples_processed += len(batch_ids)
            if batch_index and (batch_index * batch_size / 100) % 100 == 0:
                # 11 if batch_index == 10 else 10
                logger.debug("%d/%d time_%.3f loss_%.3f acc_%.3f avgacc_%.3f lastacc_%.3f lastloss_%.3f", samples_processed, num_of_train,
                             time.time() - in_start_time, batch_loss, batch_acc, avg_acc / (batch_index + 1),
                             last_ten_acc / last_counts, last_ten_loss / last_counts)
                last_ten_acc = 0.
                last_ten_loss = 0.
                last_counts = 0
            elif batch_index % 10 == 0:
                logger.debug("%d/%d time_%.3f loss_%.3f acc_%.3f avgacc_%.3f", samples_processed, num_of_train,
                             time.time() - in_start_time, batch_loss, batch_acc, avg_acc / (batch_index + 1))
            sys.stdout.flush()
            time_minus = time.time() - start_time
            # for saving model
            new_time_flag = int(time_minus / float(300))
            if new_time_flag > time_flag:
                time_flag = new_time_flag
                model.save_model('%s.%d' % (save_path, iteration))
        valid_loss, valid_acc = [], []
        for batch_index, (batch_start, batch_end) in enumerate(batches_val):
            batch_ids = index_array_val[batch_start:batch_end]
            ins_batch = _slice_arrays(data, batch_ids)
            batch_loss, batch_acc, _ = model.test_func(*ins_batch)
            valid_loss.append(batch_loss)
            valid_acc.append(batch_acc)
        mean_loss = sum(valid_loss) / len(valid_loss)
        mean_acc = sum(valid_acc) / len(valid_acc)
        logger.info('Valid result: loss_%.3f, acc_%.3f', mean_loss, mean_acc)
        sys.stdout.flush()
        model.save_model('%s.%d' % (save_path, iteration))
        # early stop
        if best_val > mean_loss:
            best_val = mean_loss
            best_val_epoch = iteration
            best_val_past = 0
        else:
            best_val_past += 1
        if best_val_past >= 2:
            break
    logger.info('%.3f %.3f', best_val_epoch, best_val)
    if cv >= 0:
        logger.info('Start predict validation...')
        probas = predict(model, data[:-2], best_val_epoch, batch_size, save_path, build=False, indexes=index_array_val)
        preds = probas[:, 1]
        with open(save_path.replace('./model', './cv').replace('.model', '.val%d' % cv), 'w') as fo:
            pickle.dump(preds, fo)
        logger.info('Start predict test...')
        data, vocab, max_len = load_data(False)
        probas = predict(model, data, best_val_epoch, batch_size, save_path, build=False)
        preds = probas[:, 1]
        with open(save_path.replace('./model', './cv').replace('.model', '.test%d' % cv), 'w') as fo:
            pickle.dump(preds, fo)


def predict(model, data, iteration, batch_size, save_path, build=True, indexes=None):
    num_of_data = len(data[0])
    if indexes is not None:
        index_array = indexes
        batches_val = _make_batches(len(indexes), batch_size)
    else:
        index_array = np.arange(num_of_data)
        batches_val = _make_batches(len(index_array), batch_size)
    if build:
        logger.info('Building model...')
        model.build()
    logger.info('Load model...')
    model.load_model('%s.%d' % (save_path, iteration))
    logger.info('Start predict...')
    probas = np.zeros((len(index_array), 2), dtype=np.float32)
    for batch_index, (batch_start, batch_end) in enumerate(batches_val):
        batch_ids = index_array[batch_start:batch_end]
        ins_batch = _slice_arrays(data, batch_ids)
        pred, prob, _ = model.predict_func(*ins_batch)
        probas[batch_start:batch_end] = prob
    if batch_end < len(index_array):
        batch_ids = index_array[len(index_array) - batch_size:len(index_array)]
        ins_batch = _slice_arrays(data, batch_ids)
        pred, prob, _ = model.predict_func(*ins_batch)
        left = len(index_array) - batch_end
        probas[batch_end:len(index_array)] = prob[-left:]
    return probas


def stem_str(sen, snowball_stemmer):
    tt = sen.lower()
    tt = text.re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", tt)
    tt = text.re.sub(r"what's", "what is ", tt)
    tt = text.re.sub(r"\'s", " ", tt)
    tt = text.re.sub(r"\'ve", " have ", tt)
    tt = text.re.sub(r"can't", "cannot ", tt)
    tt = text.re.sub(r"n't", " not ", tt)
    tt = text.re.sub(r"i'm", "i am ", tt)
    tt = text.re.sub(r"\'re", " are ", tt)
    tt = text.re.sub(r"\'d", " would ", tt)
    tt = text.re.sub(r"\'ll", " will ", tt)
    tt = text.re.sub(r",", " ", tt)
    tt = text.re.sub(r"\.", " ", tt)
    tt = text.re.sub(r"!", " ! ", tt)
    tt = text.re.sub(r"\/", " ", tt)
    tt = text.re.sub(r"\^", " ^ ", tt)
    tt = text.re.sub(r"\+", " + ", tt)
    tt = text.re.sub(r"\-", " - ", tt)
    tt = text.re.sub(r"\=", " = ", tt)
    tt = text.re.sub(r"'", " ", tt)
    tt = text.re.sub(r"(\d+)(k)", r"\g<1>000", tt)
    tt = text.re.sub(r":", " : ", tt)
    tt = text.re.sub(r" e g ", " eg ", tt)
    tt = text.re.sub(r" b g ", " bg ", tt)
    tt = text.re.sub(r" u s ", " american ", tt)
    tt = text.re.sub(r"\0s", "0", tt)
    tt = text.re.sub(r" 9 11 ", "911", tt)
    tt = text.re.sub(r"e - mail", "email", tt)
    tt = text.re.sub(r"j k", "jk", tt)
    tt = text.re.sub(r"\s{2,}", " ", tt)
    # sen = nltk.word_tokenize(tt)
    sen = tt.split()
    sen = map(snowball_stemmer.stem, sen)
    # sen = map(wordnet_lemmatizer.lemmatize, sen)
    return ' '.join(sen)


path = "../data/"


def load_data(is_train):
    logger.info('Loading data')
    if is_train:
        train = pd.read_csv(path + "train.csv")
        logger.info('Clean and format data')
        train['q1_clean'] = train['question1'].astype(str).apply(
            lambda x: stem_str(x, snowball_stemmer))
        train['q2_clean'] = train['question2'].astype(str).apply(
            lambda x: stem_str(x, snowball_stemmer))
        train_q1 = train['q1_clean'].tolist()
        train_q2 = train['q2_clean'].tolist()
        logger.info('Building voacb and mapping data')
        sentences = train_q1 + train_q2
        vocab, max_len, all_words = buildVocab(sentences, thes=1)
        logger.info('max length %d', max_len)
        logger.info('%d words before cut', all_words)
        max_len = min(35, max_len)
        logger.info('max length cut to %d' % max_len)
        vocab.save_vocab('./vocab.txt')
        data = build_inputs(vocab, max_len, train_q1, train_q2,
            train.is_duplicate.apply(lambda x: int(x)).values.tolist(), fast_debug)
    else:
        test = pd.read_csv(path + "test.csv")
        test['is_duplicated'] = [-1] * test.shape[0]
        vocab = Vocab()
        vocab.load_vocab('./vocab.txt')
        max_len = 35
        logger.info('Clean and format data')
        test['q1_clean'] = test['question1'].astype(str).apply(
            lambda x: stem_str(x, snowball_stemmer))
        test['q2_clean'] = test['question2'].astype(str).apply(
            lambda x: stem_str(x, snowball_stemmer))
        test_q1 = test['q1_clean'].tolist()
        test_q2 = test['q2_clean'].tolist()
        data = build_inputs(vocab, max_len, test_q1, test_q2, None)[:-2]
    return data, vocab, max_len


data, vocab, max_len = load_data(is_train)

# pretrain_dict_file = None
pretrain_dict_file = path + 'glove.6B.%dd.txt' % embedding_size

pretrain_dict = load_pretrain_emb(pretrain_dict_file)


if cv > 0:
    cvs = range(cv)
else:
    cvs = [-1]

for _cv in cvs:
    if _cv >= 0:
        logger = logging.getLogger(__name__)
        logPath = './cv/log'
        logFile = 'tmp_drop%d_watt%d_b%d_l%d_cv%d' % (
            int(drop_rate * 100), (1 if word_level_att else 0), (1 if balance else 0), (1 if leak_f else 0), cv)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, logFile))
        logger.addHandler(fileHandler)
    model = JasonNetwork(batch_size, max_len, embedding_size, hidden_size, vocab, pretrain_dict=pretrain_dict, backlstm=True, batch_norm=batch_norm, hyper_lstm=hyper_lstm,
                         lr=lr, reg_type=reg_type, reg_rate=reg_rate, drop_rate=drop_rate, watt_drop_rate=watt_drop_rate,
                         attn_type=attn_type, word_level_att=word_level_att, leak_num=leak_num, fast_debug=fast_debug)

    if is_train:
        train_model(model, data, 50, batch_size,
                    './%smodel/jasonnetwork.bs%d.h%d.att_%s.drop%d.watt%d.b%d.l%d%s.model' % (
                        '' if cv < 0 else 'cv/',
                        batch_size, hidden_size, attn_type, int(drop_rate * 100),
                        1 if word_level_att else 0, 1 if balance else 0, 1 if leak_f else 0,
                        '' if cv < 0 else ('.' + str(_cv))),
                    balance=balance, cv=_cv)
    else:
        probas = predict(model, data, 4, batch_size,
                         './model/jasonnetwork.bs%d.h%d.att_%s.drop%d.watt%d.b%d.l%d.match.model' % (batch_size, hidden_size, attn_type, int(drop_rate * 100), 1 if word_level_att else 0, 1 if balance else 0, 1 if leak_f else 0))
        with open('./submit/jasonnetwork.bs%d.h%d.att_%s.csv' % (batch_size, hidden_size, attn_type), 'w') as fo:
            fo.write('test_id,is_duplicate\n')
            for i, prob in enumerate(probas):
                fo.write('%d,%f\n' % (i, prob[1]))

# 459386 cases
#
# no drop out
# epoch 0, valid result 0.535, 0.739
# epoch 1, valid result 0.509, 0.761
# epoch 2, valid result 0.492, 0.781
# epoch 3, valid result 0.497, 0.787
