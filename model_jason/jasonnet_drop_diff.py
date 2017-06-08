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
from utils.vocab_char import pad_sequences_char
from utils.DropLSTMLayer import DropLSTMLayer

__authors__ = ['bowenwu']


is_train = True
fast_debug = False
balance = False
large_debug_cases = False

cv = -1
t_cv = 2

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
# word_level_att = None
word_level_att = 'general'

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
    fileHandler = logging.FileHandler("{0}/{1}.diff.log".format(logPath, logFile))
    logger.addHandler(fileHandler)
if cv >= 0:
    logger = logging.getLogger(__name__)
    logPath = './cv/log'
    logFile = 'tmp_drop%d_watt%d_b%d_l%d_cv%d' % (
        int(drop_rate * 100), (1 if word_level_att else 0), (1 if balance else 0), (1 if leak_f else 0), t_cv)
    fileHandler = logging.FileHandler("{0}/{1}.diff.log".format(logPath, logFile))
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

    def _build_encoder(self, tensors=None):
        logger.debug(">>> Encoder <<<")

        if tensors:
            en_x_q1_sym, en_xmask_q1_sym, en_x_q2_sym, en_xmask_q2_sym = tensors
            # en_x_q1_sym, en_xmask_q1_sym, en_x_q2_sym, en_xmask_q2_sym, en_x_q1_char_sym, en_x_q2_char_sym = tensors
        else:
            en_x_q1_sym = T.imatrix()
            en_xmask_q1_sym = T.fmatrix()
            en_x_q2_sym = T.imatrix()
            en_xmask_q2_sym = T.fmatrix()
            # en_x_q1_char_sym = T.itensor3()
            # en_x_q2_char_sym = T.itensor3()

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
        l_emb_drop = lasagne.layers.DropoutLayer(self.l_emb, p=self.drop_rate)

        # en_x_char = np.random.randint(0, 10, size=(
        #     self.batch_size, self.max_seq_len, 20)).astype("int32")
        # filters = [2, 3, 5]
        # l_en_x_char = lasagne.layers.InputLayer((self.batch_size, self.max_seq_len, 20))
        # l_en_x_char_reshape = lasagne.layers.ReshapeLayer(l_en_x_char, (self.batch_size * self.max_seq_len, 20))
        # l_en_emb_char = lasagne.layers.EmbeddingLayer(
        #     l_en_x_char_reshape, 63, 40, name="EnCharEmbeddingLayer")
        # l_en_emb_char_t = lasagne.layers.DimshuffleLayer(
        #     l_en_emb_char, (0, 'x', 1, 2), name="EnCharBeforeCNNLayer")
        # l_pool = []
        # for i, filter_size in enumerate(filters):
        #     l_cnn_temp = lasagne.layers.Conv2DLayer(l_en_emb_char_t, 2, filter_size, pad='valid', name="CharCNN%dLayer" % filter_size)
        #     l_pool_temp = lasagne.layers.MaxPool2DLayer(l_cnn_temp, (20 - filter_size + 1, 1), name="CharPool%dLayer" % filter_size)
        #     l_flatten_temp = lasagne.layers.FlattenLayer(l_pool_temp, 2, name="CharFlatten%dLayer" % filter_size)
        #     l_pool.append(l_flatten_temp)
        # l_concat_cnn = lasagne.layers.ConcatLayer(l_pool, axis=1)
        # l_concat_cnn_reshape = lasagne.layers.ReshapeLayer(l_concat_cnn, (self.batch_size, self.max_seq_len, -1))
        # l_emb_new = lasagne.layers.ConcatLayer([l_en_emb, l_concat_cnn_reshape], axis=2)
        # l_emb_drop = lasagne.layers.DropoutLayer(l_emb_new, p=self.drop_rate)

        l_en_enc = DropLSTMLayer(l_emb_drop, num_units=self.hidden_size, unroll_scan=False, backwards=False,
            gradient_steps=-1, name="EnLSTMLayer", mask_input=l_en_mask, only_return_final=False, inter_drop=0.05)
        if self.backlstm:
            l_en_enc_back = DropLSTMLayer(l_emb_drop, num_units=self.hidden_size, unroll_scan=False, backwards=True,
                gradient_steps=-1, name="EnLSTMLayerBack", mask_input=l_en_mask, only_return_final=False, inter_drop=0.05)
            l_en_enc = lasagne.layers.ConcatLayer(
                [l_en_enc, l_en_enc_back], axis=2)
            if self.hyper_lstm:
                if self.batch_norm:
                    l_en_enc = lasagne.layers.batch_norm(l_en_enc)
                l_en_enc = DropLSTMLayer(l_en_enc, num_units=self.hidden_size, unroll_scan=False, backwards=False,
                    gradient_steps=-1, name="EnLSTMLayerHyper", mask_input=l_en_mask, only_return_final=False, inter_drop=0.05)
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

        # en_enc_q1_sym = lasagne.layers.get_output(
        #     l_en_enc_drop, inputs={l_en_x: en_x_q1_sym, l_en_mask: en_xmask_q1_sym, l_en_x_char: en_x_q1_char_sym}, deterministic=False)
        # en_enc_q2_sym = lasagne.layers.get_output(
        #     l_en_enc_drop, inputs={l_en_x: en_x_q2_sym, l_en_mask: en_xmask_q2_sym, l_en_x_char: en_x_q2_char_sym}, deterministic=False)
        # logger.debug("Encoder Layer: %s", en_enc_q1_sym.eval(
        #     {en_x_q1_sym: en_x, en_xmask_q1_sym: en_xmask, en_x_q1_char_sym: en_x_char}).shape)

        # en_enc_q1_sym_test = lasagne.layers.get_output(
        #     l_en_enc_drop, inputs={l_en_x: en_x_q1_sym, l_en_mask: en_xmask_q1_sym, l_en_x_char: en_x_q1_char_sym}, deterministic=True)
        # en_enc_q2_sym_test = lasagne.layers.get_output(
        #     l_en_enc_drop, inputs={l_en_x: en_x_q2_sym, l_en_mask: en_xmask_q2_sym, l_en_x_char: en_x_q2_char_sym}, deterministic=True)

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
        # return [en_x_q1_sym, en_xmask_q1_sym, en_x_q2_sym, en_xmask_q2_sym, en_x_q1_char_sym, en_x_q2_char_sym], en_out_syms_train, en_out_syms_test, encoder_parameters

    def word_level_attn_func(self, q1_hs, q1_hs_mask, q2_hs, q2_hs_mask, q1_diff_mask, q2_diff_mask):
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

        # express q1 by q2
        max_attn = T.max(e, axis=1, keepdims=True)
        attn_masked = q2_hs_mask.dimshuffle(0, 1, 'x') * T.exp(e - max_attn)
        attn_vec = q1_hs_mask.dimshuffle(
            0, 'x', 1) * (attn_masked / attn_masked.sum(axis=1, keepdims=True))
        attented_express_q1 = T.batched_dot(
            attn_vec.dimshuffle(0, 2, 1), q2_hs * q2_diff_mask.dimshuffle(0, 1, 'x'))

        # express q2 by q1
        max_attn = T.max(e, axis=2, keepdims=True)
        attn_masked = q1_hs_mask.dimshuffle(0, 'x', 1) * T.exp(e - max_attn)
        attn_vec = q2_hs_mask.dimshuffle(
            0, 1, 'x') * (attn_masked / attn_masked.sum(axis=2, keepdims=True))
        attented_express_q2 = T.batched_dot(attn_vec, q1_hs * q1_diff_mask.dimshuffle(0, 1, 'x'))

        return attented_express_q1, attented_express_q2

    def _build_predictor(self, en_out_q1_syms, en_out_q2_syms, en_q1_mask_sym, en_q2_mask_sym,
                         diff_q1_mask_sym, diff_q2_mask_sym, leak_feaes=None, train=True):
        preds = self._build_word_att_predictor(en_out_q1_syms, en_out_q2_syms, en_q1_mask_sym, en_q2_mask_sym,
                         diff_q1_mask_sym, diff_q2_mask_sym, train=train)

        # combine diffs
        if train:
            self.W_final = lasagne.utils.create_param(
                lasagne.init.GlorotUniform(), (len(preds), 1), "W_final")
            self.b_final = lasagne.utils.create_param(
                lasagne.init.Constant(0.), (1, ), "b_final")
            self.parameters += [self.W_final, self.b_final]

        hidden = T.concatenate(preds, axis=1)

        predict = lasagne.nonlinearities.sigmoid(
            T.dot(hidden, self.W_final) + self.b_final)
        predict = T.clip(predict, 1e-7, 1.0 - 1e-7)
        return [predict.flatten(), hidden]

    def _build_word_att_predictor(self, en_out_q1_syms, en_out_q2_syms, en_q1_mask_sym, en_q2_mask_sym,
        diff_q1_mask_sym, diff_q2_mask_sym, train=True):
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
        attented_express_q1, attented_express_q2 = self.word_level_attn_func(
            en_out_q1_syms, en_q1_mask_sym, en_out_q2_syms, en_q2_mask_sym, diff_q1_mask_sym, diff_q2_mask_sym)
        if train:
            retain_prob = 1 - self.watt_drop_rate
            attented_express_q1 /= retain_prob
            mask = self._srng.binomial(attented_express_q1.shape, p=retain_prob, dtype=attented_express_q1.dtype)
            attented_express_q1 = attented_express_q1 * mask
            attented_express_q2 /= retain_prob
            mask = self._srng.binomial(attented_express_q2.shape, p=retain_prob, dtype=attented_express_q2.dtype)
            attented_express_q2 = attented_express_q2 * mask
        if train:
            eval_list[en_q1_mask_sym] = en_q1_mask
            eval_list[en_q2_mask_sym] = en_q2_mask
            eval_list[diff_q2_mask_sym] = en_q2_mask
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
        input_list_q1[self.l_wp_mask] = diff_q1_mask_sym
        input_list_q2 = {}
        input_list_q2[self.l_wp_in] = concated_q2
        input_list_q2[self.l_wp_mask] = diff_q2_mask_sym
        q1_pred = lasagne.layers.get_output(
            self.l_wp_pred_reshape, inputs=input_list_q1, deterministic=not train)
        q1_pred = (q1_pred * diff_q1_mask_sym)
        q1_max = q1_pred.max(axis=1, keepdims=True)
        q1_min = q1_pred.min(axis=1, keepdims=True)
        q1_pred = q1_pred.mean(axis=1, keepdims=True)
        q2_pred = lasagne.layers.get_output(
            self.l_wp_pred_reshape, inputs=input_list_q2, deterministic=not train)
        q2_pred = (q2_pred * diff_q2_mask_sym)
        q2_max = q2_pred.max(axis=1, keepdims=True)
        q2_min = q2_pred.min(axis=1, keepdims=True)
        q2_pred = q2_pred.mean(axis=1, keepdims=True)
        if train:
            en_concat_q1 = np.zeros((self.batch_size, self.max_seq_len,
                                     2 * self.hidden_output_size)).astype("float32")
            eval_lstm_list = {}
            eval_lstm_list[concated_q1] = en_concat_q1
            eval_lstm_list[diff_q1_mask_sym] = en_q1_mask
            logger.debug("Word Level Preds on Q1: %s",
                         q1_pred.eval(eval_lstm_list).shape)
        return [q1_pred, q1_max, q1_min, q2_pred, q2_max, q2_min]

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
    # x_q1_char = pad_sequences_char(q1[:(debug_total if fast_debug else len(q1))], max_len, 20, dtype='int32')
    # x_q2_char = pad_sequences_char(q2[:(debug_total if fast_debug else len(q2))], max_len, 20, dtype='int32')
    q1 = [vocab(q) for q in q1[:(debug_total if fast_debug else len(q1))]]
    q2 = [vocab(q) for q in q2[:(debug_total if fast_debug else len(q2))]]
    x_q1 = sequence.pad_sequences(q1, maxlen=max_len, value=0, truncating='post', padding='pre')
    x_q2 = sequence.pad_sequences(q2, maxlen=max_len, value=0, truncating='post', padding='pre')
    # print q1[0]
    # print q2[0]
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
    x_mask_diff_q2 = np.copy(x_q2).astype('float32')
    for i in xrange(x_q2.shape[0]):
        for j in xrange(x_q2.shape[1]):
            if x_q2[i, j] != 0:
                if x_q2[i, j] in q1[i]:
                    x_mask_diff_q2[i, j] = 1
                else:
                    x_mask_diff_q2[i, j] = 0
    weights = y
    if y:
        y = np.array(y[:(debug_total if fast_debug else len(y))], dtype='float32')
        weights = np.array([class_weight[label] for label in y[
                           :(debug_total if fast_debug else len(y))]], dtype='float32')
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
    if _cv != t_cv:
        continue
    model = JasonNetwork(batch_size, max_len, embedding_size, hidden_size, vocab, pretrain_dict=pretrain_dict, backlstm=True, batch_norm=batch_norm, hyper_lstm=hyper_lstm,
                         lr=lr, reg_type=reg_type, reg_rate=reg_rate, drop_rate=drop_rate, watt_drop_rate=watt_drop_rate,
                         attn_type=attn_type, word_level_att=word_level_att, leak_num=leak_num, fast_debug=fast_debug)

    if is_train:
        train_model(model, data, 50, batch_size,
                    './%smodel/jasonnetwork.bs%d.h%d.att_%s.drop%d.watt%d.b%d.l%d%s.diff.model' % (
                        '' if cv < 0 else 'cv/',
                        batch_size, hidden_size, attn_type, int(drop_rate * 100),
                        1 if word_level_att else 0, 1 if balance else 0, 1 if leak_f else 0,
                        '' if cv < 0 else ('.' + str(_cv))),
                    balance=balance, cv=_cv)
    else:
        probas = predict(model, data, 4, batch_size,
                         './model/jasonnetwork.bs%d.h%d.att_%s.drop%d.watt%d.b%d.l%d.diff.model' % (batch_size, hidden_size, attn_type, int(drop_rate * 100), 1 if word_level_att else 0, 1 if balance else 0, 1 if leak_f else 0))
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
