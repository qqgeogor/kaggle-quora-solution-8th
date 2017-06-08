from copy import deepcopy
import logging
from overrides import overrides
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tqdm import tqdm
import numpy as np
import math

from util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from util.pooling import mean_pool
from util.rnn import last_relevant_output
from basetfmodel import BaseTfModel

logger = logging.getLogger(__name__)

class SiameseBiLSTM(BaseTfModel):
    def __init__(self, config_dict):
        super(SiameseBiLSTM, self).__init__(config_dict)


    def _build_forward(self):
        """
        Using the values in the config passed to the SiameseBiLSTM object
        on creation, build the forward pass of the computation graph.
        """
        # A mask over the word indices in the sentence, indicating
        # which indices are padding and which are words.
        # Shape: (batch_size, num_sentence_words)
        sentence_one_mask = tf.sign(self.sentence_one,
                                    name="sentence_one_masking")
        sentence_two_mask = tf.sign(self.sentence_two,
                                    name="sentence_two_masking")

        # The unpadded lengths of sentence one and sentence two
        # Shape: (batch_size,)
        sentence_one_len = tf.reduce_sum(sentence_one_mask, 1)# len
        sentence_two_len = tf.reduce_sum(sentence_two_mask, 1)

        word_vocab_size = self.word_vocab_size
        word_embedding_dim = self.word_embedding_dim
        word_embedding_matrix = self.word_embedding_matrix
        fine_tune_embeddings = self.fine_tune_embeddings

        with tf.variable_scope("embeddings"):
            with tf.variable_scope("embedding_var"), tf.device("/cpu:0"):
                if self.mode == "train":
                    # Load the word embedding matrix that was passed in
                    # since we are training
                    word_emb_mat = tf.get_variable(
                        "word_emb_mat",
                        dtype="float",
                        shape=[word_vocab_size,
                               word_embedding_dim],
                        initializer=tf.constant_initializer(
                            word_embedding_matrix),
                        trainable=fine_tune_embeddings)
                else:
                    # We are not training, so a model should have been
                    # loaded with the embedding matrix already there.
                    word_emb_mat = tf.get_variable("word_emb_mat",
                                                   shape=[word_vocab_size,
                                                          word_embedding_dim],
                                                   dtype="float",
                                                   trainable=fine_tune_embeddings)

            with tf.variable_scope("word_embeddings"):
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_one = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_one)
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_two = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_two)

        rnn_hidden_size = self.rnn_hidden_size
        rnn_output_mode = self.rnn_output_mode
        output_keep_prob = self.output_keep_prob
        rnn_cell_fw_one = LSTMCell(rnn_hidden_size, state_is_tuple=True)
        d_rnn_cell_fw_one = SwitchableDropoutWrapper(rnn_cell_fw_one,
                                                     self.is_train,
                                                     output_keep_prob=output_keep_prob)
        rnn_cell_bw_one = LSTMCell(rnn_hidden_size, state_is_tuple=True)
        d_rnn_cell_bw_one = SwitchableDropoutWrapper(rnn_cell_bw_one,
                                                     self.is_train,
                                                     output_keep_prob=output_keep_prob)
        with tf.variable_scope("encode_sentences"):
            # Encode the first sentence.
            (fw_output_one, bw_output_one), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=d_rnn_cell_fw_one,
                cell_bw=d_rnn_cell_bw_one,
                dtype="float",
                sequence_length=sentence_one_len,
                inputs=word_embedded_sentence_one,
                scope="encoded_sentence_one")
            if self.share_encoder_weights:
                # Encode the second sentence, using the same RNN weights.
                tf.get_variable_scope().reuse_variables()
                (fw_output_two, bw_output_two), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=d_rnn_cell_fw_one,
                    cell_bw=d_rnn_cell_bw_one,
                    dtype="float",
                    sequence_length=sentence_two_len,
                    inputs=word_embedded_sentence_two,
                    scope="encoded_sentence_one")
            else:
                # Encode the second sentence with a different RNN
                rnn_cell_fw_two = LSTMCell(rnn_hidden_size, state_is_tuple=True)
                d_rnn_cell_fw_two = SwitchableDropoutWrapper(
                    rnn_cell_fw_two,
                    self.is_train,
                    output_keep_prob=output_keep_prob)
                rnn_cell_bw_two = LSTMCell(rnn_hidden_size, state_is_tuple=True)
                d_rnn_cell_bw_two = SwitchableDropoutWrapper(
                    rnn_cell_bw_two,
                    self.is_train,
                    output_keep_prob=output_keep_prob)
                (fw_output_two, bw_output_two), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=d_rnn_cell_fw_two,
                    cell_bw=d_rnn_cell_bw_two,
                    dtype="float",
                    sequence_length=sentence_two_len,
                    inputs=word_embedded_sentence_two,
                    scope="encoded_sentence_two")

            # Now, combine the fw_output and bw_output for the
            # first and second sentence LSTM outputs
            if rnn_output_mode == "mean_pool":
                # Mean pool the forward and backward RNN outputs
                pooled_fw_output_one = mean_pool(fw_output_one,
                                                 sentence_one_len)
                pooled_bw_output_one = mean_pool(bw_output_one,
                                                 sentence_one_len)
                pooled_fw_output_two = mean_pool(fw_output_two,
                                                 sentence_two_len)
                pooled_bw_output_two = mean_pool(bw_output_two,
                                                 sentence_two_len)
                # Shape: (batch_size, 2*rnn_hidden_size)
                encoded_sentence_one = tf.concat([pooled_fw_output_one,
                                                  pooled_bw_output_one], 1)
                encoded_sentence_two = tf.concat([pooled_fw_output_two,
                                                  pooled_bw_output_two], 1)
            elif rnn_output_mode == "last":
                # Get the last unmasked output from the RNN
                last_fw_output_one = last_relevant_output(fw_output_one,
                                                          sentence_one_len)
                last_bw_output_one = last_relevant_output(bw_output_one,
                                                          sentence_one_len)
                last_fw_output_two = last_relevant_output(fw_output_two,
                                                          sentence_two_len)
                last_bw_output_two = last_relevant_output(bw_output_two,
                                                          sentence_two_len)
                # Shape: (batch_size, 2*rnn_hidden_size)
                encoded_sentence_one = tf.concat([last_fw_output_one,
                                                  last_bw_output_one], 1)
                encoded_sentence_two = tf.concat([last_fw_output_two,
                                                  last_bw_output_two], 1)
            else:
                raise ValueError("Got an unexpected value {} for "
                                 "rnn_output_mode, expected one of "
                                 "[mean_pool, last]")

        with tf.name_scope("loss"):
            # Use the exponential of the negative L1 distance
            # between the two encoded sentences to get an output
            # distribution over labels.
            # Shape: (batch_size, 2)
            self.y_pred = self._l1_similarity(encoded_sentence_one,
                                              encoded_sentence_two)
            # Manually calculating cross-entropy, since we output
            # probabilities and can't use softmax_cross_entropy_with_logits
            # Add epsilon to the probabilities in order to prevent log(0)
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(tf.cast(self.y_true, "float") *
                               tf.log(self.y_pred),
                               axis=1))

        with tf.name_scope("accuracy"):
            # Get the correct predictions.
            # Shape: (batch_size,) of bool
            correct_predictions = tf.equal(
                tf.argmax(self.y_pred, 1),
                tf.argmax(self.y_true, 1))

            # Cast to float, and take the mean to get accuracy
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                   "float"))

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer()
            self.training_op = optimizer.minimize(self.loss,
                                                  global_step=self.global_step)

        with tf.name_scope("train_summaries"):
            # Add the loss and the accuracy to the tensorboard summary
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

