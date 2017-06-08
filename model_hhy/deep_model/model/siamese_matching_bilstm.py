from copy import deepcopy
import logging
from overrides import overrides
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tqdm import tqdm
import math
import numpy as np

from util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from util.pooling import mean_pool
from util.rnn import last_relevant_output
from basetfmodel import BaseTfModel


logger = logging.getLogger(__name__)


class SiameseMatchingBiLSTM(BaseTfModel):

    def __init__(self, config_dict):
        super(SiameseMatchingBiLSTM, self).__init__(config_dict)


    def _build_forward(self):
        """
        Using the config passed to the SiameseMatchingBiLSTM object on
        creation, build the forward pass of the computation graph.
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
        sentence_one_len = tf.reduce_sum(sentence_one_mask, 1)
        sentence_two_len = tf.reduce_sum(sentence_two_mask, 1)

        word_vocab_size = self.word_vocab_size
        word_embedding_dim = self.word_embedding_dim
        word_embedding_matrix = self.word_embedding_matrix
        fine_tune_embeddings = self.fine_tune_embeddings

        with tf.variable_scope("embeddings"):
            with tf.variable_scope("embedding_var"), tf.device("/cpu:0"):
                if self.mode == "train":
                    # Load the word embedding matrix from the config,
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
            # first and second sentence LSTM outputs by mean pooling
            #mean pooling for the second level
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

        # diff and dot concat
        with tf.name_scope("match_sentences"):
            sentence_difference = encoded_sentence_one - encoded_sentence_two
            sentence_product = encoded_sentence_one * encoded_sentence_two
            # Shape: (batch_size, 4 * 2*rnn_hidden_size)
            matching_vector = tf.concat([encoded_sentence_one, sentence_product,
                                         sentence_difference, encoded_sentence_two], 1)
        # Nonlinear projection to 2 dimensional class probabilities
        with tf.variable_scope("project_matching_vector"):
            # Shape: (batch_size, 2)
            projection = tf.layers.dense(matching_vector, 2, tf.nn.relu,
                                         name="matching_vector_projection")

        with tf.name_scope("loss"):
            # Get the predicted class probabilities
            # Shape: (batch_size, 2)
            self.y_pred = tf.nn.softmax(projection, name="softmax_probabilities")
            # Use softmax_cross_entropy_with_logits to calculate xentropy.
            # It's unideal to do the softmax twice, but I prefer the numerical
            # stability of the tf function.
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true,
                                                        logits=projection,
                                                        name="xentropy_loss"))

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

