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




logger = logging.getLogger(__name__)



class BaseTfModel():
    def __init__(self, config_dict):
        config_dict = deepcopy(config_dict)
        mode = config_dict.pop("mode")
        self.mode = mode
        self.word_vocab_size = config_dict.pop("word_vocab_size")
        self.word_embedding_dim = config_dict.pop("word_embedding_dim")
        self.word_embedding_matrix = config_dict.pop("word_embedding_matrix", None)
        self.fine_tune_embeddings = config_dict.pop("fine_tune_embeddings")
        self.rnn_hidden_size = config_dict.pop("rnn_hidden_size")
        self.share_encoder_weights = config_dict.pop("share_encoder_weights")
        self.rnn_output_mode = config_dict.pop("rnn_output_mode")
        self.output_keep_prob = config_dict.pop("output_keep_prob")
        self.global_step = tf.get_variable(name="global_step",
                                           shape=[],
                                           dtype='int32',
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)
    def _create_placeholders(self):
        """
        Create the placeholders for use in the model.
        """
        # Define the inputs here
        # Shape: (batch_size, num_sentence_words)
        # The first input sentence.
        self.sentence_one = tf.placeholder("int32",
                                           [None, None],
                                           name="sentence_one")

        # Shape: (batch_size, num_sentence_words)
        # The second input sentence.
        self.sentence_two = tf.placeholder("int32",
                                           [None, None],
                                           name="sentence_two")

        # Shape: (batch_size, 2)
        # The true labels, encoded as a one-hot vector. So
        # [1, 0] indicates not duplicate, [0, 1] indicates duplicate.
        self.y_true = tf.placeholder("int32",
                                     [None, 2],
                                     name="true_labels")

        # A boolean that encodes whether we are training or evaluating
        self.is_train = tf.placeholder('bool', [], name='is_train')

    def _get_train_feed_dict(self, batch):
        inputs, targets = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.y_true: targets,
                     self.is_train: True}
        return feed_dict

    def _get_validation_feed_dict(self, batch):
        inputs, targets = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.y_true: targets,
                     self.is_train: False}
        return feed_dict

    def _get_test_feed_dict(self, batch):
        inputs = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.is_train: False}
        return feed_dict

    def _l1_similarity(self, sentence_one, sentence_two):
        """
        Given a pair of encoded sentences (vectors), return a probability
        distribution on whether they are duplicates are not with:
        exp(-||sentence_one - sentence_two||)

        Parameters
        ----------
        sentence_one: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded sentence_ones to use in the probability calculation.

        sentence_one: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded sentence_twos to use in the probability calculation.

        Returns
        -------
        class_probabilities: Tensor
            A tensor of shape (batch_size, 2), represnting the probability
            that a pair of sentences are duplicates as
            [is_not_duplicate, is_duplicate].
        """
        with tf.name_scope("l1_similarity"):
            # Take the L1 norm of the two vectors.
            # Shape: (batch_size, 2*rnn_hidden_size)
            l1_distance = tf.abs(sentence_one - sentence_two)

            # Take the sum for each sentence pair
            # Shape: (batch_size, 1)
            summed_l1_distance = tf.reduce_sum(l1_distance, axis=1,
                                               keep_dims=True)

            # Exponentiate the negative summed L1 distance to get the
            # positive-class probability.
            # Shape: (batch_size, 1)
            positive_class_probs = tf.exp(-summed_l1_distance)

            # Get the negative class probabilities by subtracting
            # the positive class probabilities from 1.
            # Shape: (batch_size, 1)
            negative_class_probs = 1 - positive_class_probs

            # Concatenate the positive and negative class probabilities
            # Shape: (batch_size, 2)
            class_probabilities = tf.concat([negative_class_probs,
                                             positive_class_probs], 1)

            # if class_probabilities has 0's, then taking the log of it
            # (e.g. for cross-entropy loss) will cause NaNs. So we add
            # epsilon and renormalize by the sum of the vector.
            safe_class_probabilities = class_probabilities + 1e-08
            safe_class_probabilities /= tf.reduce_sum(safe_class_probabilities,
                                                      axis=1,
                                                      keep_dims=True)
            return safe_class_probabilities



    def train(self,
              dataManager,
              train_samples,
              val_samples,
              batch_size, num_train_steps_per_epoch, num_epochs,
              num_val_steps,save_path, log_path,
              log_period=10, val_period=200,save_period=250,
              max_ckpts_to_keep=10, patience=0,is_load=False,model_load_dir=''):

            global_step = 0
            init_op = tf.global_variables_initializer()

            gpu_options = tf.GPUOptions(allow_growth=True)
            sess_config = tf.ConfigProto(gpu_options=gpu_options)
            with tf.Session(config=sess_config) as sess:

                # if is_load:
                #     saver = tf.train.Saver()
                #     logger.info("Getting latest checkpoint in {}".format(model_load_dir))
                #     last_checkpoint = tf.train.latest_checkpoint(model_load_dir)
                #     logger.info("Attempting to load checkpoint at {}".format(last_checkpoint))
                #     saver.restore(sess, last_checkpoint)
                #     logger.info("Successfully loaded {}!".format(last_checkpoint))
                #
                # else:
                sess.run(init_op)
                # Set up a Saver for periodically serializing the model.
                saver = tf.train.Saver(max_to_keep=max_ckpts_to_keep)

                # Set up the classes for logging to Tensorboard.
                train_writer = tf.summary.FileWriter(log_path + "/train",
                                                     sess.graph)
                val_writer = tf.summary.FileWriter(log_path + "/val",
                                                   sess.graph)

                epoch_validation_losses = []
                # Iterate over a generator that returns batches.
                for epoch in tqdm(range(num_epochs), desc="Epochs Completed"):
                    # Get a generator of train batches
                    train_batch_gen = dataManager.get_next_batch
                    # Iterate over the generated batches
                    for it in tqdm(np.arange(num_train_steps_per_epoch)):

                        global_step = sess.run(self.global_step) + 1

                        q1,q2,targets = train_batch_gen(train_samples,batch_index=it)
                        inputs= []
                        inputs.append(q1)
                        inputs.append(q2)
                        train_batch = (inputs,targets)
                        feed_dict = self._get_train_feed_dict(train_batch)

                        # Do a gradient update, and log results to Tensorboard
                        # if necessary.
                        if global_step % log_period == 0:
                            # Record summary with gradient update
                            train_loss, _, train_summary = sess.run(
                                [self.loss, self.training_op, self.summary_op],
                                feed_dict=feed_dict)
                            train_writer.add_summary(train_summary, global_step)
                        else:
                            # Do a gradient update without recording anything.
                            train_loss, _ = sess.run(
                                [self.loss, self.training_op],
                                feed_dict=feed_dict)

                        #val_period
                        if global_step % val_period == 0:
                            # Evaluate on validation data
                            val_acc, val_loss, val_summary = self._evaluate_on_validation(
                                dataManager,val_samples=val_samples,
                                batch_size=batch_size,
                                num_val_steps=num_val_steps,
                                session=sess)
                            val_writer.add_summary(val_summary, global_step)
                        # Write a model checkpoint if necessary.
                        if global_step % save_period == 0:
                            saver.save(sess, save_path, global_step=global_step)



                    # End of the epoch, so save the model and check validation loss,
                    # stopping if applicable.
                    saver.save(sess, save_path, global_step=global_step)
                    val_acc, val_loss, val_summary = self._evaluate_on_validation(
                        dataManager, val_samples=val_samples,
                        batch_size=batch_size,
                        num_val_steps=num_val_steps,
                        session=sess)

                    val_writer.add_summary(val_summary, global_step)

                    epoch_validation_losses.append(val_loss)

                    # Get the lowest validation loss, with regards to the patience
                    # threshold.
                    patience_val_losses = epoch_validation_losses[:-(patience + 1)]
                    if patience_val_losses:
                        min_patience_val_loss = min(patience_val_losses)
                    else:
                        min_patience_val_loss = math.inf
                    if min_patience_val_loss <= val_loss:
                        # past loss was lower, so stop
                        logger.info("Validation loss of {} in last {} "
                                    "epochs, which is lower than current "
                                    "epoch validation loss of {}; stopping "
                                    "early.".format(min_patience_val_loss,
                                                    patience,
                                                    val_loss))
                        break

            # Done training!
            logger.info("Finished {} epochs!".format(epoch + 1))


    def _evaluate_on_validation(self, dataManager,val_samples,
                                batch_size,
                                num_val_steps,
                                session):
        val_batch_gen = dataManager.get_next_batch
        # Calculate the mean of the validation metrics
        # over the validation set.
        val_accuracies = []
        val_losses = []


        for it in tqdm(np.arange(num_val_steps)):

            q1, q2, targets = val_batch_gen(val_samples,batch_size=batch_size,batch_index=it)
            inputs = []
            inputs.append(q1)
            inputs.append(q2)
            val_batch = (inputs, targets)

            feed_dict = self._get_validation_feed_dict(val_batch)
            val_batch_acc, val_batch_loss = session.run(
                [self.accuracy, self.loss],
                feed_dict=feed_dict)

            val_accuracies.append(val_batch_acc)
            val_losses.append(val_batch_loss)

        mean_val_accuracy = np.mean(val_accuracies)
        mean_val_loss = np.mean(val_losses)

        # Create a new Summary object with mean_val accuracy
        # and mean_val_loss and add it to Tensorboard.
        val_summary = tf.Summary(value=[
            tf.Summary.Value(tag="val_summaries/loss",
                             simple_value=mean_val_loss),
            tf.Summary.Value(tag="val_summaries/accuracy",
                             simple_value=mean_val_accuracy)])
        return mean_val_accuracy, mean_val_loss, val_summary


    def predict(self, dataManager,test_samples, model_load_dir, batch_size,
                num_test_steps=None):

        if num_test_steps is None:
            logger.info("num_test_steps is not set, pass in a value "
                        "to show a progress bar.")

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            logger.info("Getting latest checkpoint in {}".format(model_load_dir))
            last_checkpoint = tf.train.latest_checkpoint(model_load_dir)
            logger.info("Attempting to load checkpoint at {}".format(last_checkpoint))
            saver.restore(sess, last_checkpoint)
            logger.info("Successfully loaded {}!".format(last_checkpoint))

            # Get a generator of test batches
            test_batch_gen = dataManager.get_test_next_batch

            y_pred = []
            for it in tqdm(np.arange(num_test_steps)):
                q1, q2 = test_batch_gen(test_samples, batch_size=batch_size, batch_index=it)
                inputs = []
                inputs.append(q1)
                inputs.append(q2)
                test_batch = inputs
                feed_dict = self._get_test_feed_dict(test_batch)
                y_pred_batch = sess.run(self.y_pred, feed_dict=feed_dict)
                y_pred.append(y_pred_batch)

            y_pred_flat = np.concatenate(y_pred, axis=0)
        return y_pred_flat
