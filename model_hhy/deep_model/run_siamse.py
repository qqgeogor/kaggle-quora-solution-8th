import argparse
import sys
import logging
import math
import numpy as np
import os
import pandas as pd
import pickle
import json
from util.data_manager import DataManager
from siamese_bilstm import SiameseBiLSTM
from siamese_matching_bilstm import SiameseMatchingBiLSTM


seed = 1024
np.random.seed(seed)

path = '../data/'


logger = logging.getLogger(__name__)

def main():
    # Parse config arguments
    argparser = argparse.ArgumentParser(
        description=("Run a baseline Siamese BiLSTM model "
                     "for paraphrase identification."))
    argparser.add_argument("--mode", type=str,
                           default='train',
                           help=("One of {train|predict}, to "
                                 "indicate what you want the model to do. "
                                 "If you pick \"predict\", then you must also "
                                 "supply the path to a pretrained model and "
                                 "DataIndexer to load."))

    argparser.add_argument("--model_use", type=str,
                           default='siamse',
                           help=("use the model for trian.{siamse|siamse_match|bimpm}"))

    argparser.add_argument("--model_load_dir", type=str,default='./data/models/siamse_match/0/',
                           help=("The path to a directory with checkpoints to "
                                 "load for evaluation or prediction. The "
                                 "latest checkpoint will be loaded."))
    argparser.add_argument("--dataindexer_load_path", type=str,
                           help=("The path to the dataindexer fit on the "
                                 "train data, so we can properly index the "
                                 "test data for evaluation or prediction."))
    argparser.add_argument("--train_file", type=str,
                           default=os.path.join(path,
                                            "train_final_clean.pkl"),
                           help="Path to a file to train on.")
    argparser.add_argument("--test_file", type=str,
                           default=os.path.join(path,
                                                "test_final_clean.pkl"))
    argparser.add_argument("--batch_size", type=int, default=128,
                           help="Number of instances per batch.")
    argparser.add_argument("--num_epochs", type=int, default=2,
                           help=("Number of epochs to perform in "
                                 "training."))
    argparser.add_argument("--early_stopping_patience", type=int, default=0,
                           help=("number of epochs with no validation "
                                 "accuracy improvement after which training "
                                 "will be stopped"))
    argparser.add_argument("--num_sentence_words", type=int, default=30,
                           help=("The maximum length of a sentence. Longer "
                                 "sentences will be truncated, and shorter "
                                 "ones will be padded."))
    argparser.add_argument("--word_embedding_dim", type=int, default=100,
                           help="Dimensionality of the word embedding layer")
    argparser.add_argument("--pretrained_embeddings_file_path", type=str,
                           help="Path to a file with pretrained embeddings.",
                           default=os.path.join(path,
                                                "glove.6B.100d.txt"))
    argparser.add_argument("--fine_tune_embeddings", action="store_true",
                           help=("Whether to train the embedding layer "
                                 "(if True), or keep it fixed (False)."))
    argparser.add_argument("--rnn_hidden_size", type=int, default=256,
                           help=("The output dimension of the RNN."))
    argparser.add_argument("--share_encoder_weights", action="store_true",
                           help=("Whether to use the same encoder on both "
                                 "input sentences (thus sharing weights), "
                                 "or a different one for each sentence"))
    argparser.add_argument("--rnn_output_mode", type=str, default="last",
                           choices=["mean_pool", "last"],
                           help=("How to calculate the final sentence "
                                 "representation from the RNN outputs. "
                                 "\"mean_pool\" indicates that the outputs "
                                 "will be averaged (with respect to padding), "
                                 "and \"last\" indicates that the last "
                                 "relevant output will be used as the "
                                 "sentence representation."))
    argparser.add_argument("--output_keep_prob", type=float, default=1.0,
                           help=("The proportion of RNN outputs to keep, "
                                 "where the rest are dropped out."))
    argparser.add_argument("--log_period", type=int, default=10,
                           help=("Number of steps between each summary "
                                 "op evaluation."))
    argparser.add_argument("--val_period", type=int, default=250,
                           help=("Number of steps between each evaluation of "
                                 "validation performance."))
    argparser.add_argument("--log_dir", type=str,
                           default=os.path.join(
                                                "./data/logs/"),
                           help=("Directory to save logs to."))
    argparser.add_argument("--save_period", type=int, default=250,
                           help=("Number of steps between each "
                                 "model checkpoint"))
    argparser.add_argument("--save_dir", type=str,
                           default=os.path.join(
                                                "./data/models/"),
                           help=("Directory to save model checkpoints to."))
    argparser.add_argument("--run_id", type=str,default=0,
                           help=("Identifying run ID for this run. If "
                                 "predicting, you probably want this "
                                 "to be the same as the train run_id"))
    argparser.add_argument("--model_name", type=str,default='siamse',
                           help=("Identifying model name for this run. If"
                                 "predicting, you probably want this "
                                 "to be the same as the train run_id"))
    argparser.add_argument("--reweight_predictions_for_kaggle", action="store_true",
                           help=("Only relevant when predicting. Whether to "
                                 "reweight the prediction probabilities to "
                                 "account for class proportion discrepancy "
                                 "between train and test."))

    argparser.add_argument("--is_load",default=False,
                           help=("load has trained model"))

    config = argparser.parse_args()

    model_name = config.model_name
    run_id = config.run_id
    mode = config.mode

    # Get the data.
    data_manager = DataManager()
    data_manager.set_vocab_mode('word')
    batch_size = config.batch_size
    num_sentence_words = config.num_sentence_words
    #data_manager.fit_vocab(config.train_file, config.test_file) not have generate your vocabulary
    word_dict_base = './data/dictionary/'
    word_dict_path = {'word_index':word_dict_base+'word_index.pkl','index_word':word_dict_base+'index_word.pkl',
                      'char_index':word_dict_base+'char_index.pkl','index_char':word_dict_base+'index_char.pkl'}
    data_manager.load_word_dictionary(word_dict=word_dict_path)

    if mode == "train":
        # Read the train data from a file, and use it to index the validation data

        train_samples,val_samples = data_manager.get_train_data_from_file(config.train_file,
                                                        max_lengths=num_sentence_words)
        train_data_size = len(train_samples)
        val_data_size = len(val_samples)

    else:
        test_samples = data_manager.get_test_data_from_file(config.test_file,max_lengths=num_sentence_words)
        test_data_size = len(test_samples)


    vars(config)["word_vocab_size"] = data_manager.get_vocab_size()

    # Log the run parameters.
    log_dir = config.log_dir
    log_path = os.path.join(log_dir, model_name)
    logger.info("Writing logs to {}".format(log_path))
    if not os.path.exists(log_path):
        logger.info("log path {} does not exist, "
                    "creating it".format(log_path))
        os.makedirs(log_path)
    params_path = os.path.join(log_path, mode + "params.json")
    logger.info("Writing params to {}".format(params_path))
    with open(params_path, 'w') as params_file:
        json.dump(vars(config), params_file, indent=4)


    # Get the embeddings.
    embedding_matrix = data_manager.get_embedd_matrix(True)
    vars(config)["word_embedding_matrix"] = embedding_matrix



    # Initialize the model.
    if config.model_use=='siamse':
        model = SiameseBiLSTM(vars(config))
        model._create_placeholders()
        model._build_forward()

    elif config.model_use=='siamse_match':
        model = SiameseMatchingBiLSTM(vars(config))
        model._create_placeholders()
        model._build_forward()

    if mode == "train":
        # Train the model.
        num_epochs = config.num_epochs
        num_train_steps_per_epoch = int(math.ceil(train_data_size / batch_size))
        num_val_steps = int(math.ceil(val_data_size / batch_size))
        log_period = config.log_period
        val_period = config.val_period


        save_period = config.save_period
        save_dir = os.path.join(config.save_dir, model_name + "/"+str(run_id))
        save_path = os.path.join(save_dir, model_name + "-"+str(run_id))

        logger.info("Checkpoints will be written to {}".format(save_dir))
        if not os.path.exists(save_dir):
            logger.info("save path {} does not exist, "
                        "creating it".format(save_dir))
            os.makedirs(save_dir)

        logger.info("Saving fitted DataManager to {}".format(save_dir))
        data_manager_pickle_name = "{}-{}-DataManager.pkl".format(model_name,
                                                                  run_id)
        pickle.dump(data_manager,
                    open(os.path.join(save_dir, data_manager_pickle_name), "wb"))

        patience = config.early_stopping_patience

        if config.is_load:
            model_load_dir = config.model_load_dir
        else:
            model_load_dir=''

        model.train(data_manager,
                    train_samples,
                    val_samples,
                    batch_size=batch_size,
                    num_train_steps_per_epoch=num_train_steps_per_epoch,
                    num_epochs=num_epochs,
                    num_val_steps=num_val_steps,
                    save_path=save_path,
                    log_path=log_path,
                    log_period=log_period,
                    val_period=val_period,
                    save_period=save_period,
                    patience=patience,model_load_dir=model_load_dir,is_load=config.is_load)

    else:
        # Predict with the model
        model_load_dir = config.model_load_dir
        num_test_steps = int(math.ceil(test_data_size / batch_size))
        # Numpy array of shape (num_test_examples, 2)
        raw_predictions = model.predict(data_manager,test_samples,
                                        model_load_dir=model_load_dir,
                                        batch_size=batch_size,
                                        num_test_steps=num_test_steps)
        # Remove the first column, so we're left with just the probabilities
        # that a question is a duplicate.
        is_duplicate_probabilities = np.delete(raw_predictions, 0, 1)


        output_predictions_path = os.path.join(log_path, model_name + "-" +
                                               str(run_id)+
                                               "-output_predictions.csv")
        logger.info("Writing predictions to {}".format(output_predictions_path))
        is_duplicate_df = pd.DataFrame(is_duplicate_probabilities)
        is_duplicate_df.to_csv(output_predictions_path, index_label="test_id",
                               header=["is_duplicate"])



if __name__ == '__main__':
    main()
