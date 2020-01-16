# for test:
import os
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.elmo import Elmo
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from language_prediction.dataset_readers import TextExpDataSetReader
from allennlp.nn.regularizers import RegularizerApplicator, L1Regularizer, L2Regularizer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, Seq2VecEncoder
from allennlp.modules.matrix_attention import BilinearMatrixAttention, DotProductMatrixAttention
import tempural_analysis.utils
import torch.optim as optim
import language_prediction.models as models
import language_prediction.baselines as baselines
from torch.nn.modules.activation import ReLU, LeakyReLU
from allennlp.training.metrics import *
from datetime import datetime
from allennlp.data.vocabulary import Vocabulary
import torch.nn as nn
from allennlp.modules import FeedForward
from tempural_analysis import *
import logging
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier, LogisticRegression,\
    PassiveAggressiveRegressor, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from language_prediction.keras_models import LinearKerasModel, ConvolutionalKerasModel
from sklearn.svm import SVC, SVR
import joblib
import copy
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from tempural_analysis.predict_last_decision import PredictLastDecision


# define directories
base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
data_directory = os.path.join(base_directory, 'data', condition)

train_data_file_path = os.path.join(data_directory, 'train_data_1_10_single_round_label.pkl')  # test_text_num_data
validation_data_file_path = os.path.join(data_directory, 'validation_data_1_10_single_round_label.pkl')

batch_size = 9 if '9' in train_data_file_path else 10

# test:
# train_data_file_path = os.path.join(data_directory, 'test_code_data.csv')
# validation_data_file_path = os.path.join(data_directory, 'test_code_data.csv')


def train_valid_base_text_model(model_name):
    """

    :param model_name: the full model name to use
    :return:
    """
    token_indexer = {"tokens": ELMoTokenCharactersIndexer()}

    def tokenizer(x: str):
        return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

    reader = TextExpDataSetReader(token_indexers=token_indexer, tokenizer=tokenizer, add_numeric_data=False)
    train_instances = reader.read(train_data_file_path)
    validation_instances = reader.read(validation_data_file_path)
    vocab = Vocabulary()

    # TODO: change this if necessary
    # batch_size should be: 10 or 9 depends on the input
    # and not shuffle so all the data of the same pair will be in the same batch
    iterator = BasicIterator(batch_size=batch_size)  # , instances_per_epoch=10)
    #  sorting_keys=[('sequence_review', 'list_num_tokens')])
    iterator.index_with(vocab)

    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/' \
                   'elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/' \
                  'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

    # TODO: check the output of this
    # elmo_embedder = Elmo(options_file, weight_file, num_output_representations=2)
    # word_embeddings = elmo_embedder
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    review_attention_layer = models.AttentionSoftMaxLayer(BilinearMatrixAttention(word_embeddings.get_output_dim(),
                                                                                  word_embeddings.get_output_dim()))
    seq_attention_layer = models.AttentionSoftMaxLayer(DotProductMatrixAttention())

    feed_forward = FeedForward(input_dim=batch_size, num_layers=2, hidden_dims=[batch_size, 1], activations=ReLU(),
                               dropout=[0.2, 0.0])
    fc_review_rep = FeedForward(input_dim=124, num_layers=1, hidden_dims=[10], activations=ReLU())

    criterion = nn.MSELoss()

    metrics_dict = {
        'mean_absolute_error': MeanAbsoluteError(),
    }

    model = models.BasicTextModel(word_embedding=word_embeddings,
                                  review_representation_layer=review_attention_layer,
                                  seq_representation_layer=seq_attention_layer,
                                  vocab=vocab,
                                  criterion=criterion,
                                  metrics_dict=metrics_dict,
                                  classifier_feedforward=feed_forward,
                                  fc_review_rep=fc_review_rep
                                  )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    num_epochs = 2

    run_log_directory = utils.set_folder(
        datetime.now().strftime(f'{model_name}_{num_epochs}_epochs_%d_%m_%Y_%H_%M_%S'), 'logs')

    if not os.path.exists(run_log_directory):
        os.makedirs(run_log_directory)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_instances,
        validation_dataset=validation_instances,
        num_epochs=num_epochs,
        shuffle=False,
        serialization_dir=run_log_directory,
        patience=10,
        histogram_interval=10,
    )

    model_dict = trainer.train()

    print(f'{model_name}: evaluation measures are:')
    for key, value in model_dict.items():
        print(f'{key}: {value}')


def train_valid_base_text_model_single_round_label(model_name: str, max_seq_len):
    """
    For single label with nn.BCEWithLogitsLoss()
    :param max_seq_len: if use seq so 10, if not, so 1
    :param model_name: the full model name to use
    :return:
    """
    token_indexer = {"tokens": ELMoTokenCharactersIndexer()}

    def tokenizer(x: str):
        return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

    reader = TextExpDataSetReader(token_indexers=token_indexer, tokenizer=tokenizer, add_numeric_data=False,
                                  max_seq_len=max_seq_len)
    train_instances = reader.read(train_data_file_path)
    validation_instances = reader.read(validation_data_file_path)
    vocab = Vocabulary()

    # TODO: change this if necessary
    # batch_size should be: 10 or 9 depends on the input
    # and not shuffle so all the data of the same pair will be in the same batch
    iterator = BasicIterator(batch_size=batch_size)  # , instances_per_epoch=10)
    #  sorting_keys=[('sequence_review', 'list_num_tokens')])
    iterator.index_with(vocab)

    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/' \
                   'elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/' \
                  'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

    # TODO: check the output of this
    # elmo_embedder = Elmo(options_file, weight_file, num_output_representations=2)
    # word_embeddings = elmo_embedder
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    review_attention_layer = models.AttentionSoftMaxLayer(BilinearMatrixAttention(word_embeddings.get_output_dim(),
                                                                                  word_embeddings.get_output_dim()))
    seq_attention_layer = models.AttentionSoftMaxLayer(DotProductMatrixAttention())
    fc_review_rep = FeedForward(input_dim=124, num_layers=1, hidden_dims=[10], activations=ReLU())

    feed_forward = FeedForward(input_dim=batch_size, num_layers=2, hidden_dims=[batch_size, 2], activations=ReLU(),
                               dropout=[0.2, 0.0])
    criterion = nn.BCEWithLogitsLoss()

    metrics_dict = {
        'accuracy': CategoricalAccuracy(),
        'auc': Auc(),
        'F1measure': F1Measure(positive_label=1),
    }

    model = models.BasicTextModel(word_embedding=word_embeddings,
                                  review_representation_layer=review_attention_layer,
                                  seq_representation_layer=seq_attention_layer,
                                  vocab=vocab,
                                  criterion=criterion,
                                  metrics_dict=metrics_dict,
                                  classifier_feedforward=feed_forward,
                                  fc_review_rep=fc_review_rep
                                  )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    num_epochs = 2

    run_log_directory = utils.set_folder(
        datetime.now().strftime(f'{model_name}_{num_epochs}_epochs_%d_%m_%Y_%H_%M_%S'), 'logs')

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_instances,
        validation_dataset=validation_instances,
        num_epochs=num_epochs,
        shuffle=False,
        serialization_dir=run_log_directory,
        patience=10,
        histogram_interval=10,
    )

    model_dict = trainer.train()

    print(f'{model_name}: evaluation measures are:')
    for key, value in model_dict.items():
        print(f'{key}: {value}')

    model.predictions.to_csv(os.path.join(run_log_directory, 'predictions.csv'))


def train_valid_base_text_decision_results_ep_model(model_name: str, single_round_label: bool, use_only_prev_round: bool,
                                                    train_data_file_name: str, validation_data_file_name: str,
                                                    no_history: bool=False):
    """
    This function train and validate model that use texts and numbers.
    :param: model_name: the full model name
    :param single_round_label: the label to use: single round of total payoff
    :param use_only_prev_round: if to use all the history or only the previous round
    :param train_data_file_name: the name of the train_data to use
    :param validation_data_file_name: the name of the validation_data to use
    :param no_history: if we don't want to use any history data
    :return:
    """
    token_indexer = {"tokens": ELMoTokenCharactersIndexer()}

    def tokenizer(x: str):
        return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

    reader = TextExpDataSetReader(token_indexers=token_indexer, tokenizer=tokenizer, add_numeric_data=True,
                                  use_only_prev_round=use_only_prev_round, single_round_label=single_round_label,
                                  three_losses=True, no_history=no_history)
    train_data_file_inner_path = os.path.join(data_directory, train_data_file_name)
    validation_data_file_inner_path = os.path.join(data_directory, validation_data_file_name)
    train_instances = reader.read(train_data_file_inner_path)
    validation_instances = reader.read(validation_data_file_inner_path)
    vocab = Vocabulary()

    # TODO: change this if necessary
    # batch_size should be: 10 or 9 depends on the input
    # and not shuffle so all the data of the same pair will be in the same batch
    iterator = BasicIterator(batch_size=9)  # , instances_per_epoch=10)
    #  sorting_keys=[('sequence_review', 'list_num_tokens')])
    iterator.index_with(vocab)

    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/' \
                   'elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/' \
                  'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

    # TODO: check the output of this
    # elmo_embedder = Elmo(options_file, weight_file, num_output_representations=2)
    # word_embeddings = elmo_embedder
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    review_attention_layer =\
        models.AttentionSoftMaxLayer(BilinearMatrixAttention(word_embeddings.get_output_dim(), word_embeddings.get_output_dim()))
    seq_attention_layer = models.AttentionSoftMaxLayer(DotProductMatrixAttention())
    fc_review_rep_output_dim = reader.max_tokens_len
    fc_review_rep = FeedForward(input_dim=reader.max_tokens_len, num_layers=1, hidden_dims=[fc_review_rep_output_dim],
                                activations=ReLU())
    # seq_attention_layer = FeedForward(input_dim=)

    # numbers_lstm: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(2, 10, bidirectional=True, batch_first=True))
    # the shape of the flatten data rep
    feed_forward_input_dim = reader.max_seq_len*(fc_review_rep_output_dim + reader.number_length)
    feed_forward_classification = FeedForward(input_dim=feed_forward_input_dim, num_layers=1, hidden_dims=[2],
                                              activations=ReLU(), dropout=[0.0])
    feed_forward_regression = FeedForward(input_dim=feed_forward_input_dim, num_layers=1, hidden_dims=[1],
                                          activations=ReLU(), dropout=[0.0])
    criterion_classification = nn.BCEWithLogitsLoss()
    criterion_regression = nn.MSELoss()

    metrics_dict = {
        "accuracy": CategoricalAccuracy(),
        # 'auc': Auc(),
        # 'F1measure': F1Measure(positive_label=1),
    }

    model = models.BasicTextDecisionResultModel(
        word_embedding=word_embeddings,
        review_representation_layer=review_attention_layer,
        seq_representation_layer=seq_attention_layer,
        vocab=vocab,
        classifier_feedforward_classification=feed_forward_classification,
        classifier_feedforward_regression=feed_forward_regression,
        fc_review_rep=fc_review_rep,
        criterion_classification=criterion_classification,
        criterion_regression=criterion_regression,
        metrics_dict=metrics_dict,
        add_numbers=True,
        max_tokens_len=reader.max_tokens_len,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    num_epochs = 2

    run_log_directory = utils.set_folder(
        datetime.now().strftime(f'{model_name}_{num_epochs}_epochs_%d_%m_%Y_%H_%M_%S'), 'logs')

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_instances,
        validation_dataset=validation_instances,
        num_epochs=num_epochs,
        shuffle=False,
        serialization_dir=run_log_directory,
        patience=10,
        histogram_interval=10,
    )

    model_dict = trainer.train()

    print(f'{model_name}: evaluation measures are:')
    for key, value in model_dict.items():
        if 'accuracy' in key:
            value = value*100
        print(f'{key}: {value}')

    # save the model predictions
    model.predictions.to_csv(os.path.join(run_log_directory, 'predictions.csv'))


def train_valid_base_text_decision_results_ep_fix_text_features_model(
        model_name: str, single_round_label: bool, use_only_prev_round: bool, train_data_file_name: str,
        validation_data_file_name: str, no_history: bool = False, func_batch_size: int = 9,
        numbers_columns: list = None, add_numeric_data: bool=True):
    """
    This function train and validate model that use fix texts features and numbers.
    :param: model_name: the full model name
    :param single_round_label: the label to use: single round of total payoff
    :param use_only_prev_round: if to use all the history or only the previous round
    :param train_data_file_name: the name of the train_data to use
    :param validation_data_file_name: the name of the validation_data to use
    :param no_history: if we don't want to use any history data
    :param func_batch_size: the batch size to use
    :param model_name: the name of the model we run
    :param numbers_columns: the names of the columns to use for the numeric data
    :param add_numeric_data: if we want to add numbers data
    :return:
    """

    reader = TextExpDataSetReader(add_numeric_data=add_numeric_data, use_only_prev_round=use_only_prev_round,
                                  single_round_label=single_round_label, three_losses=True, fix_text_features=True,
                                  no_history=no_history, numbers_columns_name=numbers_columns)
    train_data_file_inner_path = os.path.join(data_directory, train_data_file_name)
    validation_data_file_inner_path = os.path.join(data_directory, validation_data_file_name)
    train_instances = reader.read(train_data_file_inner_path)
    validation_instances = reader.read(validation_data_file_inner_path)
    vocab = Vocabulary()

    # TODO: change this if necessary
    # batch_size should be: 10 or 9 depends on the input
    # and not shuffle so all the data of the same pair will be in the same batch
    iterator = BasicIterator(batch_size=func_batch_size)  # , instances_per_epoch=10)
    #  sorting_keys=[('sequence_review', 'list_num_tokens')])
    iterator.index_with(vocab)

    # the shape of the flatten data rep
    if 'bert' in train_data_file_name:  # fix features are BERT vector
        text_feedtorward = FeedForward(input_dim=reader.max_tokens_len, num_layers=2, hidden_dims=[300, 50],
                                       activations=ReLU(), dropout=[0.0, 0.0])
        reader.max_tokens_len = 50
    else:
        text_feedtorward = None
    feed_forward_input_dim = reader.max_seq_len*(reader.max_tokens_len + reader.number_length)
    feed_forward_classification = FeedForward(input_dim=feed_forward_input_dim, num_layers=2, hidden_dims=[10, 2],
                                              activations=LeakyReLU(),
                                              dropout=[0.3, 0.3])
    feed_forward_regression = FeedForward(input_dim=feed_forward_input_dim, num_layers=2, hidden_dims=[10, 1],
                                          activations=LeakyReLU(),
                                          dropout=[0.3, 0.3])
    criterion_classification = nn.BCEWithLogitsLoss()
    criterion_regression = nn.MSELoss()

    metrics_dict = {
        'Accuracy': CategoricalAccuracy()  # BooleanAccuracy(),
        # 'auc': Auc(),
        # 'F1measure': F1Measure(positive_label=1),
    }

    model = models.BasicFixTextFeaturesDecisionResultModel(
        vocab=vocab,
        classifier_feedforward_classification=feed_forward_classification,
        classifier_feedforward_regression=feed_forward_regression,
        criterion_classification=criterion_classification,
        criterion_regression=criterion_regression,
        metrics_dict=metrics_dict,
        add_numbers=True,
        max_tokens_len=reader.max_tokens_len,
        text_feedforward=text_feedtorward,
        # regularizer=RegularizerApplicator([("", L1Regularizer(0.2))]),
    )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    num_epochs = 40

    run_log_directory = utils.set_folder(
        datetime.now().strftime(f'{model_name}_{num_epochs}_epochs_%d_%m_%Y_%H_%M_%S'), 'logs')

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_instances,
        validation_dataset=validation_instances,
        num_epochs=num_epochs,
        shuffle=False,
        serialization_dir=run_log_directory,
        patience=10,
        histogram_interval=10,
    )

    model_dict = trainer.train()

    print(f'{model_name}: evaluation measures are:')
    for key, value in model_dict.items():
        if 'accuracy' in key:
            value = value*100
        print(f'{key}: {value}')

    # save the model predictions
    model.predictions.to_csv(os.path.join(run_log_directory, 'predictions.csv'))


def train_valid_base_text_decision_fix_text_features_model(
        model_name: str, single_round_label: bool, use_only_prev_round: bool, train_data_file_name: str,
        validation_data_file_name: str, no_history: bool = False, func_batch_size: int = 9,
        numbers_columns: list = None, add_numeric_data: bool=True):
    """
    This function train and validate model that use fix texts features only.
    :param: model_name: the full model name
    :param single_round_label: the label to use: single round of total payoff
    :param use_only_prev_round: if to use all the history or only the previous round
    :param train_data_file_name: the name of the train_data to use
    :param validation_data_file_name: the name of the validation_data to use
    :param no_history: if we don't want to use any history data
    :param func_batch_size: the batch size to use
    :param model_name: the name of the model we run
    :param numbers_columns: the names of the columns to use for the numeric data
    :param add_numeric_data: if we want to add numbers data
    :return:
    """

    reader = TextExpDataSetReader(add_numeric_data=add_numeric_data, use_only_prev_round=use_only_prev_round,
                                  single_round_label=single_round_label, three_losses=True, fix_text_features=True,
                                  no_history=no_history, numbers_columns_name=numbers_columns)
    train_data_file_inner_path = os.path.join(data_directory, train_data_file_name)
    validation_data_file_inner_path = os.path.join(data_directory, validation_data_file_name)
    train_instances = reader.read(train_data_file_inner_path)
    validation_instances = reader.read(validation_data_file_inner_path)
    vocab = Vocabulary()

    # TODO: change this if necessary
    # batch_size should be: 10 or 9 depends on the input
    # and not shuffle so all the data of the same pair will be in the same batch
    iterator = BasicIterator(batch_size=func_batch_size)  # , instances_per_epoch=10)
    #  sorting_keys=[('sequence_review', 'list_num_tokens')])
    iterator.index_with(vocab)

    # the shape of the flatten data rep
    if 'bert' in train_data_file_name:  # fix features are BERT vector
        text_feedtorward = FeedForward(input_dim=reader.max_tokens_len, num_layers=2, hidden_dims=[300, 50],
                                       activations=ReLU(), dropout=[0.0, 0.0])
        reader.max_tokens_len = 50
    else:
        text_feedtorward = None
    feed_forward_input_dim = reader.max_seq_len*(reader.max_tokens_len + reader.number_length)
    feed_forward_classification = FeedForward(input_dim=feed_forward_input_dim, num_layers=1, hidden_dims=[2],
                                              activations=LeakyReLU(),
                                              dropout=[0.3])
    criterion_classification = nn.BCEWithLogitsLoss()

    metrics_dict = {
        'Accuracy': CategoricalAccuracy()  # BooleanAccuracy(),
        # 'auc': Auc(),
        # 'F1measure': F1Measure(positive_label=1),
    }

    model = models.BasicFixTextFeaturesDecisionModel(
        vocab=vocab,
        classifier_feedforward_classification=feed_forward_classification,
        criterion_classification=criterion_classification,
        metrics_dict=metrics_dict,
        max_tokens_len=reader.max_tokens_len,
        text_feedforward=text_feedtorward,
        regularizer=RegularizerApplicator([("", L1Regularizer())]),
    )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    num_epochs = 100

    run_log_directory = utils.set_folder(
        datetime.now().strftime(f'{model_name}_{num_epochs}_epochs_%d_%m_%Y_%H_%M_%S'), 'logs')

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_instances,
        validation_dataset=validation_instances,
        num_epochs=num_epochs,
        shuffle=False,
        serialization_dir=run_log_directory,
        patience=10,
        histogram_interval=10,
    )

    model_dict = trainer.train()

    print(f'{model_name}: evaluation measures are:')
    for key, value in model_dict.items():
        if 'accuracy' in key:
            value = value*100
        print(f'{key}: {value}')

    # save the model predictions
    model.predictions.to_csv(os.path.join(run_log_directory, 'predictions.csv'))


def train_predict_simple_baseline_model(model_name: str, binary_classification: bool=False, use_first_round: bool=True):
    """
    train and predict the simple baseline model
    :param model_name: the name of the model
    :param binary_classification: if this is a binary label or not
    :param use_first_round: if to use that data from the  first round
    :return:
    """
    if binary_classification:
        metric_list = ['accuracy_score', 'f1_score',]  # 'auc']
        label_col_name = 'single_label'
    else:
        metric_list = ['mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error']
        label_col_name = 'label'
    model = getattr(baselines, model_name)(train_data_file_path, validation_data_file_path, binary_classification,
                                           label_col_name, use_first_round)
    model.fit()
    model.predict()
    train_metric_dict, validation_metric_dict = calculate_measures(model.train_data, model.validation_data, metric_list,
                                                                   label_col_name=label_col_name)

    print(f'{model_name} evaluation measures are:\nTrain data:')
    for name, metric_calc in train_metric_dict.items():
        print(f'{name}: {metric_calc*100}')
    print(f'Validation data:')
    for name, metric_calc in validation_metric_dict.items():
        print(f'{name}: {metric_calc*100}')


def train_test_simple_features_model(model_name: str, features_data_file_path: str, backward_search: bool = False,
                                     inner_data_directory: str=data_directory):
    """
    This function train and test some simple ML models that use text manual features to predict decisions
    :param features_data_file_path: hte path to the features file
    :param model_name: the full model name
    :param backward_search: use backward_search to find the best features
    :param inner_data_directory: the data directory to use
    :return:
    """
    experiment_path = utils.set_folder(datetime.now().strftime(f'{model_name}_%d_%m_%Y_%H_%M'), 'logs')
    file_name = datetime.now().strftime('LogFile.log')
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S',
                        filename=os.path.join(experiment_path, file_name),
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    ##################################################################################################

    print(f'Start running train_test_simple_features_model on file: {features_data_file_path}')
    logging.info(f'Start running train_test_simple_features_model on file {features_data_file_path}')

    folds = 5  # if folds = 1, run train-test, else - run cross-validation with relevant number of folds
    # load the features data
    if 'pkl' in features_data_file_path:
        data = joblib.load(os.path.join(inner_data_directory, features_data_file_path))
    else:
        data = pd.read_csv(os.path.join(inner_data_directory, features_data_file_path))

    data.index = data.sample_id
    # The label we want to predict
    label_dict = {
        'label': ['DM chose stay home', 'DM chose hotel'],
        'total_payoff': [],
    }
    label = 'label'
    data = data.drop(['k_size', 'sample_id'], axis=1)
    features = [item for item in data.columns.tolist() if item not in [label, 'pair_id']]
    inner_batch_size = 9 if 'history' in features_data_file_path or 'prev' in features_data_file_path else 10
    print(f'batch size is: {inner_batch_size}')
    candidates_features = copy.deepcopy(features)
    model_dict = {'label':  # classification models:
                  [[
                      PredictLastDecision(),
                      DummyClassifier(strategy='stratified'), DummyClassifier(strategy='most_frequent'),
                      XGBClassifier(max_depth=5), DecisionTreeClassifier(),
                      # LinearKerasModel(input_dim=len(features), batch_size=inner_batch_size),
                      SVC(), LogisticRegression(), Perceptron(), RandomForestClassifier(), SVC(kernel='linear'),
                      # SGDClassifier(), PassiveAggressiveClassifier(),
                      # ConvolutionalKerasModel(num_features=len([feature for feature in features if '_1' in feature]),
                      #                         input_len=len(features))
                      ],
                   'classification'],
                  'total_payoff':  # regression models
                  [[RandomForestRegressor(), SGDRegressor(), PassiveAggressiveRegressor(),
                    SVR(), SVR(kernel='linear'),
                    DecisionTreeRegressor(), XGBRegressor(),
                    # Perceptron(),
                    DummyRegressor(strategy='median'), DummyRegressor(strategy='mean')], 'regression'],
                  }

    model_dict_to_use = {label: model_dict[label][0]}
    label_dict_to_use = {label: label_dict[label]}

    if backward_search:
        execute_evaluate.evaluate_backward_search(data=data, base_features=features, window_size_list=[0],
                                                  label_dict=label_dict_to_use, num_folds=folds,
                                                  model_dict=model_dict_to_use, classifier_results_dir=experiment_path,
                                                  appendix='', personal_features=[], model_type=model_dict[label][1],
                                                  col_to_group='pair_id', candidates_features=candidates_features)
    else:
        execute_evaluate.evaluate_grid_search(data=data, base_features=features, add_feature_list=[''],
                                              window_size_list=[0], label_dict=label_dict_to_use, num_folds=folds,
                                              model_dict=model_dict_to_use, classifier_results_dir=experiment_path,
                                              appendix='', personal_features=[], model_type=model_dict[label][1],
                                              col_to_group='pair_id')

    logging.info('Done!')
