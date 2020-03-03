import torch
import torch.nn as nn
from typing import *
from torch.autograd import Variable

from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules import FeedForward
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import masked_softmax
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from collections import defaultdict


def save_predictions_seq_models(prediction_df: dict, predictions: torch.Tensor, gold_labels: torch.Tensor,
                                metadata, epoch: int, is_train: bool, mask: torch.Tensor) -> dict:
    """
    This function get the predictions class and save with the gold labels for sequence models where each sample had a
    list of predictions and labels
    :param prediction_df: dataframe with all the predictions by now
    :param predictions: the predictions for a specific batch in a specific epoch
    :param gold_labels: the labels for a specific batch
    :param metadata: the metadata for a specific batch
    :param epoch: the number of epoch
    :param is_train: if this is train data or not
    :param mask: mask for each sample- which index is a padding
    :return:
    """

    for i in range(predictions.shape[0]):  # go over each sample
        if epoch == 0:  # if this is the first epoch - keep the label
            gold_labels_list = gold_labels[i][:mask[i].sum()].tolist()
            prediction_df[metadata[i]['sample_id']] = \
                {'is_train': is_train, f'labels': gold_labels_list,
                 f'total_payoff_label': sum(gold_labels_list) / len(gold_labels_list)}
        predictions_list = predictions[i][:mask[i].sum()].argmax(1).tolist()
        prediction_df[metadata[i]['sample_id']][f'predictions_{epoch}'] = predictions_list
        prediction_df[metadata[i]['sample_id']][f'total_payoff_prediction_{epoch}'] =\
            sum(predictions_list) / len(predictions_list)

    # prediction_df_temp = pd.DataFrame.from_dict(predictions_dict, orient='index')
    #
    # if epoch != 0:
    #     prediction_df = prediction_df.merge(prediction_df_temp, how='left', right_index=True, left_index=True)
    # else:
    #     prediction_df = pd.concat([prediction_df, prediction_df_temp])

    return prediction_df


def save_predictions(prediction_df: pd.DataFrame, predictions: torch.Tensor, gold_labels: torch.Tensor, metadata,
                     epoch: int, is_train: bool) -> pd.DataFrame:
    """
    This function get the predictions class and save with the gold labels
    :param prediction_df: dataframe with all the predictions by now
    :param predictions: the predictions for a specific batch in a specific epoch
    :param gold_labels: the labels for a specific batch
    :param metadata: the metadata for a specific batch
    :param epoch: the number of epoch
    :param is_train: if this is train data or not
    :return:
    """

    # Create a data frame with the sample ID, the prediction, the label and if the prediction is correct
    label_prediction = \
        pd.concat([pd.DataFrame(metadata).sample_id,
                   pd.DataFrame(gold_labels.view(gold_labels.shape[0], -1).argmax(1), columns=['label']),
                   pd.DataFrame(predictions.view(predictions.shape[0], -1).argmax(1),
                                columns=[f'prediction_{epoch}'])], axis=1)
    label_prediction[f'correct_{epoch}'] =\
        np.where(label_prediction[f'prediction_{epoch}'] == label_prediction.label, 1, 0)
    if epoch == 0:  # if this is the first epoch - keep the label
        train = pd.DataFrame(pd.Series([is_train], name='is_train').repeat(label_prediction.shape[0]))
        train.index = label_prediction.index
        label_prediction = pd.concat([train, label_prediction], axis=1)
        prediction_df = pd.concat([prediction_df, label_prediction])

    else:  # if this is not the first label drop the label
        label_prediction = label_prediction.drop('label', axis=1)
        prediction_df = prediction_df.merge(label_prediction, how='left', on='sample_id')
        if f'prediction_{epoch}_x' in prediction_df.columns:
            prediction_df[f'prediction_{epoch}_x'] = np.where(prediction_df[f'prediction_{epoch}_x'].isnull(),
                                                              prediction_df[f'prediction_{epoch}_y'],
                                                              prediction_df[f'prediction_{epoch}_x'])
            prediction_df[f'correct_{epoch}_x'] = np.where(prediction_df[f'correct_{epoch}_x'].isnull(),
                                                           prediction_df[f'correct_{epoch}_y'],
                                                           prediction_df[f'correct_{epoch}_x'])
            prediction_df = prediction_df.drop([f'correct_{epoch}_y', f'prediction_{epoch}_y'], axis=1)
            prediction_df.rename(columns={f'correct_{epoch}_x': f'correct_{epoch}',
                                          f'prediction_{epoch}_x': f'prediction_{epoch}'}, inplace=True)

    return prediction_df


def get_reviews_representation(word_embedding: TextFieldEmbedder, review_representation_layer: Model,
                               sequence_review: Dict[str, torch.LongTensor], fc_layer: Model, max_tokens_len: int=None)\
        -> list:
    """
    Get sequnce of reviews and create its representation by creating a representation for each review
    by creating a representation for each token in the review and average these representations
    :param word_embedding: TextFieldEmbedder model to represent tokens as vector
    :param review_representation_layer: a model that get tensor with each reviews' tokens representation and
    :param sequence_review: dict with one key: 'tokens': this is a tensor with shape:
    [batch_size, max_seq_length (in this batch), 124, 50]
    :param fc_layer: fully connected layer to convert from [10, 124] to [10, 10] each review
    :param max_tokens_len: the max number of tokens in a review - for padding
    :return: the seq reviews shape: [seq_length, seq_length, max_num_words]
    """

    sequence_reviews_representation = list()
    for seq in sequence_review[list(word_embedding._token_embedders.keys())[0]]:
        # create emnedding for each token in each review
        tokens = {'tokens': seq}
        mask = get_text_field_mask(tokens)  # shape: [seq_length, max_num_words]
        embeddings = word_embedding(tokens)  # shape: [seq_length, max_num_words, embedding_size]
        # attention and average each review's tokens
        # shape: [seq_length, max_num_words, max_num_words]
        reviews_representation = review_representation_layer(embeddings, mask)
        # convert reviews_representation to have the max size of num_tokens
        if max_tokens_len is not None:
            padding_size = max_tokens_len - reviews_representation.shape[1]
            reviews_representation = torch.nn.functional.pad(input=reviews_representation, pad=[0, padding_size, 0, 0],
                                                             value=0)
            reviews_representation_final = fc_layer(reviews_representation)
            sequence_reviews_representation.append(reviews_representation_final)

        else:
            sequence_reviews_representation.append(reviews_representation)

    return sequence_reviews_representation


def calculate_loss_metrics(class_logits, class_probabilities, label, loss, metrics) -> dict:
    """
    This function calculate the loos and other metrics and return the output dict
    :param class_logits:
    :param class_probabilities:
    :param label:
    :param loss:
    :param metrics:
    :return:
    """

    output_dict = dict()
    output_dict['class_logits'] = class_logits
    output_dict['class_probabilities'] = class_probabilities

    # predictions = class_probabilities.cpu().data.numpy()

    if label is not None:
        # if class_logits.shape != label.squeeze(-1).shape:
        loss = loss(class_logits.squeeze(-1), label.squeeze(-1).float())
        for metric_name, metric in metrics.items():
            metric(class_logits.squeeze(-1), label.squeeze(-1))
        output_dict["loss"] = loss

    return output_dict


def calculate_loss(logits, label, criterion):
    """
    This function calculate the loos and other metrics and return the output dict
    :param logits: the prediction
    :param label: the truth label
    :param criterion: the criterion to use
    :return:
    """

    # if class_logits.shape != label.squeeze(-1).shape:
    # loss = criterion(logits.squeeze(-1), torch.max(label, -1)[1].float())
    loss = criterion(logits.squeeze(-1), label.squeeze(-1).float())

    return loss


class BasicTextModel(Model):
    """
    This model is a basic model that predict the label based only on the text
    1. represent the text of each review in the sequence using embedding model (e.g average of ELMO's vectors)
    2. average (or weight average) the reviews' representations
    3. projection layer
    4. classify the sequence (loss)
    """
    def __init__(self,
                 word_embedding: TextFieldEmbedder,
                 review_representation_layer: Model,
                 seq_representation_layer: Model,
                 vocab: Vocabulary,
                 classifier_feedforward: FeedForward,
                 fc_review_rep: FeedForward,
                 criterion,
                 metrics_dict: dict,
                 add_numbers: bool=False,
                 initializer: InitializerApplicator = InitializerApplicator()):
        """

        :param word_embedding: TextFieldEmbedder model to represent tokens as vector
        :param review_representation_layer: a model that get tensor with each reviews' tokens representation and
        return the review's representation, its input should be:
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None
        :param seq_representation_layer: a model that get tensor with each seq's reviews representation and
        return the seq's representation, its input should be:
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None
        :param vocab: the vocabulary to use
        :param classifier_feedforward: a feedforward layer with input_dim=10 (max_seq_length)
                and output_dim=(10 - number of classes or 1 --> according to the criterion)
        :param criterion: the loss to use
        :param add_numbers: whether to add numbers (trial payoff and trail lottery result)
        :param metrics_dict: dict with the metrics to measure
        :param fc_review_rep: fully connected layer to convert from [10, 124] to [10, 10] each review
        :param initializer:
        """
        super(BasicTextModel, self).__init__(vocab)
        self.word_embedding = word_embedding
        self.review_representation_layer = review_representation_layer
        self.seq_attention_layer = seq_representation_layer
        # self.projection = nn.Linear(self.encoder.get_output_dim(), out_size)
        self.classifier_feedforward = classifier_feedforward
        self.softmax = F.softmax
        self.metrics = metrics_dict
        self.loss = criterion
        self.add_numbers = add_numbers
        self.fc_review_rep = fc_review_rep
        self._epoch = 0
        self.predictions = pd.DataFrame()
        self._first_pair = None
        initializer(self)

    def forward(self,
                sequence_review: Dict[str, torch.LongTensor],
                metadata,
                label: torch.LongTensor = None,
                numbers: torch.FloatTensor = None,
                lottery_ep_label_field: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param sequence_review: dict with one key: 'tokens': this is a tensor with shape:
        [batch_size, max_seq_length (in this batch), 114, 50]
        :param metadata: list of dicts, each dict is the metadata of a single sequence of reviews
        :param label: tensor of size [batch_size] with the label for each sequence
        :param numbers: list of 2 floats: trial payoff and trail lottery result
        :param lottery_ep_label_field: 2 other labels to count loss on: lottery results and expected payoff
        :return: output_dict with loss, class_logits and class_probabilities
        """

        if self._first_pair is not None:
            if self._first_pair == metadata[0]['pair_id']:
                self._epoch += 1
        else:
            self._first_pair = metadata[0]['pair_id']

        sequence_reviews_representation =\
            get_reviews_representation(self.word_embedding, self.review_representation_layer, sequence_review,
                                       self.fc_review_rep)
        sequence_reviews_representation = torch.stack(sequence_reviews_representation)

        # attention over the seq representation
        seq_representation = self.seq_attention_layer(sequence_reviews_representation)

        class_logits = self.classifier_feedforward(seq_representation)
        class_probabilities = self.softmax(class_logits, dim=1)

        output_dict = calculate_loss_metrics(class_logits, class_probabilities, label, self.loss, self.metrics)

        self.predictions = save_predictions(prediction_df=self.predictions, predictions=class_logits,
                                            gold_labels=label, metadata=metadata, epoch=self._epoch,
                                            is_train=self.training)

        return output_dict

    def get_metrics(self, train=True, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = torch.tensor([self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices])
        output_dict['label'] = labels
        return output_dict


class BasicTextDecisionResultModel(Model):
    """
    This model is the first model:
    1. represent the text of each review in the sequence using embedding model (e.g average of ELMO's vectors)
    2. average (or weight average) the reviews' representations
    3. projection layer
    4. classify the sequence (loss)
    """
    def __init__(self,
                 word_embedding: TextFieldEmbedder,
                 review_representation_layer: Model,
                 seq_representation_layer: Model,
                 vocab: Vocabulary,
                 classifier_feedforward_classification: FeedForward,
                 classifier_feedforward_regression: FeedForward,
                 fc_review_rep: FeedForward,
                 criterion_classification,
                 criterion_regression,
                 metrics_dict: dict,
                 add_numbers: bool=False,
                 max_tokens_len: int = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        """
        :param word_embedding: TextFieldEmbedder model to represent tokens as vector
        :param review_representation_layer: a model that get tensor with each reviews' tokens representation and
        return the review's representation, its input should be:
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None
        :param seq_representation_layer: a model that get tensor with each seq's reviews representation and
        return the seq's representation, its input should be:
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None
        :param vocab: the vocabulary to use
        :param classifier_feedforward_classification: a feedforward layer with input_dim=130
                and output_dim=(2 - number of classes)
        :param classifier_feedforward_regression: a feedforward layer with input_dim=130
                and output_dim=1
        :param criterion_classification: the loss to use for classification loss
        :param criterion_regression: the loss to use for regression loss
        :param add_numbers: whether to add numbers (trial payoff and trail lottery result)
        :param metrics_dict: dict with the metrics to measure
        :param max_tokens_len: the max number of tokens in a review - for padding
        :param initializer:
        """
        super(BasicTextDecisionResultModel, self).__init__(vocab)
        self.word_embedding = word_embedding
        self.review_representation_layer = review_representation_layer
        self.seq_attention_layer = seq_representation_layer
        # self.projection = nn.Linear(self.encoder.get_output_dim(), out_size)
        self.classifier_feedforward_decision = classifier_feedforward_classification
        self.classifier_feedforward_lottery = classifier_feedforward_regression
        self.classifier_feedforward_expected_payoff = classifier_feedforward_regression
        self.softmax = F.softmax
        self.metrics = metrics_dict
        self.loss_decision = criterion_classification
        self.loss_lottery = criterion_regression
        self.loss_expected_payoff = criterion_regression
        self.add_numbers = add_numbers
        self.fc_review_rep = fc_review_rep
        self.loss_weights = [1/3, 1/3, 1/3]
        self._max_tokens_len = max_tokens_len
        self.predictions = pd.DataFrame()
        self._epoch = 0
        self._first_pair = None
        initializer(self)

    def forward(self,
                sequence_review: Dict[str, torch.LongTensor],
                metadata,
                numbers: torch.FloatTensor,
                label: torch.LongTensor = None,
                lottery_ep_label_field: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param sequence_review: dict with one key: 'tokens': this is a tensor with shape:
        [batch_size, max_seq_length (in this batch), 114, 50]
        :param metadata: list of dicts, each dict is the metadata of a single sequence of reviews
        :param label: tensor of size [batch_size] with the label for each sequence
        :param numbers: list of 2 floats and 1 int: round expected payoff, round lottery result and DM decision
        :param lottery_ep_label_field: 2 other labels to count loss on: lottery results and expected payoff
        :return: output_dict with loss, class_logits and class_probabilities
        """

        if self._first_pair is not None:
            if self._first_pair == metadata[0]['pair_id']:
                self._epoch += 1
        else:
            self._first_pair = metadata[0]['pair_id']

        output_dict = dict()
        sequence_reviews_representation =\
            get_reviews_representation(self.word_embedding, self.review_representation_layer, sequence_review,
                                       self.fc_review_rep, self._max_tokens_len)
        sequence_reviews_representation = torch.stack(sequence_reviews_representation)

        # attention over the seq representation
        # seq_representation = self.seq_attention_layer(sequence_reviews_representation)

        # concat texts representation and numbers
        # TODO: from here to the end - create a function
        if numbers is not None:
            texts_numbers = list()
            for i in range(numbers.shape[0]):
                texts_numbers.append(torch.flatten(torch.cat((sequence_reviews_representation[i], numbers[i]), dim=1)))
            texts_numbers = torch.stack(texts_numbers)
        else:
            texts_numbers = sequence_reviews_representation

        # get predictions
        output_dict['decision_logits'] = self.classifier_feedforward_decision(texts_numbers)
        output_dict['lottery_logits'] = self.classifier_feedforward_lottery(texts_numbers)
        output_dict['expected_payoff_logits'] = self.classifier_feedforward_expected_payoff(texts_numbers)

        # class_probabilities = self.softmax(class_logits, dim=1)

        # calculate loss and metrics
        if label is not None:
            if lottery_ep_label_field is not None:
                decision_loss = calculate_loss(output_dict['decision_logits'], label, self.loss_decision)
                lottery_loss = torch.sqrt(calculate_loss(output_dict['lottery_logits'], lottery_ep_label_field[:, :1],
                                                         self.loss_lottery))
                expected_payoff_loss = torch.sqrt(calculate_loss(output_dict['expected_payoff_logits'],
                                                                 lottery_ep_label_field[:, 1:],
                                                                 self.loss_expected_payoff))

                losses = [decision_loss, lottery_loss, expected_payoff_loss]
                final_loss = list()
                for i in range(len(losses)):
                    final_loss.append(self.loss_weights[i] * losses[i])
                final_loss = sum(final_loss)
                output_dict['loss'] = final_loss

                for metric_name, metric in self.metrics.items():
                    metric(output_dict['decision_logits'], label.view(label.shape[0], -1).argmax(1))
            else:  # use only decision label # TODO: debug this
                class_probabilities = self.softmax(output_dict['decision_logits'], dim=1)
                output_dict = calculate_loss_metrics(output_dict['decision_logits'], class_probabilities, label,
                                                     self.loss, self.metrics)

        self.predictions = save_predictions(prediction_df=self.predictions, predictions=output_dict['decision_logits'],
                                            gold_labels=label, metadata=metadata, epoch=self._epoch,
                                            is_train=self.training)

        return output_dict

    def get_metrics(self, train=True, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = torch.tensor([self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices])
        output_dict['label'] = labels
        return output_dict


class BasicFixTextFeaturesDecisionResultModel(Model):
    """
    This model is the first model:
    1. Represent the text of each review in the sequence using the fix features
    2. Concat the text and the numbers features
    3. Use 3 FF layers for the lottery results, decision and the DM expected payoff
    4. calculte the average loss
    """
    def __init__(self,
                 vocab: Vocabulary,
                 classifier_feedforward_classification: FeedForward,
                 classifier_feedforward_regression: FeedForward,
                 criterion_classification,
                 criterion_regression,
                 metrics_dict: dict,
                 add_numbers: bool = False,
                 max_tokens_len: int = None,
                 text_feedforward: FeedForward = None,
                 regularizer: RegularizerApplicator = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        """
        :param vocab: the vocabulary to use
        :param text_feedforward: if the text representation is too big- add this layer
        :param classifier_feedforward_classification: a feedforward layer with input_dim=130
                and output_dim=(2 - number of classes)
        :param classifier_feedforward_regression: a feedforward layer with input_dim=130
                and output_dim=1
        :param criterion_classification: the loss to use for classification loss
        :param criterion_regression: the loss to use for regression loss
        :param add_numbers: whether to add numbers (trial payoff and trail lottery result)
        :param metrics_dict: dict with the metrics to measure
        :param max_tokens_len: the max number of tokens in a review - for padding
        :param regularizer: regularizer to use
        :param initializer:
        """
        super(BasicFixTextFeaturesDecisionResultModel, self).__init__(vocab, regularizer)
        self.classifier_feedforward_decision = classifier_feedforward_classification
        self.classifier_feedforward_lottery = classifier_feedforward_regression
        self.classifier_feedforward_expected_payoff = classifier_feedforward_regression
        self.text_feedforward = text_feedforward
        self.softmax = F.softmax
        self.metrics = metrics_dict
        self.loss_decision = criterion_classification
        self.loss_lottery = criterion_regression
        self.loss_expected_payoff = criterion_regression
        self._add_numbers = add_numbers
        self._loss_weights = [1/3, 1 / 3, 1 / 3]
        self._max_tokens_len = max_tokens_len
        self.predictions = pd.DataFrame()
        self._epoch = 0
        self._first_pair = None
        initializer(self)

    def forward(self,
                sequence_review: torch.FloatTensor,
                metadata,
                numbers: torch.FloatTensor,
                label: torch.LongTensor = None,
                lottery_ep_label_field: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param sequence_review: dict with one key: 'tokens': this is a tensor with shape:
        [batch_size, max_seq_length (in this batch), 114, 50]
        :param metadata: list of dicts, each dict is the metadata of a single sequence of reviews
        :param label: tensor of size [batch_size] with the label for each sequence
        :param numbers: list of 2 floats and 1 int: round expected payoff, round lottery result and DM decision
        :param lottery_ep_label_field: 2 other labels to count loss on: lottery results and expected payoff
        :return: output_dict with loss, class_logits and class_probabilities
        """

        if self._first_pair is not None:
            if self._first_pair == metadata[0]['pair_id']:
                self._epoch += 1
        else:
            self._first_pair = metadata[0]['pair_id']

        output_dict = dict()
        if self.text_feedforward is None:
            sequence_reviews_representation = sequence_review
        else:
            sequence_reviews_representation = self.text_feedforward(sequence_review)
        # concat texts representation and numbers
        if numbers is not None:
            texts_numbers = torch.cat((sequence_reviews_representation, numbers), axis=-1)
            texts_numbers = texts_numbers.view(len(texts_numbers), -1)
        else:
            texts_numbers = sequence_reviews_representation

        # get predictions
        output_dict['decision_logits'] = self.classifier_feedforward_decision(texts_numbers)
        if numbers is not None:
            output_dict['lottery_logits'] = self.classifier_feedforward_lottery(texts_numbers)
            output_dict['expected_payoff_logits'] = self.classifier_feedforward_expected_payoff(texts_numbers)

        # class_probabilities = self.softmax(class_logits, dim=1)

        # calculate loss and metrics
        if label is not None:
            if lottery_ep_label_field is not None:
                decision_loss = calculate_loss(output_dict['decision_logits'], label, self.loss_decision)
                lottery_loss = torch.sqrt(calculate_loss(output_dict['lottery_logits'], lottery_ep_label_field[:, :1],
                                                         self.loss_lottery))
                expected_payoff_loss = torch.sqrt(calculate_loss(output_dict['expected_payoff_logits'],
                                                                 lottery_ep_label_field[:, 1:],
                                                                 self.loss_expected_payoff))

                losses = [decision_loss, lottery_loss, expected_payoff_loss]
                final_loss = list()
                for i in range(len(losses)):
                    final_loss.append(self._loss_weights[i] * losses[i])
                final_loss = sum(final_loss)
                output_dict['loss'] = final_loss

                predictions = (output_dict['decision_logits'].squeeze() > 0.5).type(label.dtype)
                for metric_name, metric in self.metrics.items():
                    metric(predictions, label.view(label.shape[0], -1).argmax(1))
            else:
                decision_loss = calculate_loss(output_dict['decision_logits'], label, self.loss_decision)
                output_dict['loss'] = decision_loss
                predictions = (output_dict['decision_logits'].squeeze() > 0.5).type(label.dtype)
                for metric_name, metric in self.metrics.items():
                    metric(predictions, label.view(label.shape[0], -1).argmax(1))

        self.predictions = save_predictions(prediction_df=self.predictions, predictions=output_dict['decision_logits'],
                                            gold_labels=label, metadata=metadata, epoch=self._epoch,
                                            is_train=self.training)

        return output_dict

    def get_metrics(self, train=True, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = torch.tensor([self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices])
        output_dict['label'] = labels
        return output_dict


class BasicFixTextFeaturesDecisionModel(Model):
    """
    This model is the first model:
    1. Represent the text of each review in the sequence using the fix features
    2. Classify the DM decision and calculate the loss (classification task)
    """
    def __init__(self,
                 vocab: Vocabulary,
                 classifier_feedforward_classification: FeedForward,
                 criterion_classification,
                 metrics_dict: dict,
                 max_tokens_len: int = None,
                 text_feedforward: FeedForward = None,
                 regularizer: RegularizerApplicator = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        """
        :param vocab: the vocabulary to use
        :param text_feedforward: if the text representation is too big- add this layer
        :param classifier_feedforward_classification: a feedforward layer with input_dim=130
                and output_dim=(2 - number of classes)
        :param criterion_classification: the loss to use for classification loss
        :param metrics_dict: dict with the metrics to measure
        :param max_tokens_len: the max number of tokens in a review - for padding
        :param regularizer: regularizer to use
        :param initializer:
        """
        super(BasicFixTextFeaturesDecisionModel, self).__init__(vocab, regularizer)
        self.classifier_feedforward_decision = classifier_feedforward_classification
        self.text_feedforward = text_feedforward
        self.softmax = F.softmax
        self.metrics = metrics_dict
        self.loss_decision = criterion_classification
        self._max_tokens_len = max_tokens_len
        self.predictions = pd.DataFrame()
        self._epoch = 0
        self._first_pair = None
        initializer(self)

    def forward(self,
                sequence_review: torch.FloatTensor,
                metadata,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param sequence_review: dict with one key: 'tokens': this is a tensor with shape:
        [batch_size, max_seq_length (in this batch), 114, 50]
        :param metadata: list of dicts, each dict is the metadata of a single sequence of reviews
        :param label: tensor of size [batch_size] with the label for each sequence
        :return: output_dict with loss, class_logits and class_probabilities
        """

        if self._first_pair is not None:
            if self._first_pair == metadata[0]['pair_id']:
                self._epoch += 1
        else:
            self._first_pair = metadata[0]['pair_id']

        output_dict = dict()
        if self.text_feedforward is None:
            sequence_reviews_representation = sequence_review
        else:
            sequence_reviews_representation = self.text_feedforward(sequence_review)
        # concat texts representation and numbers

        # get predictions
        output_dict['decision_logits'] = self.classifier_feedforward_decision(sequence_reviews_representation)
        output_dict['decision_logits'] = output_dict['decision_logits'].squeeze(1)

        # class_probabilities = self.softmax(class_logits, dim=1)

        # calculate loss and metrics
        if label is not None:
            decision_loss = calculate_loss(output_dict['decision_logits'], label, self.loss_decision)
            output_dict['loss'] = decision_loss
            predictions = (output_dict['decision_logits'].squeeze() > 0.5).type(label.dtype)
            for metric_name, metric in self.metrics.items():
                metric(predictions, label.view(label.shape[0], -1).argmax(1))

        # self.predictions = save_predictions(prediction_df=self.predictions, predictions=output_dict['decision_logits'],
        #                                     gold_labels=label, metadata=metadata, epoch=self._epoch,
        #                                     is_train=self.training)

        return output_dict

    def get_metrics(self, train=True, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = torch.tensor([self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices])
        output_dict['label'] = labels
        return output_dict


class AttentionSoftMaxLayer(Model):
    """
    This model get a MatrixAttention model and return for each row in the input matrix
     a vector that represent the sentence in this row
    """

    def __init__(self,
                 matrix_attention_layer: MatrixAttention,
                 vocab=Vocabulary()):
        super(AttentionSoftMaxLayer, self).__init__(vocab)
        self.matrix_attention_layer = matrix_attention_layer
        self.softmax = nn.Softmax(dim=-1)
        self.masked_softmax = masked_softmax

    def forward(self,
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None):
        """
        :param matrix: tensor of shape: [max_seq_length (in this batch), max_num_words, dim_size] or
        [batch_size, seq_length (in this batch), max_num_words]
        :param matrix_mask: tensor with shape: [max_seq_length (in this batch), max_num_words] that tells which tokens
        are only padding
        and which are real
        :return: tensor of the reviews representation with shape: [seq_length, max_num_words]
        """
        attention_scores = self.matrix_attention_layer(matrix, matrix)
        attention_weights = self.masked_softmax(attention_scores, matrix_mask)

        output = attention_weights.mean(dim=1)

        return output


class BasicLSTMFixTextFeaturesDecisionResultModel(Model):
    """
    This is a LSTM model that predict the class for each 't'
    """
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 metrics_dict: dict,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.metrics = metrics_dict
        self.predictions = defaultdict(dict)
        self._epoch = 0
        self._first_pair = None

    def forward(self,
                sequence_review: torch.Tensor,
                metadata: list,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        if self._first_pair is not None:
            if self._first_pair == metadata[0]['pair_id']:
                self._epoch += 1
        else:
            self._first_pair = metadata[0]['pair_id']

        mask = get_text_field_mask({'tokens': sequence_review})
        encoder_out = self.encoder(sequence_review, mask)
        decision_logits = self.hidden2tag(encoder_out)
        output = {'decision_logits': decision_logits}
        if labels is not None:
            for metric_name, metric in self.metrics.items():
                metric(decision_logits, labels, mask)
            output['loss'] = sequence_cross_entropy_with_logits(decision_logits, labels, mask)

        self.predictions = save_predictions_seq_models(prediction_df=self.predictions, mask=mask,
                                                       predictions=output['decision_logits'], gold_labels=labels,
                                                       metadata=metadata, epoch=self._epoch, is_train=self.training)

        return output

    def get_metrics(self, train=True, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
