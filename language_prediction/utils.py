from allennlp.training.metrics import *
from typing import *
import torch
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import math
import os
import logging
import copy


def calculate_measures(train_data: pd.DataFrame, validation_data: pd.DataFrame, metric_list: List[str],
                       label_col_name: str='label') -> tuple([dict, dict]):
    """
    This function get train and validation data that has label and prediction columns and calculate the measures in
    the metric_list
    :param train_data: pd.DataFrame with the train data, has to have at least label and prediction columns
    :param validation_data: pd.DataFrame with the validation data, has to have at least label and prediction columns
    :param metric_list: a list with the metric names to calculate
    :param label_col_name: the name of the label column
    :return:
    """
    # calculate metric(y_true, y_pred)
    validation_metric_dict = dict()
    train_metric_dict = dict()
    for metric in metric_list:
        validation_metric_dict[metric] =\
            getattr(metrics, metric)(validation_data[label_col_name], validation_data.prediction)
        train_metric_dict[metric] = getattr(metrics, metric)(train_data[label_col_name], train_data.prediction)

    return train_metric_dict, validation_metric_dict


def create_bin_columns(predictions: pd.Series, validation_y: pd.Series, hotel_label_0: bool=False):
    """
    Create the bin analysis column
    :param predictions: the continues prediction column
    :param validation_y: the continues label column
    :param hotel_label_0: if the label of the hotel option is 0
    :return:
    """

    # bin measures,
    # class: hotel_label == 1: predictions < 0.33 --> 0, 0.33<predictions<0.67 --> 1, predictions > 0.67 --> 2
    #        hotel_label == 0: predictions < 0.33 --> 2, 0.33<predictions<0.67 --> 1, predictions > 0.67 --> 0
    low_entry_rate_class = 2 if hotel_label_0 else 0
    high_entry_rate_class = 0 if hotel_label_0 else 2
    # for prediction
    keep_mask = predictions < 0.33
    bin_prediction = np.where(predictions < 0.67, 1, high_entry_rate_class)
    bin_prediction[keep_mask] = low_entry_rate_class
    bin_prediction = pd.Series(bin_prediction, name='bin_predictions', index=validation_y.index)
    # for test_y
    keep_mask = validation_y < 0.33
    bin_test_y = np.where(validation_y < 0.67, 1, high_entry_rate_class)
    bin_test_y[keep_mask] = low_entry_rate_class
    bin_test_y = pd.Series(bin_test_y, name='bin_label', index=validation_y.index)

    return bin_prediction, bin_test_y


def calculate_per_round_per_raisha_measures(all_predictions: pd.DataFrame, predictions_column: str, label_column: str,
                                            label_options: list):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :return:
    """
    raishas = all_predictions.raisha.unique()
    results_dict = dict()
    for raisha in raishas:
        data = copy.deepcopy(all_predictions.loc[all_predictions.raisha == raisha])
        results = calculate_per_round_measures(data, predictions_column, label_column, label_options,
                                               raisha=f'raisha_{raisha}')
        results_dict.update(results)

    return results_dict


def calculate_per_round_measures(all_predictions: pd.DataFrame, predictions_column: str, label_column: str,
                                 label_options: list, raisha: str=''):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :param raisha: the suffix for the columns in raisha analysis
    :return:
    """
    results_dict = dict()
    precision, recall, fbeta_score, support =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column])
    accuracy = metrics.accuracy_score(all_predictions[label_column], all_predictions[predictions_column])

    # number of DM chose stay home
    status_size = all_predictions[label_column].where(all_predictions[label_column] == -1).dropna().shape[0]
    index_in_support = np.where(support == status_size)
    if index_in_support[0][0] == 0:  # the first in support is the label -1
        label_options = label_options
        f_score_list = [fbeta_score[0], fbeta_score[1]]
    else:
        label_options = [label_options[1], label_options[0]]
        f_score_list = [fbeta_score[1], fbeta_score[0]]

    # create the results to return
    for measure, measure_name in [[precision, 'Precision'], [recall, 'recall'], [f_score_list, 'Fbeta_score'],
                                  [accuracy, 'Accuract']]:
        for i, label in enumerate(label_options):
            results_dict[f'Per_round_{measure_name}_{label}{raisha}'] = 100 * round(measure[i], 2)

    return results_dict


def calculate_measures_for_continues_labels(all_predictions: pd.DataFrame, final_total_payoff_prediction_column: str,
                                            total_payoff_label_column: str, label_options: list,
                                            raisha: str='') -> (pd.DataFrame, dict):
    """
    Calc and print the regression measures, including bin analysis
    :param all_predictions:
    :param total_payoff_label_column: the name of the label column
    :param final_total_payoff_prediction_column: the name of the prediction label
    :param label_options: list of the label option names
    :param raisha: if we run a raisha analysis this is the raisha we worked with
    :return:
    """

    if 'is_train' in all_predictions.columns:
        data = all_predictions.loc[all_predictions.is_train == False]
    else:
        data = all_predictions

    results_dict = dict()
    predictions = data[final_total_payoff_prediction_column]
    gold_labels = data[total_payoff_label_column]
    mse = metrics.mean_squared_error(predictions, gold_labels)
    rmse = round(100 * math.sqrt(mse), 2)
    mae = round(100 * metrics.mean_absolute_error(predictions, gold_labels), 2)
    mse = round(100 * mse, 2)

    # calculate bin measures
    try:
        if 'bin_label' and 'bin_predictions' in all_predictions.columns:
            precision, recall, fbeta_score, support =\
                metrics.precision_recall_fscore_support(all_predictions.bin_label, all_predictions.bin_predictions)
            accuracy = metrics.accuracy_score(all_predictions.bin_label, all_predictions.bin_predictions)

            # number of DM chose stay home
            zero_status_size = all_predictions.bin_label.where(all_predictions.bin_label == 0).dropna().shape[0]
            one_status_size = all_predictions.bin_label.where(all_predictions.bin_label == 1).dropna().shape[0]
            zero_index_in_support = np.where(support == zero_status_size)
            one_index_in_support = np.where(support == one_status_size)
            if one_index_in_support[0][0] == 1 and zero_index_in_support[0][0] == 0:
                label_options = label_options
                f_score_list = [fbeta_score[0], fbeta_score[1], fbeta_score[2]]
            else:
                temp_labels = list()
                temp_f_score = list()
                if one_index_in_support[0][0] == 0:  # the first label option in support[0]
                    temp_labels.append(label_options[1])
                    temp_f_score.append(fbeta_score[1])
                else:  # the second label option in support[0]
                    temp_labels.append(label_options[2])
                    temp_f_score.append(fbeta_score[2])
                if zero_index_in_support[0][0] == 1:  # the zero label option in support[1]
                    temp_labels.append(label_options[0])
                    temp_f_score.append(fbeta_score[0])
                else:  # the second label option in support[1]
                    temp_labels.append(label_options[2])
                    temp_f_score.append(fbeta_score[2])
                if zero_index_in_support[0][0] == 2:  # the zero label option in support[2]
                    temp_labels.append(label_options[0])
                    temp_f_score.append(fbeta_score[0])
                else:  # the first label option in support[2]
                    temp_labels.append(label_options[1])
                    temp_f_score.append(fbeta_score[1])
                label_options = temp_labels
                f_score_list = temp_f_score

            accuracy = metrics.accuracy_score(all_predictions.bin_label, all_predictions.bin_predictions)
            accuracy = round(100 * accuracy, 2)
            results_dict['Bin_Accuracy'] = accuracy

            # create the results to return
            for measure, measure_name in [[precision, 'Precision'], [recall, 'recall'], [fbeta_score, 'Fbeta_score'],
                                          [accuracy, 'Accuracy']]:
                for i, label in enumerate(label_options):
                    results_dict[f'Bin_{measure_name}_{label}{raisha}'] = 100*round(measure[i], 2)
    except Exception:
        logging.exception(f'No bin analysis because the bin columns has not created')
        return

    results_dict['MSE'] = mse
    results_dict['RMSE'] = rmse
    results_dict['MAE'] = mae

    results_pd = pd.DataFrame.from_dict(results_dict, orient='index')

    return results_pd, results_dict


def main(log_directory: str, model_output_file_name: str, final_total_payoff_prediction_column_name: str,
         total_payoff_label_column_name: str):
    if 'csv' in model_output_file_name:
        all_predictions = pd.read_csv(os.path.join(log_directory, model_output_file_name))
    else:
        all_predictions = pd.read_excel(os.path.join(log_directory, model_output_file_name))
    results, _ = calculate_measures_for_continues_labels(
        all_predictions, final_total_payoff_prediction_column_name, total_payoff_label_column_name,
        label_options=['total future payoff < 1/3', '1/3 < total future payoff < 2/3', 'total future payoff > 2/3'])
    results.to_csv(os.path.join(log_directory, 'results.csv'))


if __name__ == '__main__':
    log_directory_main = '/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/' \
                         'language_prediction/logs/LSTMDatasetReader_100_epochs_5_folds_04_03_2020_11_46_40'
    model_output_file_name_main = 'predictions.csv'
    final_total_payoff_prediction_column_name_main = 'final_total_payoff_prediction'
    total_payoff_label_column_name_main = 'total_payoff_label'
    main(log_directory_main, model_output_file_name_main, final_total_payoff_prediction_column_name_main,
         total_payoff_label_column_name_main)
