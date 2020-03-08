from allennlp.training.metrics import *
from typing import *
import torch
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import math
import os


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


def calculate_measures_seq_models(all_predictions: pd.DataFrame, final_total_payoff_prediction_column: str,
                                  total_payoff_label_column: str, label_options: list):
    """
    Calc and print the regression measures, including bin analysis
    :param all_predictions:
    :return:
    """

    if 'is_train' in all_predictions.columns:
        data = all_predictions.loc[all_predictions.is_train == False]
    else:
        data = all_predictions

    predictions = data[final_total_payoff_prediction_column]
    gold_labels = data[total_payoff_label_column]
    mse = metrics.mean_squared_error(predictions, gold_labels)
    rmse = round(100 * math.sqrt(mse), 2)
    rmse_pd = pd.Series([rmse, '-'])
    mae = round(100 * metrics.mean_absolute_error(predictions, gold_labels), 2)
    mae_pd = pd.Series([mae, '-'])
    mse = round(100 * mse, 2)
    mse_pd = pd.Series([mse, '-'])

    # bin measures
    # for prediction
    keep_mask = predictions < 0.33
    bin_prediction = np.where(predictions < 0.67, 1, 2)
    bin_prediction[keep_mask] = 0
    bin_prediction = pd.Series(bin_prediction, name='bin_predictions', index=gold_labels.index)
    # for test_y
    keep_mask = gold_labels < 0.33
    bin_gold_label = np.where(gold_labels < 0.67, 1, 2)
    bin_gold_label[keep_mask] = 0
    bin_gold_label = pd.Series(bin_gold_label, name='bin_label', index=gold_labels.index)

    # calculate bin measures
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(bin_gold_label, bin_prediction)

    # number of DM chose stay home
    zero_status_size = bin_gold_label.where(bin_gold_label == 0).dropna().shape[0]
    one_status_size = bin_gold_label.where(bin_gold_label == 1).dropna().shape[0]
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

    accuracy = metrics.accuracy_score(bin_gold_label, bin_prediction)
    accuracy = round(100 * accuracy, 2)
    accuracy_pd = pd.Series([accuracy, '-'])

    results = pd.concat([pd.Series(100 * precision).round(2), pd.Series(100 * recall).round(2),
                         pd.Series(100 * fbeta_score).round(2), accuracy_pd, mse_pd, rmse_pd, mae_pd], axis=1).T

    results.index = ['Precision', 'Recall', 'Fbeta_score', 'Accuracy', 'MSE', 'RMSE', 'MAE']
    results.columns = label_options

    return results


def main(log_directory: str, model_output_file_name: str, final_total_payoff_prediction_column_name: str,
         total_payoff_label_column_name: str):
    if 'csv' in model_output_file_name:
        all_predictions = pd.read_csv(os.path.join(log_directory, model_output_file_name))
    else:
        all_predictions = pd.read_excel(os.path.join(log_directory, model_output_file_name))
    results = calculate_measures_seq_models(
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
