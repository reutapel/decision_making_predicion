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
from collections import defaultdict
from os.path import isfile, join, isdir
import ray
import analyze_model_results
base_directory = os.path.abspath(os.curdir)

per_round_predictions_name = 'per_round_predictions'
per_round_labels_name = 'per_round_labels'


def update_default_dict(orig_dict: defaultdict, dict2: defaultdict, dict3: defaultdict=None):
    """This function get an orig defaultdict and 1-2 defaultdicts and merge them"""
    for my_dict in (dict2, dict3):
        if my_dict is not None:
            for k, v in my_dict.items():
                if k in orig_dict.keys():
                    orig_dict[k].update(v)
                else:
                    orig_dict[k] = v

    return orig_dict


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


def per_round_analysis(all_predictions: pd.DataFrame, predictions_column: str, label_column: str, label_options: list,
                       function_to_run):
    """
    Analyze per round results: calculate measures for all rounds and per round
    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :param function_to_run: the function to run: calculate_per_round_per_raisha_measures or calculate_per_round_measures
    :return:
    """
    results_dict = globals()[function_to_run](all_predictions, predictions_column, label_column, label_options)

    if 'round_number' in all_predictions.columns:  # analyze the results per round
        for current_round_number in all_predictions.round_number.unique():
            data = all_predictions.loc[all_predictions.round_number == current_round_number].copy(deep=True)
            results = globals()[function_to_run](data, predictions_column, label_column, label_options,
                                                 round_number=f'round_{int(current_round_number)}')
            results_dict = update_default_dict(results_dict, results)

    return results_dict


def calculate_per_round_per_raisha_measures(all_predictions: pd.DataFrame, predictions_column: str, label_column: str,
                                            label_options: list, round_number: str='All_rounds'):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :param round_number: if we analyze specific round number
    :return:
    """
    raishas = all_predictions.raisha.unique()
    results_dict = defaultdict(dict)
    for raisha in raishas:
        data = copy.deepcopy(all_predictions.loc[all_predictions.raisha == raisha])
        results = calculate_per_round_measures(data, predictions_column, label_column, label_options,
                                               raisha=f'raisha_{int(raisha)}', round_number=round_number)
        results_dict.update(results)

    return results_dict


def calculate_per_round_measures(all_predictions: pd.DataFrame, predictions_column: str, label_column: str,
                                 label_options: list, raisha: str='All_raishas', round_number: str='All_rounds'):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :param raisha: the suffix for the columns in raisha analysis
    :param round_number: if we analyze specific round number
    :return:
    """
    results_dict = defaultdict(dict)
    dict_key = f'{raisha} {round_number}'
    precision, recall, fbeta_score, support =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column])
    accuracy = metrics.accuracy_score(all_predictions[label_column], all_predictions[predictions_column])

    # number of DM chose stay home
    final_labels = list(range(len(support)))
    # get the labels in the all_predictions DF
    true_labels = all_predictions[label_column].unique()
    true_labels.sort()
    for label_index, label in enumerate(true_labels):
        status_size = all_predictions[label_column].where(all_predictions[label_column] == label).dropna().shape[0]
        if status_size in support:
            index_in_support = np.where(support == status_size)[0][0]
            final_labels[index_in_support] = label_options[label_index]

    # create the results to return
    for measure, measure_name in [[precision, 'precision'], [recall, 'recall'], [fbeta_score, 'Fbeta_score']]:
        for i, label in enumerate(final_labels):
            results_dict[dict_key][f'Per_round_{measure_name}_{label}'] = round(measure[i]*100, 2)
    results_dict[dict_key][f'Per_round_Accuracy'] = round(accuracy*100, 2)

    return results_dict


def calculate_measures_for_continues_labels(all_predictions: pd.DataFrame, final_total_payoff_prediction_column: str,
                                            total_payoff_label_column: str, label_options: list,
                                            raisha: str = 'All_raishas', round_number: str = 'All_rounds',
                                            bin_label: pd.Series=None, bin_predictions: pd.Series=None) ->\
        (pd.DataFrame, dict):
    """
    Calc and print the regression measures, including bin analysis
    :param all_predictions:
    :param total_payoff_label_column: the name of the label column
    :param final_total_payoff_prediction_column: the name of the prediction label
    :param label_options: list of the label option names
    :param raisha: if we run a raisha analysis this is the raisha we worked with
    :param round_number: for per round analysis
    :param bin_label: the bin label series, the index is the same as the total_payoff_label_column index
    :param bin_predictions: the bin predictions series, the index is the same as the total_payoff_label_column index
    :return:
    """
    dict_key = f'{raisha} {round_number}'
    if 'is_train' in all_predictions.columns:
        data = all_predictions.loc[all_predictions.is_train == False]
    else:
        data = all_predictions

    results_dict = defaultdict(dict)
    predictions = data[final_total_payoff_prediction_column]
    gold_labels = data[total_payoff_label_column]
    mse = metrics.mean_squared_error(predictions, gold_labels)
    rmse = round(100 * math.sqrt(mse), 2)
    mae = round(100 * metrics.mean_absolute_error(predictions, gold_labels), 2)
    mse = round(100 * mse, 2)

    # calculate bin measures
    if 'bin_label' and 'bin_predictions' in all_predictions.columns:
        bin_label = all_predictions.bin_label
        bin_predictions = all_predictions.bin_predictions
    elif bin_label is None and bin_predictions is None:
        print(f'No bin labels and bin predictions')
        logging.info(f'No bin labels and bin predictions')
        raise Exception

    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(bin_label, bin_predictions)

    # number of DM chose stay home
    final_labels = list(range(len(support)))
    for bin in range(len(label_options)):
        status_size = bin_label.where(bin_label == bin).dropna().shape[0]
        if status_size in support:
            index_in_support = np.where(support == status_size)[0][0]
            final_labels[index_in_support] = label_options[bin]

    accuracy = metrics.accuracy_score(bin_label, bin_predictions)
    results_dict[dict_key][f'Bin_Accuracy'] = round(accuracy * 100, 2)

    # create the results to return
    for measure, measure_name in [[precision, 'precision'], [recall, 'recall'], [fbeta_score, 'Fbeta_score']]:
        for i in range(len(measure)):
            results_dict[dict_key][f'Bin_{measure_name}_{final_labels[i]}'] = round(measure[i]*100, 2)

    results_dict[dict_key][f'MSE'] = mse
    results_dict[dict_key][f'RMSE'] = rmse
    results_dict[dict_key][f'MAE'] = mae

    results_pd = pd.DataFrame.from_dict(results_dict, orient='index')

    return results_pd, results_dict


def write_to_excel(table_writer: pd.ExcelWriter, sheet_name: str, headers: list, data: pd.DataFrame):
    """
    This function get header and data and write to excel
    :param table_writer: the ExcelWrite object
    :param sheet_name: the sheet name to write to
    :param headers: the header of the sheet
    :param data: the data to write
    :return:
    """
    workbook = table_writer.book
    if sheet_name not in table_writer.sheets:
        worksheet = workbook.add_worksheet(sheet_name)
    else:
        worksheet = workbook.get_worksheet_by_name(sheet_name)
    table_writer.sheets[sheet_name] = worksheet

    data.to_excel(table_writer, sheet_name=sheet_name, startrow=len(headers), startcol=0)
    all_format = workbook.add_format({
        'valign': 'top',
        'border': 1})
    worksheet.set_column(0, data.shape[1], None, all_format)

    # headers format
    merge_format = workbook.add_format({
        'bold': True,
        'border': 2,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True,
    })
    for i, header in enumerate(headers):
        worksheet.merge_range(first_row=i, first_col=0, last_row=i, last_col=data.shape[1], data=header,
                              cell_format=merge_format)
        # worksheet_header = pd.DataFrame(columns=[header])
        # worksheet_header.to_excel(table_writer, sheet_name=sheet_name, startrow=0+i, startcol=0)

    return


def set_folder(folder_name: str, father_folder_name: str = None, father_folder_path=None):
    """
    This function create new folder for results if does not exists
    :param folder_name: the name of the folder to create
    :param father_folder_name: the father name of the new folder
    :param father_folder_path: if pass the father folder path and not name
    :return: the new path or the father path if folder name is None
    """
    # create the father folder if not exists
    if father_folder_name is not None:
        path = os.path.join(base_directory, father_folder_name)
    else:
        path = father_folder_path
    if not os.path.exists(path):
        os.makedirs(path)
    # create the folder
    if folder_name is not None:
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

    return path


# @ray.remote
def eval_model(folder_list: list, fold_num: int):
    model_nums = list(range(29, 77)) + list(range(102, 178)) + list(range(190, 222)) + list(range(238, 318)) +\
                 list(range(334, 382))

    all_models_results = pd.DataFrame()

    for folder in folder_list:
        folder_path = os.path.join(base_directory, 'logs', folder)
        inner_folder = f'fold_{fold_num}'
        files_path = join(folder_path, inner_folder, 'excel_models_results')
        print(f'load from {files_path}')
        for model_num in model_nums:
            file_name = f'Results_{inner_folder}_model_{model_num}.xlsx'
            if isfile(join(files_path, file_name)):
                print(f'work on model num: {model_num}')
                df = pd.read_excel(os.path.join(files_path, file_name),
                                   sheet_name=f'Model_{model_num}_seq_predictions', skiprows=[0])
                validation_sample_ids = df.loc[df.is_train == False]['Unnamed: 0'].tolist()
                df = pd.read_excel(os.path.join(files_path, file_name),
                                   sheet_name=f'Model_{model_num}_per_round_predictions', skiprows=[0])
                df = df.loc[df.sample_id.isin(validation_sample_ids)]
                # measures per round
                label_options = ['DM chose hotel', 'DM chose stay home']
                results_dict = per_round_analysis(df, predictions_column=per_round_predictions_name,
                                                  label_column=per_round_labels_name, label_options=label_options,
                                                  function_to_run='calculate_per_round_measures')

                if 'raisha' in df:
                    results_dict_per_round_per_raisha = per_round_analysis(
                        df, predictions_column=per_round_predictions_name,
                        label_column=per_round_labels_name, label_options=label_options,
                        function_to_run='calculate_per_round_per_raisha_measures')
                    results_dict = update_default_dict(results_dict, results_dict_per_round_per_raisha)

                results_df = pd.DataFrame.from_dict(results_dict).T
                results_df['raisha_round'] = results_df.index
                results_df[['Raisha', 'Round']] = results_df.raisha_round.str.split(expand=True)
                results_df = results_df.drop('raisha_round', axis=1)
                results_df.index = np.zeros(shape=(results_df.shape[0],))
                metadata_df = pd.read_excel(os.path.join(files_path, file_name), sheet_name=f'Model results',
                                            skiprows=[0], index_col=0)
                metadata_df = pd.DataFrame(metadata_df[['model_num', 'model_type', 'model_name', 'data_file_name',
                                                        'hyper_parameters_str']].iloc[0])
                results_df = results_df.join(metadata_df.T)
                results_df.index = [f'{folder}_Results_{inner_folder}_model_{model_num}.xlsx'] * results_df.shape[0]
                all_models_results = pd.concat([all_models_results, results_df], sort='False')

    return all_models_results


def flat_seq_predictions_list_column(label_column_name_per_round: str,
                                     prediction_column_name_per_round: str,
                                     prediction: pd.DataFrame) -> pd.DataFrame:
    """
    Use the prediction DF to get one column of all rounds predictions and labels, in order to calculate
    the per round measures
    :param label_column_name_per_round: the name od the label column per round (for example: y_0, y_1, ..., y_9)
    :param prediction_column_name_per_round: the name od the prediction column per round
    (for example: y_prime_0, y_prime_1, ..., y_prime_9)
    :param prediction: the data
    :return: pd.Dataframe with 2 columns: labels and predictions with the labels and predictions per round for
    the saifa data
    """

    flat_data_dict = dict()
    for list_column, new_column in [[label_column_name_per_round, per_round_labels_name],
                                    [prediction_column_name_per_round, per_round_predictions_name]]:
        # create a pd with [new_column, 'raisha', 'sample_id'] columns
        flat_data = copy.deepcopy(prediction)
        # reset index to get numeric index for the future merge
        flat_data['sample_id'] = flat_data.index
        flat_data.reset_index(inplace=True, drop=True)
        flat_data = flat_data[[list_column, 'raisha', 'sample_id']]
        flat_data[list_column] =\
            flat_data[list_column].apply(lambda row: [int(item) for item in list(row) if item in ['0', '1']])
        lens_of_lists = flat_data[list_column].apply(len)
        origin_rows = range(flat_data.shape[0])
        destination_rows = np.repeat(origin_rows, lens_of_lists)
        non_list_cols = [idx for idx, col in enumerate(flat_data.columns) if col != list_column]
        expanded_df = flat_data.iloc[destination_rows, non_list_cols].copy()
        expanded_df[new_column] = [i for items in flat_data[list_column] for i in items]
        # remove non 0/1 rows and reset index
        expanded_df = expanded_df.loc[expanded_df[new_column].isin(['0', '1'])]
        # create round number column
        round_number = pd.Series()
        for index, round_num in lens_of_lists.iteritems():
            round_number =\
                round_number.append(pd.Series(list(range(11-round_num, 11)), index=np.repeat(index, round_num)))
        expanded_df['round_number'] = round_number
        expanded_df.reset_index(inplace=True, drop=True)
        flat_data_dict[new_column] = expanded_df[[new_column]]
        flat_data_dict['metadata'] = expanded_df[['raisha', 'sample_id', 'round_number']]

    # concat the new labels and new predictions per round
    flat_data = flat_data_dict[per_round_labels_name].join(flat_data_dict[per_round_predictions_name]).\
        join(flat_data_dict['metadata'])
    flat_data.reset_index(inplace=True, drop=True)

    return flat_data


def eval_model_total_prediction(folder_list: list, fold_num: int, total_payoff_true_label: pd.Series):
    """
    :param folder_list:
    :param fold_num:
    :param total_payoff_true_label: total_payoff_true_label for this fold
    :return:
    """
    models_to_compare = pd.read_excel(os.path.join(base_directory, 'models_to_compare.xlsx'),
                                      sheet_name='table_to_load', skiprows=[0])
    models_to_eval = models_to_compare.loc[models_to_compare.function_to_run == 'ExecuteEvalLSTM']
    model_nums = models_to_eval.model_num.unique().tolist()
    all_models_combinations = copy.deepcopy(model_nums)
    for num in model_nums:
        for dropout in [None, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
            all_models_combinations.append(f'{str(num)}_{dropout}')
    # model_nums = list(range(29, 77)) + list(range(102, 178)) + list(range(190, 222)) + list(range(238, 318)) +\
    #              list(range(334, 382))
    all_models_results = list()
    final_total_payoff_prediction_column = 'final_total_payoff_prediction'
    total_payoff_label_column = 'total_payoff_label'
    raisha_column_name = 'raisha'

    for folder in folder_list:
        # if folder == 'compare_prediction_models_03_05_2020_11_53' and fold == 0:  # already done
        #     continue
        folder_path = os.path.join(base_directory, 'logs', folder)
        inner_folder = f'fold_{fold_num}'
        if isdir(join(folder_path, inner_folder, 'excel_models_results')):
            files_path = join(folder_path, inner_folder, 'excel_models_results')
        elif join(folder_path, inner_folder):
            files_path = join(folder_path, inner_folder)
        else:
            print('No correct directory')
            return
        print(f'load from {files_path}')
        all_results_file_name_inner = f'Results_{inner_folder}_all_models.xlsx'
        if isfile(join(files_path, all_results_file_name_inner)):
            xls = pd.ExcelFile(os.path.join(files_path, all_results_file_name_inner))
            all_results_file = True
        else:
            xls = None
            all_results_file = False

        # all_models_combinations = [76, 77]
        for model_num in all_models_combinations:
            all_reg_predictions = None
            all_seq_predictions = None
            print(f'work on model num: {model_num}')
            file_name = f'Results_{inner_folder}_model_{model_num}.xlsx'
            if not all_results_file:  # no all_results_file
                if isfile(join(files_path, file_name)):
                    xls = pd.ExcelFile(os.path.join(files_path, file_name))
                else:
                    print(f'No all results file and not model results file for model num: {model_num}')
                    continue
            if f'Model_{model_num}_reg_predictions' in xls.sheet_names:
                all_reg_predictions = pd.read_excel(xls, f'Model_{model_num}_reg_predictions', skiprows=[0],
                                                    index_col=0, parse_dates=True)
            elif f'Model_{model_num}_reg' in xls.sheet_names:
                all_reg_predictions = pd.read_excel(xls, f'Model_{model_num}_reg', skiprows=[0], index_col=0,
                                                    parse_dates=True)
            if f'Model_{model_num}_seq_predictions' in xls.sheet_names:
                all_seq_predictions = pd.read_excel(xls, f'Model_{model_num}_seq_predictions', skiprows=[0],
                                                    index_col=0, parse_dates=True)
            elif f'Model_{model_num}_seq' in xls.sheet_names:
                all_seq_predictions = pd.read_excel(xls, f'Model_{model_num}_seq', skiprows=[0], index_col=0,
                                                    parse_dates=True)
            if all_seq_predictions is None and all_reg_predictions is None:
                print('No sheet is in the file')
                continue

            if not all_results_file:  # no all_results_file
                metadata_df = pd.read_excel(xls, sheet_name=f'Model results', skiprows=[0], index_col=0)
            else:
                metadata_df = pd.read_excel(xls, f'All models results', skiprows=[0], index_col=0)
                metadata_df = metadata_df.loc[metadata_df.model_num == model_num]

            model_type = metadata_df['model_type'].unique()[0]
            model_name = metadata_df['model_name'].unique()[0]
            if 'turn' in model_type:
                predict_seq = True
            else:
                predict_seq = False

            if 'avg' in model_type:
                predict_avg_total_payoff = True
            else:
                predict_avg_total_payoff = False

            table_writer = pd.ExcelWriter(os.path.join(base_directory, 'logs', 'after_fix_bin_raisha',
                                                       f'{folder}_Results_{inner_folder}_model_{model_num}.xlsx'),
                                          engine='xlsxwriter')

            if predict_seq:
                if all_seq_predictions.shape[0] > 4000:  # duplicated predictions (bug in epoch+=1)
                    all_seq_predictions =\
                        all_seq_predictions.drop_duplicates(subset=['sample_id', raisha_column_name], keep='last')
                max_predict_column = max([int(column.split('_')[1]) for column in all_seq_predictions.columns if
                                          'predictions' in column])
                all_seq_predictions['final_prediction'] = all_seq_predictions[f'predictions_{max_predict_column}']
                max_total_payoff_predict_column = max([int(column.split('_')[3]) for column in
                                                       all_seq_predictions.columns if
                                                       'total_payoff_prediction_' in column])
                all_seq_predictions['final_total_payoff_prediction'] = \
                    all_seq_predictions[f'total_payoff_prediction_{max_total_payoff_predict_column}']
                all_seq_predictions = all_seq_predictions.loc[all_seq_predictions.is_train == False]
                all_seq_predictions = all_seq_predictions[['is_train', 'labels', 'total_payoff_label', 'raisha',
                                                           'final_prediction', 'final_total_payoff_prediction']]
                write_to_excel(
                    table_writer, f'Model_{model_num}_seq',
                    headers=[f'Seq predictions for model {model_num}: {model_name} of type {model_type} in fold {fold}'],
                    data=all_seq_predictions)
                all_prediction = all_seq_predictions

            if predict_avg_total_payoff:
                if all_reg_predictions.shape[0] > 4000:  # duplicated predictions (bug in epoch+=1)
                    all_reg_predictions =\
                        all_reg_predictions.drop_duplicates(subset=['sample_id', raisha_column_name], keep='last')
                # get the final prediction for reg models
                max_predict_column = max([int(column.split('_')[1]) for column in all_reg_predictions.columns if
                                          'prediction_' in column])
                all_reg_predictions['final_total_payoff_prediction'] =\
                    all_reg_predictions[f'prediction_{max_predict_column}']
                all_reg_predictions = all_reg_predictions.loc[all_reg_predictions.is_train == False]
                all_reg_predictions = all_reg_predictions[['is_train', 'sample_id', 'total_payoff_label', 'raisha',
                                                           'final_total_payoff_prediction']]
                write_to_excel(
                    table_writer, f'Model_{model_num}_reg',
                    headers=[f'Reg predictions for model {model_num}: {model_name} of type {model_type} in fold {fold}'],
                    data=all_reg_predictions)
                all_prediction = all_reg_predictions
            if not predict_avg_total_payoff and not predict_seq:
                print('Model type do not have avg or turn')
                return

            final_total_payoff_prediction = all_prediction[final_total_payoff_prediction_column]
            total_payoff_label = all_prediction[total_payoff_label_column]
            if type(total_payoff_label.values[0]) == str:  # tensor(value, cuda=...) instead of float
                total_payoff_label = all_prediction[['sample_id']].merge(
                    total_payoff_true_label, right_index=True, left_on='sample_id')[total_payoff_label_column]
                all_prediction = all_prediction.drop(total_payoff_label_column, axis=1).merge(
                    total_payoff_true_label, right_index=True, left_on='sample_id')
            # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
            bin_predictions, bin_label = create_bin_columns(final_total_payoff_prediction, total_payoff_label,
                                                            hotel_label_0=True)
            # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
            _, results_dict = calculate_measures_for_continues_labels(
                all_prediction, final_total_payoff_prediction_column=final_total_payoff_prediction_column,
                total_payoff_label_column=total_payoff_label_column,
                bin_label=bin_label, bin_predictions=bin_predictions,
                label_options=['total future payoff < 1/3', '1/3 < total future payoff < 2/3',
                               'total future payoff > 2/3'])

            if raisha_column_name in all_prediction.columns:  # do the raisha analysis
                raisha_options = all_prediction[raisha_column_name].unique()
                bin_predictions = pd.DataFrame(bin_predictions).join(all_prediction[[raisha_column_name]], how='left')
                bin_label = pd.DataFrame(bin_label).join(all_prediction[[raisha_column_name]], how='left')
                all_raisha_dict = defaultdict(dict)
                for raisha in raisha_options:
                    raisha_data = all_prediction.loc[all_prediction[raisha_column_name] == raisha]
                    raisha_bin_label = bin_label.loc[bin_label[raisha_column_name] == raisha]
                    raisha_bin_predictions = bin_predictions.loc[bin_predictions[raisha_column_name] == raisha]
                    # bin_label_raisha = bin_label.loc[]
                    _, results_dict_raisha = calculate_measures_for_continues_labels(
                        raisha_data, final_total_payoff_prediction_column=final_total_payoff_prediction_column,
                        total_payoff_label_column=total_payoff_label_column,
                        bin_label=raisha_bin_label['bin_label'],
                        bin_predictions=raisha_bin_predictions['bin_predictions'],
                        label_options=['total future payoff < 1/3', '1/3 < total future payoff < 2/3',
                                       'total future payoff > 2/3'], raisha=f'raisha_{str(int(raisha))}')
                    all_raisha_dict.update(results_dict_raisha)
                results_dict = update_default_dict(results_dict, all_raisha_dict)

            # measures per round
            if predict_seq:  # and not self.predict_avg_total_payoff:
                all_prediction = all_seq_predictions
                flat_seq_predictions = flat_seq_predictions_list_column(
                    prediction=all_prediction, label_column_name_per_round='labels',
                    prediction_column_name_per_round='final_prediction')
                write_to_excel(
                    table_writer, f'Model_{model_num}_per_round',
                    headers=[f'Per round predictions for model {model_num}: {model_name} of type {model_type} in fold '
                             f'{fold}'], data=flat_seq_predictions)
                label_options = ['DM chose hotel', 'DM chose stay home']
                results_dict_per_round = per_round_analysis(
                    flat_seq_predictions, predictions_column=per_round_predictions_name,
                    label_column=per_round_labels_name, label_options=label_options,
                    function_to_run='calculate_per_round_measures')
                results_dict = update_default_dict(results_dict, results_dict_per_round)

                if 'raisha' in flat_seq_predictions:
                    results_dict_per_round_per_raisha = per_round_analysis(
                        flat_seq_predictions, predictions_column=per_round_predictions_name,
                        label_column=per_round_labels_name, label_options=label_options,
                        function_to_run='calculate_per_round_per_raisha_measures')
                    results_dict = update_default_dict(results_dict, results_dict_per_round_per_raisha)

            results_df = pd.DataFrame.from_dict(results_dict).T
            results_df['raisha_round'] = results_df.index
            results_df[['Raisha', 'Round']] = results_df.raisha_round.str.split(expand=True)
            results_df = results_df.drop('raisha_round', axis=1)
            results_df.index = np.zeros(shape=(results_df.shape[0],))
            if 'hyper_parameters_dict' in metadata_df.columns:
                hyper_parameters = 'hyper_parameters_dict'
            else:
                hyper_parameters = 'hyper_parameters_str'

            metadata_df = pd.DataFrame(metadata_df[['model_num', 'model_type', 'model_name', 'data_file_name',
                                                    hyper_parameters]].iloc[0])
            metadata_df['fold'] = f'fold_{fold_num}'
            results_df = results_df.join(metadata_df.T)
            results_df.index = [f'{folder}_Results_{inner_folder}_model_{model_num}.xlsx'] * results_df.shape[0]
            write_to_excel(table_writer, 'Model results', ['Model results'], results_df)
            table_writer.save()
            all_models_results.append(results_df)

    if len(all_models_results) > 0:
        all_models_results_df = pd.concat(all_models_results, sort='False')
        return all_models_results_df
    else:
        return None


def concat_all_rounds_per_rounds_results(all_resutls_file_name: str, new_eval_all_rounds: pd.DataFrame):
    """
    Concat all results with the new eval of the all_rounds measures
    :param all_resutls_file_name:
    :param new_eval_all_rounds:
    :return:
    """
    all_resutls = pd.read_csv(os.path.join(base_directory, 'logs', all_resutls_file_name))
    folders_to_eval = new_eval_all_rounds.index.unique()
    all_resutls_not_LSTM = all_resutls.loc[~all_resutls.folder.isin(folders_to_eval)]
    # merge
    all_resutls_LSTM = all_resutls.loc[all_resutls.folder.isin(folders_to_eval)]
    all_resutls_LSTM = all_resutls_LSTM[['folder', 'Per_round_Accuracy', 'Per_round_Fbeta_score_DM chose hotel',
                                         'Per_round_Fbeta_score_DM chose stay home',
                                         'Per_round_precision_DM chose hotel', 'Per_round_precision_DM chose stay home',
                                         'Per_round_recall_DM chose hotel', 'Per_round_recall_DM chose stay home',
                                         'Raisha', 'Round', 'data_file_name', 'fold', 'hyper_parameters_str',
                                         'model_name', 'model_num', 'model_type']]
    new_eval_all_rounds['folder'] = new_eval_all_rounds.index
    new_eval_all_rounds = new_eval_all_rounds[[
        'folder', 'Bin_Accuracy', 'Bin_Fbeta_score_1/3 < total future payoff < 2/3',
        'Bin_Fbeta_score_total future payoff < 1/3', 'Bin_Fbeta_score_total future payoff > 2/3',
        'Bin_precision_1/3 < total future payoff < 2/3', 'Bin_precision_total future payoff < 1/3',
        'Bin_precision_total future payoff > 2/3', 'Bin_recall_1/3 < total future payoff < 2/3',
        'Bin_recall_total future payoff < 1/3', 'Bin_recall_total future payoff > 2/3', 'MAE', 'MSE', 'RMSE', 'Raisha',
        'Round']]

    all_resutls_LSTM = all_resutls_LSTM.merge(new_eval_all_rounds, on=['folder', 'Raisha', 'Round'], how='left')
    # join all results
    all_resutls = pd.concat([all_resutls_LSTM, all_resutls_not_LSTM], sort='False')

    all_results_file_name = 'all_results_03_06_new_eval_select_correct_final_pred.csv'
    all_resutls.to_csv(os.path.join(base_directory, 'logs', all_results_file_name))
    return all_results_file_name


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
    # final_results = pd.DataFrame()

    # for fold in range(6):
    #     total_payoff = pd.read_excel(join(base_directory, 'logs', 'compare_prediction_models_07_05_2020_11_49',
    #                                       f'fold_{fold}', 'excel_models_results',
    #                                       f'Results_fold_{fold}_model_29_0.1.xlsx'),
    #                                  sheet_name=f'Model_29_0.1_seq', skiprows=[0], index_col=0)
    #     total_payoff = total_payoff.loc[total_payoff.is_train == False]
    #     total_payoff = total_payoff.total_payoff_label
    #     # all_models.xslx: compare_prediction_models_04_05_2020_19_15, compare_prediction_models_25_05_2020_12_53
    #     curr_df = eval_model_total_prediction(['compare_prediction_models_04_05_2020_19_15',
    #                                            'compare_prediction_models_07_05_2020_11_49',
    #                                            'compare_prediction_models_03_05_2020_11_53',
    #                                            'compare_prediction_models_10_05_2020_11_02',
    #                                            'compare_prediction_models_11_05_2020_11_58',
    #                                            'compare_prediction_models_12_05_2020_14_51',
    #                                            'compare_prediction_models_13_05_2020_22_07',
    #                                            'compare_prediction_models_14_05_2020_17_46',
    #                                            'compare_prediction_models_16_05_2020_21_52',
    #                                            'compare_prediction_models_17_05_2020_22_12',
    #                                            'compare_prediction_models_19_05_2020_22_44',
    #                                            'compare_prediction_models_20_05_2020_17_39',
    #                                            'compare_prediction_models_25_04_2020_19_00',
    #                                            'compare_prediction_models_25_05_2020_14_04',
    #                                            'compare_prediction_models_26_05_2020_11_04',
    #                                            'compare_prediction_models_26_05_2020_18_17',
    #                                            'compare_prediction_models_29_04_2020_23_59'
    #                                            'compare_prediction_models_31_05_2020_12_35',
    #                                            'compare_prediction_models_29_05_2020_15_10',
    #                                            'compare_prediction_models_25_05_2020_12_53',
    #                                            'compare_prediction_models_01_06_2020_15_23',
    #                                            'compare_prediction_models_02_06_2020_13_18',
    #                                            'compare_prediction_models_30_05_2020_17_17'], fold, total_payoff)
    #     # curr_df = eval_model_total_prediction(['compare_prediction_models_22_04_2020_23_32'], fold, total_payoff)
    #     if curr_df is not None:  # if no data in this fold so curr_df = None
    #         final_results = pd.concat([final_results, curr_df], sort='False')
    #
    # final_results.to_csv(os.path.join(base_directory, 'logs', 'after_fix_bin_raisha',
    #                                   'total_prediction_results_check_no_train_select_correct_final_results_04_06.csv'))

    new_eval_all_rounds = list()
    for new_eval_file_name in ['total_prediction_results_check_no_train_select_correct_final_results_04_06.csv',
                               'total_prediction_results_check_no_train_select_correct_final_results_22_04.csv']:
        new_eval_all_rounds.append(pd.read_csv(os.path.join(base_directory, 'logs', 'after_fix_bin_raisha',
                                                            new_eval_file_name), index_col=0))

    final_results = pd.concat(new_eval_all_rounds, sort=False)
    final_results['folder'] = final_results.index

    final_results['fold'] = final_results.folder.str.extract(r"fold_(\d)")
    final_results['temp'] = 'fold'
    final_results['fold'] = final_results['temp'] + '_' + final_results['fold'].map(str)

    # all_results_file_name = concat_all_rounds_per_rounds_results('all_results_01_06.csv', new_eval_all_rounds)

    all_resutls = pd.read_csv(os.path.join(base_directory, 'logs', 'all_results_01_06_new_eval.csv'), index_col=0)
    folders_to_eval = final_results.index.unique()
    all_resutls_not_LSTM = all_resutls.loc[~all_resutls.folder.isin(folders_to_eval)]
    # merge
    all_resutls_LSTM = all_resutls.loc[all_resutls.folder.isin(folders_to_eval)]
    # join all results
    all_resutls = pd.concat([final_results, all_resutls_not_LSTM], sort='False')
    all_resutls.reset_index(inplace=True, drop=True)
    all_results_file_name = 'all_results_05_06_new_eval_select_correct_final_pred.csv'
    all_resutls.to_csv(os.path.join(base_directory, 'logs', all_results_file_name))

    analyze_model_results.main(new_results_file_name=all_results_file_name, results_to_use=all_resutls)

    # ray.init()
    # final_results = pd.DataFrame()
    # all_ready_lng = \
    #     ray.get([eval_model.remote(['compare_prediction_models_22_04_2020_23_32',
    #                                 'compare_prediction_models_24_04_2020_15_58'], i) for i in range(6)])

    # for curr_df in all_ready_lng:
    #     final_results = pd.concat([final_results, curr_df], sort='False')

    # final_results.to_csv(os.path.join(base_directory, 'logs', 'per_round_results.csv'))

    # combine_models_results(['compare_prediction_models_03_05_2020_11_53'])
    # combine_models_all_results(['compare_prediction_models_04_05_2020_19_15'])
    #
    # log_directory_main = '/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/' \
    #                      'language_prediction/logs/LSTMDatasetReader_100_epochs_5_folds_04_03_2020_11_46_40'
    # model_output_file_name_main = 'predictions.csv'
    # final_total_payoff_prediction_column_name_main = 'final_total_payoff_prediction'
    # total_payoff_label_column_name_main = 'total_payoff_label'
    # main(log_directory_main, model_output_file_name_main, final_total_payoff_prediction_column_name_main,
    #      total_payoff_label_column_name_main)
