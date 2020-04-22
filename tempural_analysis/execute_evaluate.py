__all__ = ['evaluate_grid_search', 'get_regression_final_results', 'get_classification_final_results',
           'plot_confusion_matrix', 'evaluate_backward_search']

import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from tempural_analysis import split_data, simulation
from sklearn import metrics
import tempural_analysis.utils as utils
import copy
import operator
import math


def evaluate_backward_search(data: pd.DataFrame, base_features: list, candidates_features: list, window_size_list: list,
                             label_dict: dict, num_folds: int, model_dict: dict, classifier_results_dir: str,
                             appendix: str, personal_features: list, model_type: str=None,
                             col_to_group: str='participant_code'):
    """
    Evaluate the model with a list of features, labels and window size
    :param data: the data to use
    :param base_features: the base features - all the features we can use
    :param candidates_features: the features to remove. if None- use all features
    :param window_size_list: the options to window size
    :param label_dict: the options to labels
    :param model_dict: list of models to evaluate
    :param num_folds: number of folds, if 1: run train-test
    :param classifier_results_dir: the directory to save the results
    :param appendix: the appendix of the columns (group/player)
    :param personal_features: list of personal features such as age and gender --> use them only in t=0
    :param col_to_group: the column to split the samples by (participant_code or pair)
    :param model_type: if we do classification or regression
    :return:
    """

    table_writer = pd.ExcelWriter(os.path.join(classifier_results_dir, 'classification_results.xlsx'),
                                  engine='xlsxwriter')
    running_meta_data = pd.DataFrame(
        columns=['model_name', 'model', 'features list', 'add feature', 'use first round',
                 'folds per participants', 'label', 'window size', 'accuracy', 'F1-score_pos', 'F1-score_neg',
                 'F1-score_third_bin', 'AUC', 'MSE', 'RMSE', 'MAE'])

    run_id = 0

    # start running the experiments
    for folds_per_participants in [True]:  # , False]:  # split participants between folds or not
        if folds_per_participants:
            # get the folds at the beginning before running any model
            if num_folds > 1:
                folds = split_data.get_folds_per_participant(data, col_to_group=col_to_group)
            else:
                # split data to 3 folds --> 2 will be train and 1 test
                folds = split_data.get_folds_per_participant(data, k_folds=3, col_to_group=col_to_group)
        else:
            # get the folds at the beginning before running any model
            if num_folds > 1:
                folds = split_data.get_folds(data, col_to_group=col_to_group)
            else:
                # split data to 3 folds --> 2 will be train and 1 test
                folds = split_data.get_folds(data, k_folds=3, col_to_group=col_to_group)

        for curr_label, label_options in label_dict.items():
            for curr_window_size in window_size_list:
                for use_first_round in [False]:
                    remaining_features = copy.deepcopy(candidates_features)
                    remaining_features.append('all')
                    selected_features = copy.deepcopy(base_features)
                    selected_features.append('all')
                    features_removed = list()
                    subsession_round_number_removed = False
                    current_accuracy, best_accuracy = 0.0, 0.0
                    remain_number_of_candidate = len(remaining_features)

                    while remaining_features and current_accuracy == best_accuracy and remain_number_of_candidate > 0:
                        score_with_candidates = list()
                        for candidate in remaining_features:
                            curr_features = copy.deepcopy(selected_features)
                            if candidate != 'all' and 'all' in curr_features:
                                curr_features.remove('all')
                            # if it is subsession_round_number: remove later
                            if candidate != 'subsession_round_number':
                                curr_features.remove(candidate)
                            # remove the candidate, also if it is None
                            if candidate is not None and None in curr_features:
                                curr_features.remove(None)

                            # check if there are still features to check
                            if len(curr_features) == 0:
                                logging.info(f'No features remained, the best accuracy is:{current_accuracy}')
                                remain_number_of_candidate = 0
                                continue
                            curr_personal_features = [feature for feature in personal_features if feature in
                                                      curr_features]
                            x, y, personal_features_data =\
                                split_data.get_data_label_personal(data, curr_label, curr_features,
                                                                   curr_personal_features)
                            # get the relevant models for the label (regression/classification)
                            model_list = model_dict[curr_label]
                            for curr_model in model_list:
                                run_id += 1
                                logging.info(f'Starting Run ID {run_id}: running model: {curr_model} \n with features:'
                                             f' {curr_features}, \nlabel: {curr_label}, '
                                             f'window size: {curr_window_size} and the candidate is: {candidate}')
                                simulator = simulation.Simulator(run_id, curr_window_size, curr_model, curr_label,
                                                                 curr_features, classifier_results_dir, candidate,
                                                                 subsession_round_number_removed,
                                                                 col_to_group=col_to_group)
                                try:
                                    if num_folds > 1:
                                        simulator.split_folds_run_model_cross_validation(
                                            x, y, folds, appendix, personal_features_data, use_first_round)
                                    else:
                                        simulator.run_model_train_test(run_id, x, y, folds, appendix,
                                                                       personal_features_data, use_first_round)

                                except Exception as e:
                                    logging.exception(f'Exception while running Run ID {run_id}: {e}')
                                    continue

                                if curr_label == appendix + '_receiver_choice' or model_type == 'classification':
                                    accuracy, auc, f1_score_pos, f1_score_neg =\
                                        get_classification_final_results(simulator, classifier_results_dir,
                                                                         label_options, table_writer, run_id)
                                    features_to_save = copy.deepcopy(curr_features)
                                    # remove subsession_round_number from the features we save
                                    if subsession_round_number_removed:
                                        features_to_save.remove('subsession_round_number')
                                    running_meta_data.loc[run_id] =\
                                        [type(curr_model).__name__, str(curr_model), features_to_save,
                                         len(features_to_save), candidate, copy.deepcopy(features_removed),
                                         use_first_round, folds_per_participants, curr_label, curr_window_size,
                                         accuracy, round(100*f1_score_pos, 2), round(100*f1_score_neg, 2),
                                         '-', auc, '-', '-', '-']
                                    score_with_candidates.append((accuracy, candidate))
                                    logging.info(
                                        f'Finish Run ID {run_id}: accuracy is {accuracy} and auc is {auc}')

                                else:  # regression
                                    features_to_save = copy.deepcopy(curr_features)
                                    # remove subsession_round_number from the features we save
                                    if subsession_round_number_removed:
                                        features_to_save.remove('subsession_round_number')
                                    mse, rmse, mae = get_regression_final_results(simulator, run_id, table_writer)
                                    accuracy, auc, f1_score_0, f1_score_1, f1_score_2 = \
                                        get_classification_final_results(simulator, classifier_results_dir,
                                                                         [0, 1, 2], table_writer, run_id,
                                                                         label_column='bin_label',
                                                                         predictions_column='bin_predictions')
                                    running_meta_data.loc[run_id] =\
                                        [type(curr_model).__name__, str(curr_model), features_to_save,
                                         len(features_to_save), candidate, copy.deepcopy(features_removed),
                                         use_first_round, folds_per_participants, curr_label, curr_window_size,
                                         accuracy, round(100*f1_score_0, 2), round(100*f1_score_1, 2),
                                         round(100 * f1_score_2, 2), auc, mse, rmse, mae]
                                    score_with_candidates.append((mse, candidate))

                        score_with_candidates.sort(key=operator.itemgetter(0))
                        best_accuracy, best_candidate = score_with_candidates.pop()
                        if current_accuracy <= best_accuracy:
                            # don't remove subsession_round_number from the features, even if we need to remove it
                            if best_candidate == 'subsession_round_number':
                                subsession_round_number_removed = True
                            else:
                                remaining_features.remove(best_candidate)
                                selected_features.remove(best_candidate)
                            features_removed.append(best_candidate)
                            current_accuracy = best_accuracy
                            logging.info(f'the feature {best_candidate} was removed from selected features '
                                         f'with accuracy without it {best_accuracy}')
                        else:
                            logging.info(f'No candidate was chosen, number of selected features is '
                                         f'{len(selected_features)}.')

                        # one candidate can be chosen, if not- we go forward to the next step.
                        remain_number_of_candidate -= 1
                        if None in remaining_features:  # after the first loop, no need to check all the features again
                            remaining_features.remove(None)
                            selected_features.remove(None)

    if subsession_round_number_removed:
        selected_features.remove('subsession_round_number')
    logging.info(f'Selected features are: {selected_features} and the best accuracy is: {current_accuracy}')
    running_meta_data.index.name = 'run_id'
    utils.write_to_excel(table_writer, 'meta data', ['Experiment meta data'], running_meta_data)
    logging.info('Save results')
    table_writer.save()

    return


def evaluate_grid_search(data: pd.DataFrame, base_features: list, add_feature_list: list, window_size_list: list,
                         label_dict: dict, num_folds: int, model_dict: dict, classifier_results_dir: str, appendix: str,
                         personal_features: list, model_type: str=None, col_to_group: str='participant_code'):
    """
    Evaluate the model with a list of features, labels and window size
    :param data: the data to use
    :param base_features: the base features - all the features we can use
    :param add_feature_list: the features to remove. if None- use all features
    :param window_size_list: the options to window size
    :param label_dict: the options to labels
    :param model_dict: list of models to evaluate
    :param num_folds: number of folds, if 1: run train-test
    :param classifier_results_dir: the directory to save the results
    :param appendix: the appendix of the columns (group/player)
    :param personal_features: list of personal features such as age and gender --> use them only in t=0
    :param model_type: if we do classification or regression
    :param col_to_group: the column to split the samples by (participant_code or pair)
    :return:
    """

    table_writer = pd.ExcelWriter(os.path.join(classifier_results_dir, 'classification_results.xlsx'),
                                  engine='xlsxwriter')
    running_meta_data = pd.DataFrame(
        columns=['model_name', 'model', 'features list', 'add feature', 'use first round',
                 'folds per participants', 'label', 'window size', 'accuracy', 'F1-score_pos', 'F1-score_neg',
                 'F1-score_third_bin', 'AUC', 'MSE', 'RMSE', 'MAE'])

    run_id = 0

    # start running the experiments
    for folds_per_participants in [True]:  # , False]:  # split participants between folds or not
        if folds_per_participants:
            # get the folds at the beginning before running any model
            if num_folds > 1:
                folds = split_data.get_folds_per_participant(data, col_to_group=col_to_group)
            else:
                # split data to 3 folds --> 2 will be train and 1 test
                folds = split_data.get_folds_per_participant(data, k_folds=3, col_to_group=col_to_group)
        else:
            # get the folds at the beginning before running any model
            if num_folds > 1:
                folds = split_data.get_folds(data, col_to_group=col_to_group)
            else:
                # split data to 3 folds --> 2 will be train and 1 test
                folds = split_data.get_folds(data, k_folds=3, col_to_group=col_to_group)

        for curr_label, label_options in label_dict.items():
            for curr_window_size in window_size_list:
                for use_first_round in [False]:
                    curr_base_features = copy.deepcopy(base_features)
                    for feature_to_add in add_feature_list:
                        if feature_to_add == '':  # if there are no features to add
                            curr_features = curr_base_features
                        else:
                            curr_features = curr_base_features + feature_to_add
                        curr_personal_features = [feature for feature in personal_features if feature in curr_features]
                        x, y, personal_features_data =\
                            split_data.get_data_label_personal(data, curr_label, curr_features, curr_personal_features)
                        # get the relevant models for the label (regression/classification)
                        model_list = model_dict[curr_label]
                        for curr_model in model_list:
                            run_id += 1
                            logging.info(f'Starting Run ID {run_id}: running model: {curr_model} \n with features:'
                                         f' {curr_features}, \nlabel: {curr_label}, window size: {curr_window_size}')
                            simulator = simulation.Simulator(run_id, curr_window_size, curr_model, curr_label,
                                                             curr_features, classifier_results_dir,
                                                             col_to_group=col_to_group)
                            try:
                                if num_folds > 1:
                                    simulator.split_folds_run_model_cross_validation(
                                        x, y, folds, appendix, personal_features_data, use_first_round)
                                else:
                                    simulator.run_model_train_test(run_id, x, y, folds, appendix,
                                                                   personal_features_data, use_first_round)

                            except Exception as e:
                                logging.exception(f'Exception while running Run ID {run_id}: {e}')
                                continue

                            if curr_label == appendix + '_receiver_choice' or model_type == 'classification':
                                accuracy, auc, f1_score_pos, f1_score_neg =\
                                    get_classification_final_results(simulator, classifier_results_dir, label_options,
                                                                     table_writer, run_id)
                                running_meta_data.loc[run_id] =\
                                    [type(curr_model).__name__, str(curr_model), copy.deepcopy(curr_base_features),
                                     feature_to_add, use_first_round, folds_per_participants, curr_label,
                                     curr_window_size, accuracy, round(100*f1_score_pos, 2),
                                     round(100*f1_score_neg, 2), '-', auc, '-', '-', '-']

                            else:  # regression
                                mse, rmse, mae = get_regression_final_results(simulator, run_id, table_writer)
                                accuracy, auc, f1_score_0, f1_score_1, f1_score_2 = \
                                    get_classification_final_results(
                                        simulator, classifier_results_dir,
                                        ['total future payoff < 1/3', '1/3 < total future payoff < 2/3',
                                         'total future payoff > 2/3'],
                                        table_writer, run_id, label_column='bin_label', predictions_column='bin_predictions')
                                running_meta_data.loc[run_id] = [type(curr_model).__name__, str(curr_model),
                                                                 copy.deepcopy(curr_base_features), feature_to_add,
                                                                 use_first_round, folds_per_participants, curr_label,
                                                                 curr_window_size, accuracy,
                                                                 round(100*f1_score_0, 2), round(100*f1_score_1, 2),
                                                                 round(100 * f1_score_2, 2),
                                                                 auc, mse, rmse, mae]

    running_meta_data.index.name = 'run_id'
    utils.write_to_excel(table_writer, 'meta data', ['Experiment meta data'], running_meta_data)
    logging.info('Save results')
    table_writer.save()

    return


def get_regression_final_results(simulator: simulation, run_id: int, table_writer: pd.ExcelWriter) -> \
        (float, float, float):
    """
    create results DF for regressions model
    :param simulator: the simulator object of this run
    :param run_id: the running number of this run
    :param table_writer: the excel writer
    :return: score: the MSE score
    """
    all_predictions = pd.concat(simulator.predictions_per_fold.values(), axis=0)
    all_predictions.index.name = 'participant_id_trial_num'
    mse = metrics.mean_squared_error(all_predictions.label, all_predictions.predictions)
    rmse = round(100*math.sqrt(mse), 2)
    mae = round(100*metrics.mean_absolute_error(all_predictions.label, all_predictions.predictions), 2)
    mse = round(100*mse, 2)
    logging.info(f'Measures for run ID {run_id} are: MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

    model_name = type(simulator.model).__name__
    utils.write_to_excel(
        table_writer, str(run_id) + '_predictions',
        headers=[f'Regression all predictions for {simulator.label} and model {model_name} and run ID {run_id}',
                 f'Measure for run ID {run_id} are: MSE: {mse}, RMSE: {rmse}, MAE: {mae}'], data=all_predictions)

    return mse, rmse, mae


def get_classification_final_results(simulator: simulation, classifier_results_dir: str, label_options: list,
                                     table_writer: pd.ExcelWriter, run_id: int, label_column: str='label',
                                     predictions_column: str='predictions'):
    """
    create results DF for regressions model
    :param simulator: the simulator object of this run
    :param classifier_results_dir: the directory to save the results
    :param label_options: the options for labels
    :param table_writer: the excel writer
    :param run_id: running number of the experiment
    :param label_column: the name of the label column
    :param predictions_column: the name of the predictions column
    :return: accuracy and AUC score
    """
    all_predictions = pd.concat(simulator.predictions_per_fold.values(), axis=0)
    all_predictions.index.name = 'participant_id_trial_num'
    model_name = type(simulator.model).__name__

    precision, recall, fbeta_score, support =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column])

    # f1_score_pos = metrics.f1_score(all_predictions[label_column], all_predictions[predictions_column])
    # f1_score_neg = metrics.f1_score(all_predictions[label_column], all_predictions[predictions_column], pos_label=-1)

    # number of DM chose stay home
    if label_column == 'label':
        status_size = all_predictions[label_column].where(all_predictions[label_column] == -1).dropna().shape[0]
        index_in_support = np.where(support == status_size)
        if index_in_support[0][0] == 0:  # the first in support is the label -1
            label_options = label_options
            f_score_list = [fbeta_score[0], fbeta_score[1]]
        else:
            label_options = [label_options[1], label_options[0]]
            f_score_list = [fbeta_score[1], fbeta_score[0]]

    else:
        zero_status_size = all_predictions[label_column].where(all_predictions[label_column] == 0).dropna().shape[0]
        one_status_size = all_predictions[label_column].where(all_predictions[label_column] == 1).dropna().shape[0]
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

    accuracy = metrics.accuracy_score(all_predictions[label_column], all_predictions[predictions_column])
    accuracy = round(100*accuracy, 2)
    accuracy_pd = pd.Series([accuracy, '-'])
    if label_column == 'labe':  # 2 classes
        auc = metrics.roc_auc_score(all_predictions[label_column], all_predictions[predictions_column])
        auc = round(100*auc, 2)
        auc_pd = pd.Series([auc, '-'])
    else:
        auc = None
        auc_pd = pd.Series(['-', '-'])

    results = pd.concat([pd.Series(100*precision).round(2), pd.Series(100*recall).round(2),
                         pd.Series(100*fbeta_score).round(2),
                         # pd.Series(100*f1_score_pos).round(2), pd.Series(100*f1_score_neg).round(2),
                         accuracy_pd, auc_pd], axis=1).T
    results.index = ['precision', 'recall', 'fbeta_score',
                     # 'f1_score_pos', 'f1_score_neg',
                     'accuracy', 'AUC']
    results.columns = label_options

    # save measures and all predictions to excel
    utils.write_to_excel(
        table_writer, str(run_id) + '_measures',
        headers=[f'Classifier measures for {simulator.label} and model {model_name} and run ID {run_id}'], data=results)
    if label_column == 'label':  # classification case
        utils.write_to_excel(
            table_writer, str(run_id) + '_predictions',
            headers=[f'Classifier all predictions for {simulator.label} and model {model_name} and run ID {run_id}'],
            data=all_predictions)

    logging.info(f'Classifier measures for {simulator.label} and model {model_name} and run ID {run_id} are:')
    logging.info(
        # f'f1_score_neg: {round(100*f1_score_neg, 2)}, f1_score_pos: {round(100*f1_score_pos, 2)}, '
        f'f_score: {f_score_list}, accuracy: {accuracy}, AUC: {auc}')
    # plot confusion matrix
    fig = plt.figure()
    title = f'Confusion matrix {run_id}'
    plot_confusion_matrix(simulator.total_conf_mtx, classes=label_options, title=title)
    fig_to_save = fig
    fig_to_save.savefig(os.path.join(classifier_results_dir, title + '.png'), bbox_inches='tight')

    if label_column == 'label':
        return accuracy, auc, f_score_list[0], f_score_list[1]
    else:
        return accuracy, auc, f_score_list[0], f_score_list[1], f_score_list[2]


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Normalized confusion matrix")
    else:
        logging.info('Confusion matrix')

    logging.info(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(int(cm[i, j]), fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return
