__all__ = ['Simulator']
import copy
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tempural_analysis.split_data import get_data_label_window_per_folds
import joblib
import os
import tempural_analysis.utils as utils


class Simulator:
    def __init__(self, run_id, window_size, model, label, features, classifier_results_dir, candidate=None,
                 subsession_round_number_removed=None, col_to_group: str='participant_code'):
        self.run_id = run_id
        self.window_size = window_size
        self.label = label
        self.features = features
        self.model = model
        self.fs_per_fold = dict()
        self.model_per_fold = dict()
        self.predictions_per_fold = dict()
        self.explanations_per_fold = dict()
        self.total_conf_mtx = None
        self.reset_confusion_matrix()
        self.classifier_results_dir = classifier_results_dir
        self.candidate = candidate
        self.subsession_round_number_removed = subsession_round_number_removed
        self.col_to_group = col_to_group

    def reset_confusion_matrix(self):
        self.total_conf_mtx = np.zeros([2, 2])

    def execute_single_experiment(self, experiment_id: int, train_x: pd.DataFrame, train_y: pd.Series,
                                  test_x: pd.DataFrame, test_y: pd.Series, test_folds: pd.DataFrame=None):
        """
        This function get train and test data and labels and execute the model
        :param experiment_id: running number
        :param train_x: train features
        :param train_y: train labels
        :param test_x: test features
        :param test_folds: test fold numbers (option - for cross validation)
        :param test_y: test labels
        :return:
        """

        trained_models_directory_path = utils.set_folder('trained_models',
                                                         father_folder_path=self.classifier_results_dir)
        logging.info('Starting model fit and save model')
        self.model_per_fold[experiment_id] = copy.deepcopy(self.model).fit(train_x, train_y)
        joblib.dump(self.model_per_fold[experiment_id], os.path.join(
            trained_models_directory_path, type(self.model).__name__ + str(experiment_id) + '.pkl'))
        logging.info('Starting model predict')
        # if hasattr(self.model_per_fold[experiment_id].__class__, 'predict_proba'):
        #     predictions = self.model_per_fold[experiment_id].predict_proba(test_x)
        # else:
        predictions = self.model_per_fold[experiment_id].predict(test_x)

        if type(predictions) == Exception:
            return predictions

        test_y.name = 'label'
        train_y.name = 'label'
        if predictions.dtype == float:  # regression- create bins to measure the F-score
            # for prediction
            keep_mask = predictions < 0.33
            bin_prediction = np.where(predictions < 0.67, 1, 2)
            bin_prediction[keep_mask] = 0
            bin_prediction = pd.Series(bin_prediction, name='bin_predictions', index=test_y.index)
            # for test_y
            keep_mask = test_y < 0.33
            bin_test_y = np.where(test_y < 0.67, 1, 2)
            bin_test_y[keep_mask] = 0
            bin_test_y = pd.Series(bin_test_y, name='bin_label', index=test_y.index)
        else:
            bin_prediction, bin_test_y = pd.Series(name='bin_prediction'), pd.Series(name='bin_label')
        if test_folds is not None:
            self.predictions_per_fold[experiment_id] =\
                test_folds.join(test_y).join(pd.Series(predictions, name='predictions', index=test_y.index)).join(
                    bin_test_y).join(bin_prediction)
        else:  # for train_test
            self.predictions_per_fold[experiment_id] = \
                pd.concat([test_y, pd.Series(predictions, name='predictions', index=test_y.index)], axis=1)

        # add correct column:
        self.predictions_per_fold[experiment_id]['correct'] = \
            np.where(self.predictions_per_fold[experiment_id].predictions ==
                     self.predictions_per_fold[experiment_id].label, 1, 0)
        self.predictions_per_fold[experiment_id]['bin_correct'] = \
            np.where(self.predictions_per_fold[experiment_id].bin_predictions ==
                     self.predictions_per_fold[experiment_id].bin_label, 1, 0)

        # create confusion matrix for classification
        if predictions.dtype == 'int':
            cnf_matrix = confusion_matrix(test_y, predictions)
            self.total_conf_mtx += cnf_matrix
        # else:
        #     cnf_matrix = confusion_matrix(bin_test_y, bin_prediction)
        #     self.total_conf_mtx += cnf_matrix

        return

    def run_model_train_test(self, run_id: int, x: pd.DataFrame, y: pd.Series, folds: pd.DataFrame, appendix: str,
                             personal_features: pd.DataFrame, use_first_round: bool):
        """
        This function split the data to train and test and execute the model
        :param run_id: the running number
        :param x: the features
        :param y: the labels
        :param folds: the fold number for each sample
        :param personal_features: data frame with the personal features of the participants
        :param appendix: the appendix of the columns (group/player)
        :param use_first_round: do we want to use the first round in all samples
        :return:
        """

        # split to train_test not the same user in the both train and test
        logging.info(f'Starting train test')
        try:
            train_x, train_y, test_x, test_y, test_folds =\
                get_data_label_window_per_folds(x=x, y=y, folds=folds, personal_features_data=personal_features,
                                                window_size=self.window_size, appendix=appendix, label=self.label, k=1,
                                                use_first_round=use_first_round, candidate=self.candidate,
                                                col_to_group=self.col_to_group)
            test_x = test_x.astype(float)
            train_x = train_x.astype(float)
        except ValueError as e:
            logging.info(f'ValueError when split to train and test with error {e}')
        else:
            self.execute_single_experiment(run_id, train_x, train_y, test_x, test_y)
        return

    def run_model_cross_validation(self, x: pd.DataFrame, y: pd.Series, folds: pd.DataFrame):
        """
        This function run cross validation for the given model
        :param x: the features
        :param y: the labels
        :param folds: the fold number for each participant
        :return:
        """
        x = x.astype(float)
        for k in folds.fold_number.unique():
            logging.info(f'Starting fold {k}')
            train_x = x.loc[folds.fold_number != k]
            train_y = y[folds.fold_number != k]
            # train_folds = folds[folds.fold_number != k]
            test_x = x.loc[folds.fold_number == k]
            test_y = y[folds.fold_number == k]
            test_folds = folds[folds.fold_number == k]
            self.execute_single_experiment(k, train_x, train_y, test_x, test_y, test_folds)

        return

    def split_folds_run_model_cross_validation(self, x: pd.DataFrame, y: pd.Series, folds: pd.DataFrame, appendix: str,
                                               personal_features: pd.DataFrame, use_first_round: bool):
        """
        This function first split data to train and test based on the folds and then run the model
        :param x: the features
        :param y: the labels
        :param folds: the fold number for each sample
        :param personal_features: data frame with the personal features of the participants
        :param appendix: the appendix of the columns (group/player)
        :return:
        """

        for k in folds.fold_number.unique():
            logging.info(f'Starting fold {k}')
            train_x, train_y, test_x, test_y, test_folds =\
                get_data_label_window_per_folds(x=x, y=y, folds=folds, personal_features_data=personal_features,
                                                window_size=self.window_size, appendix=appendix, label=self.label, k=k,
                                                use_first_round=use_first_round, candidate=self.candidate,
                                                subsession_round_number_removed=self.subsession_round_number_removed,
                                                col_to_group=self.col_to_group)
            train_x = train_x.astype(float)
            test_x = test_x.astype(float)
            self.execute_single_experiment(k, train_x, train_y, test_x, test_y, test_folds)

        return
