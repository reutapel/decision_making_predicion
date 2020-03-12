import pandas as pd
import tempural_analysis.utils as utils
import os
import argparse
import logging
from language_prediction.crf import LinearChainCRF
from sklearn import metrics
import math
import joblib
import language_prediction.SVM_models as SVM_models
from tempural_analysis.execute_evaluate import get_classification_final_results, get_regression_final_results


class ExecuteEvalModel:
    """
    This is a father class for all models we want to compare.
    Load data, split to train-validation based on the fold_split_dict.
    Train model on train data.
    Predict on validation data.
    Evaluate model on validation data.
    Save model, log, return the evaluation.
    """
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str):

        self.model_num = model_num
        self.fold = fold
        self.fold_dir = fold_dir
        self.model_type = model_type
        self.model_name = model_name
        self.data_file_name = data_file_name
        self.fold_split_dict = fold_split_dict
        self.table_writer = table_writer
        self.model_dir = utils.set_folder(f'model_num_{model_num}_type_{model_type}_name_{model_name}', fold_dir)
        self.model = None
        self.prediction = pd.DataFrame()
        self.data_directory = data_directory
        self.train_pair_ids = self.fold_split_dict['train']
        self.val_pair_ids = self.fold_split_dict['validation']
        print(f'Create Model: model num: {model_num}, model_type: {model_type}, model_name: {model_name}.'
              f'Data file name: {data_file_name}')
        logging.info(f'Create Model: model num: {model_num}, model_type: {model_type}, model_name: {model_name}.'
                     f'Data file name: {data_file_name}')

    def load_data_create_model(self):
        """This function should load the data, split to train-validation and create the model"""
        raise NotImplementedError

    def fit_predict(self):
        """This function should fit the model on the train data, predict on the validation data and dave the results"""
        raise NotImplementedError

    def eval_model(self):
        """This function should use the prediction of the model and eval these results"""
        raise NotImplementedError

    def save_model_prediction(self):
        utils.write_to_excel(self.table_writer, f'{self.model_name}_predictions',
                             headers=[f'All predictions for model {self.model_num}: {self.model_name} '
                                      f'of type {self.model_type} in fold {self.fold}'], data=self.prediction)


class ExecuteEvalCRF(ExecuteEvalModel):
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str, hyper_parameters_dict: dict):
        super(ExecuteEvalCRF, self).__init__(model_num, fold, fold_dir, model_type, model_name, data_file_name,
                                             fold_split_dict, table_writer, data_directory)
        self.squared_sigma = hyper_parameters_dict['squared_sigma']
        self.predict_future = hyper_parameters_dict['self.predict_future']
        self.use_forward_backward_fix_history = hyper_parameters_dict['use_forward_backward_fix_history']
        self.use_viterbi_fix_history = hyper_parameters_dict['use_viterbi_fix_history']
        self.predict_only_last = hyper_parameters_dict['predict_only_last']
        features_file_name = hyper_parameters_dict['features_file']
        self.model = None
        parser = argparse.ArgumentParser()
        parser.add_argument('train_data_file', help="data file for training input")
        parser.add_argument('features_file', help="the features file name input.")
        parser.add_argument('model_file', help="the model file name. (output)")
        self.args = parser.parse_args([
                os.path.join(self.data_directory, self.data_file_name),
                os.path.join(self.data_directory, features_file_name),
                os.path.join(self.model_dir, 'crf_model.pkl'),
            ])
        self.correct_count = None
        self.total_count = None
        self.total_seq_count = None

    def load_data_create_model(self):
        print(f'Data file name: {self.args.train_data_file}')
        logging.info(f'Data file name: {self.args.train_data_file}')
        self.model = LinearChainCRF(squared_sigma=self.squared_sigma, predict_future=self.predict_future)

    def fit_predict(self):
        self.model.train(self.args.train_data_file, self.args.model_file, self.args.features_file,
                         vector_rep_input=True, pair_ids=self.train_pair_ids,
                         use_forward_backward_fix_history=self.use_forward_backward_fix_history)

        self.model.load(self.args.model_file, self.args.features_file, vector_rep_input=True)
        self.correct_count, self.total_count, self.prediction, self.total_seq_count = \
            self.model.test(self.args.train_data_file, predict_only_last=self.predict_only_last,
                            pair_ids=self.val_pair_ids, use_viterbi_fix_history=self.use_viterbi_fix_history)
        self.save_model_prediction()

    def eval_model(self):
        try:
            if 'total_payoff_label' in self.prediction.columns and 'total_payoff_prediction' in self.prediction.columns:
                mse = metrics.mean_squared_error(self.prediction.total_payoff_label,
                                                 self.prediction.total_payoff_prediction)
                rmse = math.sqrt(mse)
                mae = metrics.mean_absolute_error(self.prediction.total_payoff_label,
                                                  self.prediction.total_payoff_prediction)

                # TODO: add bin analysis, per round measures

                return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

        except Exception:
            logging.exception(f'total_payoff_label or total_payoff_prediction not in CRF prediction DF')
            return


class ExecuteEvalSVM(ExecuteEvalModel):
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str, hyper_parameters_dict: dict):
        super(ExecuteEvalSVM, self).__init__(model_num, fold, fold_dir, model_type, model_name, data_file_name,
                                             fold_split_dict, table_writer, data_directory)
        self.label_name = hyper_parameters_dict['label_name']
        self.train_x, self.train_y, self.validation_x, self.validation_y = None, None, None, None

    def load_data_create_model(self):
        if 'pkl' in self.data_file_name:
            data = joblib.load(os.path.join(self.data_directory, self.data_file_name))
        else:
            data = pd.read_csv(os.path.join(self.data_directory, self.data_file_name))

        # get the feature columns
        x_columns = data.columns.tolist()
        x_columns.remove(self.label_name)
        x_columns.remove('participant_code')
        # get train data
        train_data = data.loc[data.participant_code.isin(self.train_pair_ids)]
        self.train_y = train_data[self.label_name]
        self.train_x = train_data[x_columns]
        # get validation data
        validation_data = data.loc[data.participant_code.isin(self.val_pair_ids)]
        self.validation_y = validation_data[self.label_name]
        self.validation_x = validation_data[x_columns]

        # create model
        self.model = getattr(SVM_models, self.model_type)

    def fit_predict(self):
        self.model.fit(self.train_x, self.train_y)
        self.prediction = self.model.predict(self.validation_x)
        self.save_model_prediction()

    def eval_model(self):
        # TODO: think what to do here with both SVM models and change get_regression_final_results and
        # get_classification_final_results accordingly
        mse, rmse, mae = get_regression_final_results(simulator, run_id, table_writer)
        accuracy, auc, f1_score_0, f1_score_1, f1_score_2 = \
            get_classification_final_results(
                simulator, classifier_results_dir,
                ['total future payoff < 1/3', '1/3 < total future payoff < 2/3',
                 'total future payoff > 2/3'],
                table_writer, run_id, label_column='bin_label', predictions_column='bin_predictions')

