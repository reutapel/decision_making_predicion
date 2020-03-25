import pandas as pd
import ray
import logging
import os
import json
import language_prediction.execute_cv_models as execute_cv_models
import tempural_analysis.utils as utils
from datetime import datetime


# define directories
base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
data_directory = os.path.join(base_directory, 'data', condition)

all_model_types = ['regression_SVM', 'SVM_turn', 'LSTM_turn', 'LSTM_avg', 'LSTM_avg_turn', 'CRF']


@ray.remote
def execute_fold_parallel(participants_fold: pd.Series, fold: int):
    """
    This function get a dict that split the participant to train-val-test (for this fold) and run all the models
    we want to compare --> it train them using the train data and evaluate them using the val data
    :param participants_fold: split the participant to train-val-test (for this fold)
    :param fold: the fold number
    :return:
    """
    # get the train, test, validation participant code for this fold
    fold_split_dict = dict()
    for data_set in ['train', 'test', 'validation']:
        fold_split_dict[data_set] = participants_fold.loc[participants_fold == data_set].index.tolist()

    # models_to_compare should have for each row:
    # model_num, model_type, model_name, function_to_run, data_file_name, hyper_parameters
    # (strings of all parameters for the running function as dict: {'parameter_name': parameter_value})
    models_to_compare = pd.read_excel(base_directory, 'models_to_compare.xlsx', sheet_name='table_to_load')
    fold_dir = utils.set_folder(datetime.now().strftime(f'fold_{fold}_%d_%m_%Y_%H_%M'), 'logs')
    table_writer = pd.ExcelWriter(os.path.join(fold_dir, f'classification_results_fold_{fold}.xlsx'),
                                  engine='xlsxwriter')
    log_file_name = os.path.join(fold_dir, 'LogFile.log')
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )

    for model_type in all_model_types:  # compare all versions of each model type
        model_type_versions = models_to_compare.loc[models_to_compare.model_type == model_type]
        for index, row in model_type_versions.iterrows():  # iterate over all the models to compare
            # get all model parameters
            model_num = row['model_num']
            model_type = row['model_type']
            model_name = row['model_name']
            function_to_run = row['function_to_run']
            data_file_name = row['data_file_name']
            hyper_parameters_str = row['hyper_parameters']
            # get hyper parameters as dict
            hyper_parameters_dict = json.loads(hyper_parameters_str)
            # each function need to get: model_num, fold, fold_dir, model_type, model_name, data_file_name,
            # fold_split_dict, table_writer, data_directory, hyper_parameters_dict.
            # During running it needs to write the predictions to the table_writer and save the trained model with
            # the name: model_name_model_num to the fold_dir.
            # it needs to return a dict with the final results over the evaluation data: {measure_name: measure}
            model_class = getattr(execute_cv_models, function_to_run)(
                model_num, fold, fold_dir, model_type, model_name, data_file_name, fold_split_dict, table_writer,
                data_directory, hyper_parameters_dict)
            model_class.load_data_create_model()
            model_class.fit_predict()
            results_dict = model_class.eval_model()

    # utils.write_to_excel(table_writer, 'meta data', ['Experiment meta data'], model_results)

    return f'fold {fold} finish compare models'


def parallel_main():
    ray.init()
    print(f'Start run in parallel: for each fold compare all the models')
    logging.info(f'Start run in parallel: for each fold compare all the models')

    # participants_fold_split should have the following columns: fold_0, fold_1,...,fold_5
    # the index should be the participant code
    # the values will be train/test/validation
    participants_fold_split = pd.read_csv(data_directory, 'pairs_folds.csv')

    all_ready_lng =\
        ray.get([execute_fold_parallel.remote(participants_fold_split[f'fold_{i}'], i) for i in range(6)])

    print(f'Done! {all_ready_lng}')
    logging.info(f'Done! {all_ready_lng}')

    return
