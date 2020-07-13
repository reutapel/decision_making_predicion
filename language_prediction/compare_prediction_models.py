import pandas as pd
import ray
import numpy as np
import logging
import os
import json
import utils
from datetime import datetime
import execute_cv_models
from collections import defaultdict
import torch
import copy
import random

random.seed(123)

# define directories
base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
data_directory = os.path.join(base_directory, 'data', condition, 'cv_framework')
run_dir = utils.set_folder(datetime.now().strftime(f'compare_prediction_models_%d_%m_%Y_%H_%M'), 'logs')

os.environ['http_proxy'] = 'some proxy'
os.environ['https_proxy'] = 'some proxy'

lstm_gridsearch_params = [
    {'lstm_dropout': lstm_dropout, 'linear_dropout': lstm_dropout,
     'lstm_hidden_dim': hidden_size, 'num_layers': num_layers}
    for lstm_dropout in [0.0, 0.1, 0.2, 0.3]
    for hidden_size in [50, 80, 100, 200]
    for num_layers in [1, 2, 3]
]

avg_turn_gridsearch_params = [{'avg_loss': 0.5, 'turn_loss': 0.5}, {'avg_loss': 0.3, 'turn_loss': 0.7},
                              {'avg_loss': 0.3, 'turn_loss': 0.7}]

transformer_gridsearch_params = [
    {'num_encoder_layers': num_layers, 'feedforward_hidden_dim_prod': feedforward_hidden_dim_prod,
     'lstm_dropout': transformer_dropout, 'linear_dropout': transformer_dropout}
    for num_layers in [3, 4, 5, 6]
    for transformer_dropout in [0.0, 0.1, 0.2, 0.3]
    for feedforward_hidden_dim_prod in [0.5, 1, 2]
]

svm_gridsearch_params = [{'kernel': 'poly', 'degree': 3}, {'kernel': 'poly', 'degree': 5},
                         {'kernel': 'poly', 'degree': 8}, {'kernel': 'rbf', 'degree': ''},
                         {'kernel': 'linear', 'degree': ''}]

crf_gridsearch_params = [{'squared_sigma': squared_sigma} for squared_sigma in [0.005, 0.006, 0.007, 0.008]]


def execute_create_fit_predict_eval_model(
        function_to_run, model_num, fold, fold_dir, model_type, model_name, data_file_name, fold_split_dict,
        table_writer, hyper_parameters_dict, excel_models_results, all_models_results):
    metadata_dict = {'model_num': model_num, 'model_type': model_type, 'model_name': model_name,
                     'data_file_name': data_file_name, 'hyper_parameters_str': hyper_parameters_dict}
    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index').T
    model_class = getattr(execute_cv_models, function_to_run)(
        model_num, fold, fold_dir, model_type, model_name, data_file_name, fold_split_dict, table_writer,
        data_directory, hyper_parameters_dict, excel_models_results)
    model_class.load_data_create_model()
    model_class.fit_validation()
    results_dict = model_class.eval_model(predict_type='validation')
    results_df = pd.DataFrame.from_dict(results_dict).T
    results_df['raisha_round'] = results_df.index
    results_df[['Raisha', 'Round']] = results_df.raisha_round.str.split(expand=True)
    results_df = results_df.drop('raisha_round', axis=1)
    results_df.index = np.zeros(shape=(results_df.shape[0],))
    results_df = metadata_df.join(results_df)
    all_models_results = pd.concat([all_models_results, results_df], sort='False')
    utils.write_to_excel(model_class.model_table_writer, 'Model results', ['Model results'], results_df)
    model_class.model_table_writer.save()
    del model_class

    return all_models_results


@ray.remote
def execute_fold_parallel(participants_fold: pd.Series, fold: int, cuda_device: str,
                          hyper_parameters_tune_mode: bool=False):
    """
    This function get a dict that split the participant to train-val-test (for this fold) and run all the models
    we want to compare --> it train them using the train data and evaluate them using the val data
    :param participants_fold: split the participant to train-val-test (for this fold)
    :param fold: the fold number
    :param cuda_device: the number of cuda device if using it
    :param hyper_parameters_tune_mode: after find good data - hyper parameter tuning
    :return:
    """
    # get the train, test, validation participant code for this fold
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    fold_split_dict = dict()
    for data_set in ['train', 'test', 'validation']:
        fold_split_dict[data_set] = participants_fold.loc[participants_fold == data_set].index.tolist()

    # models_to_compare should have for each row:
    # model_num, model_type, model_name, function_to_run, data_file_name, hyper_parameters
    # (strings of all parameters for the running function as dict: {'parameter_name': parameter_value})
    models_to_compare = pd.read_excel(os.path.join(base_directory, 'models_to_hyper_parameters.xlsx'),
                                      sheet_name='table_to_load', skiprows=[0])
    fold_dir = utils.set_folder(f'fold_{fold}', run_dir)
    excel_models_results = utils.set_folder(folder_name='excel_models_results', father_folder_path=fold_dir)
    # table_writer = pd.ExcelWriter(os.path.join(excel_models_results, f'Results_fold_{fold}_all_models.xlsx'),
    #                               engine='xlsxwriter')
    table_writer = None
    log_file_name = os.path.join(fold_dir, f'LogFile_fold_{fold}.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )
    # all_model_types = models_to_compare.model_type.unique()
    # all_model_types = ['LSTM_avg', 'LSTM_avg_turn', 'Transformer_avg_turn', 'Transformer_avg',
    #                    'LSTM_avg_turn_linear', 'Attention_avg']
    # all_model_types = ['Attention_avg']
    all_model_nums = list(set(models_to_compare.model_num))
    # already_trained_models = list(range(15, 21)) + list(range(11))
    # all_model_nums = [x for x in all_model_nums if x not in already_trained_models]
    all_model_nums = [30, 31, 44, 45]

    all_models_results = pd.DataFrame()
    for model_num in all_model_nums:  # compare all versions of each model type
        # if model_num < 42:
        #     continue
        model_type_versions = models_to_compare.loc[models_to_compare.model_num == model_num]
        for index, row in model_type_versions.iterrows():  # iterate over all the models to compare
            # get all model parameters
            model_type = row['model_type']
            model_name = row['model_name']
            function_to_run = row['function_to_run']
            data_file_name = row['data_file_name']
            hyper_parameters_str = row['hyper_parameters']
            # get hyper parameters as dict
            if type(hyper_parameters_str) == str:
                hyper_parameters_dict = json.loads(hyper_parameters_str)
            else:
                hyper_parameters_dict = None

            if hyper_parameters_dict is not None and 'features_max_size' in hyper_parameters_dict.keys():
                if int(hyper_parameters_dict['features_max_size']) > 1000:
                    continue

            # each function need to get: model_num, fold, fold_dir, model_type, model_name, data_file_name,
            # fold_split_dict, table_writer, data_directory, hyper_parameters_dict.
            # During running it needs to write the predictions to the table_writer and save the trained model with
            # the name: model_name_model_num to the fold_dir.
            # it needs to return a dict with the final results over the evaluation data: {measure_name: measure}
            if hyper_parameters_tune_mode:
                if 'LSTM' in model_type or 'Transformer' in model_type:
                    if 'LSTM' in model_type and not 'use_transformer' in model_type:
                        greadsearch = lstm_gridsearch_params
                    else:  # for Transformer models and LSTM_use_transformer models
                        greadsearch = transformer_gridsearch_params
                    for i, parameters_dict in enumerate(greadsearch):
                        # if i > 0:
                        #     continue
                        if (fold == 0 and (model_num == 30 or (model_num == 31 and i <= 25))) or \
                                (fold == 1 and ((model_num == 30) or (model_num == 31 and i <= 25))) or \
                                (fold == 2 and ((model_num == 30) or (model_num == 31 and i <= 17))) or \
                                (fold == 3 and ((model_num == 30) or (model_num == 31 and i <= 25))) or \
                                (fold == 4 and (model_num == 30 and i <= 46)) or \
                                (fold == 5 and (model_num == 30 and i <= 38)):
                            continue
                        new_hyper_parameters_dict = copy.deepcopy(hyper_parameters_dict)
                        new_hyper_parameters_dict.update(parameters_dict)
                        if 'linear' in model_type and 'lstm_hidden_dim' in new_hyper_parameters_dict:
                            new_hyper_parameters_dict['linear_hidden_dim'] =\
                                int(0.5*new_hyper_parameters_dict['lstm_hidden_dim'])
                        if '_avg_turn' in model_type:
                            for inner_i, inner_parameters_dict in enumerate(avg_turn_gridsearch_params):
                                # if inner_i > 0:
                                #     break
                                new_hyper_parameters_dict.update(inner_parameters_dict)
                                new_model_name = f'{model_name}'
                                new_model_num = f'{model_num}_{i}_{inner_i}'
                                all_models_results = execute_create_fit_predict_eval_model(
                                    function_to_run, new_model_num, fold, fold_dir, model_type, new_model_name,
                                    data_file_name, fold_split_dict, table_writer, new_hyper_parameters_dict,
                                    excel_models_results, all_models_results)
                        else:
                            new_model_name = f'{model_name}'
                            new_model_num = f'{model_num}_{i}'
                            all_models_results = execute_create_fit_predict_eval_model(
                                function_to_run, new_model_num, fold, fold_dir, model_type, new_model_name,
                                data_file_name, fold_split_dict, table_writer, new_hyper_parameters_dict,
                                excel_models_results, all_models_results)
                elif 'SVM' in model_type:
                    for i, parameters_dict in enumerate(svm_gridsearch_params):
                        # if i > 0:
                        #     continue
                        new_hyper_parameters_dict = copy.deepcopy(hyper_parameters_dict)
                        new_hyper_parameters_dict.update(parameters_dict)
                        new_model_name = f'{model_name}'
                        new_model_num = f'{model_num}_{i}'
                        all_models_results = execute_create_fit_predict_eval_model(
                            function_to_run, new_model_num, fold, fold_dir, model_type, new_model_name, data_file_name,
                            fold_split_dict, table_writer, new_hyper_parameters_dict, excel_models_results,
                            all_models_results)
                elif 'CRF' in model_type:
                    for i, parameters_dict in enumerate(crf_gridsearch_params):
                        # if i > 0:
                        #     continue
                        new_hyper_parameters_dict = copy.deepcopy(hyper_parameters_dict)
                        new_hyper_parameters_dict.update(parameters_dict)
                        new_model_name = f'{model_name}'
                        new_model_num = f'{model_num}_{i}'
                        all_models_results = execute_create_fit_predict_eval_model(
                            function_to_run, new_model_num, fold, fold_dir, model_type, new_model_name, data_file_name,
                            fold_split_dict, table_writer, new_hyper_parameters_dict, excel_models_results,
                            all_models_results)
                else:
                    print('Model type must be LSTM-kind, Transformer-kind, CRF-kind or SVM-kind')

            else:
                all_models_results = execute_create_fit_predict_eval_model(
                    function_to_run, model_num, fold, fold_dir, model_type, model_name, data_file_name, fold_split_dict,
                    table_writer, hyper_parameters_dict, excel_models_results, all_models_results)

    utils.write_to_excel(table_writer, 'All models results', ['All models results'], all_models_results)
    if table_writer is not None:
        table_writer.save()

    logging.info(f'fold {fold} finish compare models')
    print(f'fold {fold} finish compare models')

    return f'fold {fold} finish compare models'


def parallel_main():
    print(f'Start run in parallel: for each fold compare all the models')
    logging.info(f'Start run in parallel: for each fold compare all the models')

    # participants_fold_split should have the following columns: fold_0, fold_1,...,fold_5
    # the index should be the participant code
    # the values will be train/test/validation
    participants_fold_split = pd.read_csv(os.path.join(data_directory, 'pairs_folds.csv'))
    participants_fold_split.index = participants_fold_split.pair_id
    cuda_devices = {0: 0, 1: 1,
                    2: 0, 3: 1,
                    4: 0, 5: 1}

    # cuda_devices = {0: 0, 1: 0,
    #                 2: 0, 3: 0,
    #                 4: 0, 5: 0}
    """For debug"""
    # participants_fold_split = participants_fold_split.iloc[:50]
    # for fold in range(6):
    #     execute_fold_parallel(participants_fold_split[f'fold_{fold}'], fold=fold, cuda_device='1',
    #                           hyper_parameters_tune_mode=True)

    ray.init()
    all_ready_lng =\
        ray.get([execute_fold_parallel.remote(participants_fold_split[f'fold_{i}'], i, str(cuda_devices[i]),
                                              hyper_parameters_tune_mode=True)
                 for i in range(6)])

    print(f'Done! {all_ready_lng}')
    logging.info(f'Done! {all_ready_lng}')

    # ray.init()
    # all_ready_lng_1 = \
    #     ray.get([execute_fold_parallel.remote(participants_fold_split[f'fold_{i}'], i, str(cuda_devices[i]),
    #                                           hyper_parameters_tune_mode=True)
    #              for i in range(2, 4)])
    #
    # print(f'Done! {all_ready_lng_1}')
    # logging.info(f'Done! {all_ready_lng_1}')

    # ray.init()
    # all_ready_lng_2 = \
    #     ray.get([execute_fold_parallel.remote(participants_fold_split[f'fold_{i}'], i, str(cuda_devices[i]),
    #                                           hyper_parameters_tune_mode=True)
    #              for i in range(4, 6)])
    #
    # print(f'Done! {all_ready_lng_2}')
    # logging.info(f'Done! {all_ready_lng_2}')

    return


if __name__ == '__main__':
    parallel_main()
