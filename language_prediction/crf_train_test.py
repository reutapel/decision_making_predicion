#!/usr/bin/env python

import argparse
from language_prediction.crf import LinearChainCRF
import os
import tempural_analysis.utils as utils
from datetime import datetime
from tempural_analysis.split_data import get_folds_per_participant
import joblib
import pandas as pd
import random
import logging
from sklearn import metrics
import math

base_directory = os.path.abspath(os.curdir)
model_name = 'verbal'  # 'chunking_small'
data_directory = os.path.join(base_directory, 'data', model_name)

random.seed(1)


def main(test=False, cv=False, test_chunking=False, use_forward_backward_fix_history=False,
         use_viterbi_fix_history=False, squared_sigma=0.0005, predict_only_last=True, predict_future=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('traindatafile', help="data file for training input")
    parser.add_argument('featuresfile', help="the features file name input.")
    parser.add_argument('modelfile', help="the model file name. (output)")
    parser.add_argument('testdatafile', help="data file for testing input")

    if test_chunking:
        vector_rep_input = False
        args = parser.parse_args([
            os.path.join(data_directory, 'train.txt'),
            os.path.join(data_directory, 'features_single_round_label_crf_manual_binary_features_verbal_data.xlsx'),
            os.path.join(model_directory, 'crf_model.pkl'),
            os.path.join(data_directory, 'test.txt'),
        ])

        crf = LinearChainCRF(squared_sigma=10.0)
        crf.train(args.traindatafile, args.modelfile, args.featuresfile, vector_rep_input=vector_rep_input)

        if test:
            crf.load(args.modelfile, args.featuresfile, vector_rep_input=vector_rep_input)
            crf.test(args.testdatafile, predict_only_last=False)

    else:
        vector_rep_input = True
        if not cv:
            args = parser.parse_args([
                os.path.join(data_directory,
                             'train_data_1_10_single_round_label_crf_manual_binary_features_verbal_data.pkl'),
                os.path.join(data_directory,
                             'features_single_round_label_crf_manual_binary_features_verbal_data.xlsx'),
                os.path.join(model_directory, 'crf_model.pkl'),
                os.path.join(data_directory,
                             'test_data_1_10_single_round_label_crf_manual_binary_features_verbal_data.pkl'),
                ])
            crf = LinearChainCRF(squared_sigma=0.005)
            crf.train(args.traindatafile, args.modelfile, args.featuresfile, vector_rep_input=vector_rep_input)

            if test:
                crf.load(args.modelfile, args.featuresfile, vector_rep_input=vector_rep_input)
                crf.test(args.testdatafile, predict_only_last=True)

        else:
            args = parser.parse_args([
                os.path.join(data_directory, 'all_data_single_round_label_crf_manual_binary_features_verbal_data.pkl'),
                os.path.join(data_directory, 'features_single_round_label_crf_manual_binary_features_verbal_data.xlsx'),
                os.path.join(model_directory, 'crf_model.pkl'),
                os.path.join(data_directory, 'all_data_single_round_label_crf_manual_binary_features_verbal_data.pkl'),
            ])

            # split data to 5 folds
            num_folds = 5
            all_correct_count, all_total_count, all_total_seq_count = 0, 0, 0
            all_prediction_df = pd.DataFrame()
            if 'csv' in args.traindatafile:
                data_df = pd.read_csv(args.traindatafile)
            elif 'xlsx' in args.traindatafile:
                data_df = pd.read_excel(args.traindatafile)
            elif 'pkl' in args.traindatafile:
                data_df = joblib.load(args.traindatafile)
            else:
                print('Data format is not csv or csv or pkl')
                return
            folds = get_folds_per_participant(data=data_df, k_folds=num_folds, col_to_group='pair_id',
                                              col_to_group_in_df=True)
            for fold in range(num_folds):
                args = parser.parse_args([
                    os.path.join(data_directory,
                                 'all_data_single_round_label_crf_raisha_all_history_features_all_history_text_manual_'
                                 'binary_features_predict_first_round_verbal_data.pkl'),
                    os.path.join(data_directory,
                                 'features_single_round_label_crf_raisha_all_history_features_all_history_text_'
                                 'manual_binary_features_predict_first_round_verbal_data.xlsx'),
                    os.path.join(model_directory, f'crf_model_test_fold{fold}.pkl'),
                    os.path.join(data_directory,
                                 'all_data_single_round_label_crf_raisha_all_history_features_all_history_text_manual_'
                                 'binary_features_predict_first_round_verbal_data.pkl'),
                ])
                train_pair_ids = folds.loc[folds.fold_number != fold].pair_id.tolist()
                test_pair_ids = folds.loc[folds.fold_number == fold].pair_id.tolist()
                # For Debug with small train data
                # train_pair_ids = folds.loc[folds.fold_number == 4].pair_id.tolist()

                print(f'Data file name: {args.traindatafile}')
                logging.info(f'Data file name: {args.traindatafile}')
                crf = LinearChainCRF(squared_sigma=squared_sigma, predict_future=predict_future)
                crf.train(args.traindatafile, args.modelfile, args.featuresfile, vector_rep_input=vector_rep_input,
                          pair_ids=train_pair_ids, use_forward_backward_fix_history=use_forward_backward_fix_history)

                if test:
                    crf.load(args.modelfile, args.featuresfile, vector_rep_input=vector_rep_input)
                    correct_count, total_count, prediction_df, total_seq_count =\
                        crf.test(args.traindatafile, predict_only_last=predict_only_last, pair_ids=test_pair_ids,
                                 fold_num=fold, use_viterbi_fix_history=use_viterbi_fix_history)
                    all_correct_count += correct_count
                    all_total_count += total_count
                    all_total_seq_count += total_seq_count
                    all_prediction_df = all_prediction_df.append(prediction_df)

            mse = metrics.mean_squared_error(all_prediction_df.total_payoff_label,
                                             all_prediction_df.total_payoff_prediction)
            rmse = math.sqrt(mse)
            mae = metrics.mean_absolute_error(all_prediction_df.total_payoff_label,
                                              all_prediction_df.total_payoff_prediction)
            print('All folds accuracy')
            print('Correct: %d' % all_correct_count)
            print('Total: %d' % all_total_count)
            print('Performance: %f' % round((100 * all_correct_count / all_total_count), 2))
            print(f'Total sequences: {all_total_seq_count}')
            print(f'Total payoff MSE: {round(100 * mse, 2)}, RMSE: {round(100 * rmse, 2)}, MAE: {round(100 * mae, 2)}')
            logging.info('All folds accuracy')
            logging.info('Correct: %d' % all_correct_count)
            logging.info('Total: %d' % all_total_count)
            logging.info('Performance: %f' % round((100 * all_correct_count / all_total_count), 2))
            logging.info(f'Total sequences: {all_total_seq_count}')
            logging.info(f'Total payoff MSE: {round(100 * mse, 2)}, RMSE: {round(100 * rmse, 2)}, '
                         f'MAE: {round(100 * mae, 2)}')
            # print('potential score in vitervi fix history')
            # logging.info('potential score in vitervi fix history')

            all_prediction_df.to_excel(os.path.join(model_directory, 'crf_results.xlsx'))
            print(f'model directory is: {model_directory}')


if __name__ == '__main__':
    model_param = {
        'use_forward_backward_fix_history': False,
        'use_viterbi_fix_history': False,
        'squared_sigma': 0.008,
        'predict_only_last': False,
        'predict_future': False,
    }

    dir_name_component = [
        f'use_forward_backward_fix_history_' if model_param['use_forward_backward_fix_history'] else '',
        'use_viterbi_fix_history_' if model_param['use_viterbi_fix_history'] else '',
        f'squared_sigma_{model_param["squared_sigma"]}_',
        'predict_only_last_' if model_param['predict_only_last'] else '',
        'predict_future_' if model_param['predict_future'] else '',]
    base_file_name = ''.join(dir_name_component)

    model_directory = utils.set_folder(datetime.now().strftime(f'CRF_cv_{base_file_name}%d_%m_%Y_%H_%M'), 'logs')

    log_file_name = os.path.join(model_directory, datetime.now().strftime('LogFile.log'))
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )

    main(test=True, cv=True, test_chunking=False,
         use_forward_backward_fix_history=model_param['use_forward_backward_fix_history'],
         use_viterbi_fix_history=model_param['use_viterbi_fix_history'], squared_sigma=model_param['squared_sigma'],
         predict_only_last=model_param['predict_only_last'], predict_future=model_param['predict_future'])

    print(f'Model params:\n {model_param}')
    logging.info(f'Model params:\n {model_param}')
