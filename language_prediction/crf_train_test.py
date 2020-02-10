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

base_directory = os.path.abspath(os.curdir)
model_name = 'verbal'
data_directory = os.path.join(base_directory, 'data', model_name)

random.seed(1)


def main(test=False, cv=False, test_chunking=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('traindatafile', help="data file for training input")
    parser.add_argument('featuresfile', help="the features file name input.")
    parser.add_argument('modelfile', help="the model file name. (output)")
    parser.add_argument('testdatafile', help="data file for testing input")

    model_directory = utils.set_folder(datetime.now().strftime(f'CRF_cv_%d_%m_%Y_%H_%M'), 'logs')
    if test_chunking:
        vector_rep_input = False
        args = parser.parse_args([
            os.path.join(data_directory, 'train.txt'),
            os.path.join(data_directory, 'features_single_round_label_crf_manual_binary_features_verbal_data.xlsx'),
            os.path.join(model_directory, 'crf_model.pkl'),
            os.path.join(data_directory, 'test.txt'),
        ])

        crf = LinearChainCRF()
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
            crf = LinearChainCRF()
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
            all_correct_count, all_total_count = 0, 0
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
                                 'all_data_single_round_label_crf_manual_binary_features_verbal_data.pkl'),
                    os.path.join(data_directory,
                                 'features_single_round_label_crf_manual_binary_features_verbal_data.xlsx'),
                    os.path.join(model_directory, f'crf_model_test_fold{fold}.pkl'),
                    os.path.join(data_directory,
                                 'all_data_single_round_label_crf_manual_binary_features_verbal_data.pkl'),
                ])
                train_pair_ids = folds.loc[folds.fold_number != fold].pair_id.tolist()
                test_pair_ids = folds.loc[folds.fold_number == fold].pair_id.tolist()

                crf = LinearChainCRF()
                crf.train(args.traindatafile, args.modelfile, args.featuresfile, vector_rep_input=vector_rep_input,
                          pair_ids=train_pair_ids)

                if test:
                    crf.load(args.modelfile, args.featuresfile, vector_rep_input=vector_rep_input)
                    correct_count, total_count =\
                        crf.test(args.traindatafile, predict_only_last=True, pair_ids=test_pair_ids, fold_num=fold,
                                 use_viterbi_fix_history=False)
                    all_correct_count += correct_count
                    all_total_count += total_count

            print('All folds accuracy')
            print('Correct: %d' % all_correct_count)
            print('Total: %d' % all_total_count)
            print('Performance: %f' % (all_correct_count / all_total_count))


if __name__ == '__main__':
    main(test=True, cv=True)
