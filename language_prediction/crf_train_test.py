#!/usr/bin/env python

import argparse
from language_prediction.crf import LinearChainCRF
import os
import tempural_analysis.utils as utils
from datetime import datetime

base_directory = os.path.abspath(os.curdir)
model_name = 'chunking_small'
data_directory = os.path.join(base_directory, 'data', model_name)


def main(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('traindatafile', help="data file for training input")
    parser.add_argument('featuresfile', help="the features file name input.")
    parser.add_argument('modelfile', help="the model file name. (output)")
    parser.add_argument('testdatafile', help="data file for testing input")

    model_directory = utils.set_folder(datetime.now().strftime(f'CRF_{model_name}_%d_%m_%Y_%H_%M'), 'logs')

    args = parser.parse_args([
        os.path.join(data_directory, 'train_data_1_10_single_round_label_crf_manual_binary_features_verbal_data.pkl'),
        os.path.join(data_directory, 'features_single_round_label_crf_manual_binary_features_verbal_data.xlsx'),
        os.path.join(model_directory, 'crf_model.pkl'),
        os.path.join(data_directory, 'test_data_1_10_single_round_label_crf_manual_binary_features_verbal_data.pkl')])

    crf = LinearChainCRF()
    crf.train(args.traindatafile, args.featuresfile, args.modelfile)

    if test:
        crf.load(args.modelfile)
        crf.test(args.datafile)


if __name__ == '__main__':
    main(test=True)
