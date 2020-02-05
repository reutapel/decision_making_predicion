#!/usr/bin/env python

import argparse
from language_prediction.crf import LinearChainCRF
import os


base_directory = os.path.abspath(os.curdir)
model_name = 'chunking_small'
data_directory = os.path.join(base_directory, 'data', model_name)
model_directory = os.path.join(base_directory, 'data', model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help="data file for testing input")
    parser.add_argument('modelfile', help="the model file name. (output)")

    args = parser.parse_args([
        os.path.join(data_directory, 'test_data_1_10_single_round_label_crf_manual_binary_features_verbal_data.pkl'),
        os.path.join(model_directory, 'crf_model.pkl')])

    crf = LinearChainCRF()
    crf.load(args.modelfile)
    crf.test(args.datafile)
