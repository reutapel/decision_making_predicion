#!/usr/bin/env python

import argparse
from language_prediction.crf import LinearChainCRF
import os
import tempural_analysis.utils as utils
from datetime import datetime

base_directory = os.path.abspath(os.curdir)
model_name = 'chunking_small'
data_directory = os.path.join(base_directory, 'data', model_name)
model_directory = os.path.join(base_directory, 'logs', model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help="data file for training input")
    parser.add_argument('modelfile', help="the model file name. (output)")

    utils.set_folder(datetime.now().strftime(f'{model_name}_%d_%m_%Y_%H_%M'), 'logs')

    args = parser.parse_args([os.path.join(data_directory, 'small_train.data.txt'),
                              os.path.join(model_directory, 'crf_model.model')])

    crf = LinearChainCRF()
    crf.train(args.datafile, args.modelfile)
