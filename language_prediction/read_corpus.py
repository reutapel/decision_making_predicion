#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import joblib
import math
import logging


class FileFormatError(BaseException):
    pass


def read_conll_corpus(filename):
    """
    Read a corpus file with a format used in CoNLL.
    """
    data = list()
    data_string_list = list(open(filename))

    element_size = 0
    x = list()
    y = list()
    counter = 0
    for data_string in data_string_list:
        counter += 1
        words = data_string.strip().split()
        if len(words) is 0:
            data.append((x, y))
            x = list()
            y = list()
        else:
            if element_size is 0:
                element_size = len(words)
            elif element_size is not len(words):
                raise FileFormatError
            x.append(words[:-1])
            y.append(words[-1])
    if len(x) > 0:
        data.append((x, y))

    print(f'\n* Number of data points: {counter}')
    logging.info(f'\n* Number of data points: {counter}')

    return data


def read_verbal_exp_data(filename, pair_ids: list=None, predict_future: bool=False):
    """
    Read data of the verbal experiments
    :param filename: the data file name
    :param pair_ids: list of pair ids in the case of cross validation
    :param predict_future: if we want to get  all sequence data but to predict only the future
    :return:
    """
    data = list()
    if 'csv' in filename:
        data_df = pd.read_csv(filename)
    elif 'xlsx' in filename:
        data_df = pd.read_excel(filename)
    elif 'pkl' in filename:
        data_df = joblib.load(filename)
    else:
        print('Data format is not csv or csv or pkl')
        return

    if pair_ids is not None:
        data_df = data_df.loc[data_df.pair_id.isin(pair_ids)]
        data_df = data_df.drop(['pair_id'], axis=1)
    else:
        if 'pair_id' in data_df.columns:
            data_df = data_df.drop(['pair_id'], axis=1)

    # remove sequence of length 1:
    data_df = data_df.loc[data_df.labels.str.len() > 1]

    if predict_future:  # keep only sequence of length 10
        data_df = data_df.loc[data_df.labels.str.len() == 10]

    row_number = 0
    features_columns = data_df.columns.to_list()
    features_columns.remove('labels')
    for index, row in data_df.iterrows():
        # if row_number == 30:
        #     break
        x = list()
        y = row.labels
        for column in features_columns:
            if type(row[column]) == list:
                x.append(row[column])

        if predict_future:
            for raisha in range(1, 10):  # different size of raisha
                data.append((x, y, raisha))
                row_number += 1
        else:
            data.append((x, y))
            row_number += 1

    print(f'\n* Number of data points: {row_number+1}')
    logging.info(f'\n* Number of data points: {row_number+1}')

    return data


def read_feature_names(features_filename=None):
    """
    Read the names of the features to use
    :param features_filename:
    :return:
    """
    if features_filename is not None:
        features_names = pd.read_excel(features_filename)
        features_names = features_names[0].values.tolist()
    else:
        features_names = None

    return features_names

