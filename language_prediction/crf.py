#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Laon-CRF
    : Easy-to-use Linear Chain Conditional Random Fields
Author: Seong-Jin Kim
License: MIT License
Version: 0.0
Email: lancifollia@gmail.com
Created: May 13, 2015
Copyright (c) 2015 Seong-Jin Kim
"""


from language_prediction.read_corpus import read_conll_corpus, read_verbal_exp_data, read_feature_names
from language_prediction.feature import FeatureSet, STARTING_LABEL_INDEX

from math import exp, log
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import time
import json
import datetime
import joblib
import os
import logging

from .feature import VectorRepresentationInput

from collections import Counter

SCALING_THRESHOLD = 1e250

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
GRADIENT = None


def _callback(params):
    global ITERATION_NUM
    global SUB_ITERATION_NUM
    global TOTAL_SUB_ITERATIONS
    ITERATION_NUM += 1
    TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    SUB_ITERATION_NUM = 0


def _generate_potential_table(params, num_labels, feature_set, x, inference=True, y_list: list=None,
                              label_dic: dict=None, raisha: int=None):
    """
    Generates a potential table using given observations.
    * potential_table[t][prev_y, y]
        := exp(inner_product(params, feature_vector(prev_y, y, X, t)))
        (where 0 <= t < len(X))
    """
    tables = list()
    for t in range(len(x)):
        table = np.zeros((num_labels, num_labels))
        if inference:
            for (prev_y, y), score in feature_set.calc_inner_products(params, x, t):
                if prev_y == -1:
                    table[:, y] += score
                else:
                    table[prev_y, y] += score
        else:
            for (prev_y, y), feature_ids in x[t]:
                if t < raisha:  # for the steps in the raisha - use the correct label
                    if prev_y == -1:  # for unigram features
                        if y == label_dic[y_list[t]]:
                            score = sum(params[fid] for fid in feature_ids)
                        else:
                            continue
                    elif t == 0:  # for bigram in t==0, the label should be STARTING_LABEL_INDEX
                        if prev_y is not STARTING_LABEL_INDEX:
                            continue
                        else:
                            if y == label_dic[y_list[t]]:  # t=0, bigram with prev_y = STARTING_LABEL_INDEX
                                score = sum(params[fid] for fid in feature_ids)
                            else:
                                continue
                    else:
                        # for t>0, bigram, the prev_y could not be STARTING_LABEL_INDEX
                        if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                            continue
                        else:
                            if y == label_dic[y_list[t]] and prev_y == label_dic[y_list[t-1]]:
                                score = sum(params[fid] for fid in feature_ids)
                            else:
                                continue
                else:
                    score = sum(params[fid] for fid in feature_ids)
                # logging.info(f'score: {score}')
                if prev_y == -1:
                    table[:, y] += score
                else:
                    table[prev_y, y] += score
        table = np.exp(table)
        if t == 0:
            table[STARTING_LABEL_INDEX+1:] = 0
        else:
            table[:, STARTING_LABEL_INDEX] = 0
            table[STARTING_LABEL_INDEX, :] = 0
        tables.append(table)

    return tables


def _forward_backward(num_labels, time_length, potential_table):
    """
    Calculates alpha(forward terms), beta(backward terms), and Z(instance-specific normalization factor)
        with a scaling method(suggested by Rabiner, 1989).
    * Reference:
        - 1989, Lawrence R. Rabiner, A Tutorial on Hidden Markov Models and Selected Applications
        in Speech Recognition
    """
    alpha = np.zeros((time_length, num_labels))
    scaling_dic = dict()
    t = 0
    for label_id in range(num_labels):
        alpha[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
    # alpha[0, :] = potential_table[0][STARTING_LABEL_INDEX, :]  # slow
    t = 1
    while t < time_length:
        scaling_time = None
        scaling_coefficient = None
        overflow_occured = False
        label_id = 1
        while label_id < num_labels:
            alpha[t, label_id] = np.dot(alpha[t-1, :], potential_table[t][:, label_id])
            if alpha[t, label_id] > SCALING_THRESHOLD:
                if overflow_occured:
                    print('******** Consecutive overflow ********')
                    raise BaseException()
                overflow_occured = True
                scaling_time = t - 1
                scaling_coefficient = SCALING_THRESHOLD
                scaling_dic[scaling_time] = scaling_coefficient
                break
            else:
                label_id += 1
        if overflow_occured:
            alpha[t-1] /= scaling_coefficient
            alpha[t] = 0
        else:
            t += 1

    beta = np.zeros((time_length, num_labels))
    t = time_length - 1
    for label_id in range(num_labels):
        beta[t, label_id] = 1.0
    # beta[time_length - 1, :] = 1.0     # slow
    for t in range(time_length-2, -1, -1):
        for label_id in range(1, num_labels):
            beta[t, label_id] = np.dot(beta[t+1, :], potential_table[t+1][label_id, :])
        if t in scaling_dic.keys():
            beta[t] /= scaling_dic[t]

    z = sum(alpha[time_length-1])
    if z == 0.0:
        reut = 1
    # logging.info(f'alpha: {alpha}, beta: {beta}')

    return alpha, beta, z, scaling_dic


def _forward_backward_fix_history(num_labels, time_length, potential_table, y: list, label_dic: dict, raisha: int):
    """
    Calculates alpha(forward terms), beta(backward terms), and Z(instance-specific normalization factor)
        with a scaling method(suggested by Rabiner, 1989).
    * Reference:
        - 1989, Lawrence R. Rabiner, A Tutorial on Hidden Markov Models and Selected Applications
        in Speech Recognition
    """

    alpha = np.zeros((time_length, num_labels))
    scaling_dic = dict()

    t = 0
    alpha[t, label_dic[y[t]]] = potential_table[t][STARTING_LABEL_INDEX, label_dic[y[t]]]
    for t in range(1, raisha):  # the history --> the correct label
        alpha[t, label_dic[y[t]]] = np.dot(alpha[t-1, :], potential_table[t][:, label_dic[y[t]]])

    t = raisha
    while t < time_length:
        scaling_coefficient = None
        overflow_occured = False
        label_id = 1
        while label_id < num_labels:
            alpha[t, label_id] = np.dot(alpha[t-1, :], potential_table[t][:, label_id])
            if alpha[t, label_id] > SCALING_THRESHOLD:
                if overflow_occured:
                    print('******** Consecutive overflow ********')
                    raise BaseException()
                overflow_occured = True
                scaling_time = t - 1
                scaling_coefficient = SCALING_THRESHOLD
                scaling_dic[scaling_time] = scaling_coefficient
                break
            else:
                label_id += 1
        if overflow_occured:
            alpha[t-1] /= scaling_coefficient
            alpha[t] = 0
        else:
            t += 1

    beta = np.zeros((time_length, num_labels))
    for label_id in range(num_labels):
        beta[time_length - 1, label_id] = 1.0
    t = raisha - 1
    beta[t, label_dic[y[t]]] = 1.0
    for t in range(raisha-2, -1, -1):  # the history --> the correct label
        beta[t, label_dic[y[t]]] = np.dot(beta[t+1, :], potential_table[t+1][label_dic[y[t]], :])
    # beta[time_length - 1, :] = 1.0     # slow
    for t in range(time_length-2, raisha-1, -1):
        for label_id in range(1, num_labels):
            beta[t, label_id] = np.dot(beta[t+1, :], potential_table[t+1][label_id, :])
        if t in scaling_dic.keys():
            beta[t] /= scaling_dic[t]

    z = sum(alpha[time_length-1])
    # logging.info(f'alpha: {alpha}, beta: {beta}')

    return alpha, beta, z, scaling_dic


def _calc_path_score(potential_table, scaling_dic, Y, label_dic):
    score = 1.0
    prev_y = STARTING_LABEL_INDEX
    for t in range(len(Y)):
        y = label_dic[Y[t]]
        score *= potential_table[prev_y, y, t]
        if t in scaling_dic.keys():
            score = score / scaling_dic[t]
        prev_y = y
    return score


def _log_likelihood(params, *args):
    """
    Calculate likelihood and gradient
    """

    training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma,\
        use_forward_backward_fix_history = args
    expected_counts = np.zeros(len(feature_set))

    total_logZ = 0
    for i, X_features in enumerate(training_feature_data):
        potential_table = _generate_potential_table(params, len(label_dic), feature_set, X_features, inference=False,
                                                    y_list=training_data[i][1], label_dic=label_dic,
                                                    raisha=training_data[i][2])
        if use_forward_backward_fix_history:
            alpha, beta, z, scaling_dic = _forward_backward_fix_history(len(label_dic), len(X_features),
                                                                        potential_table, y=training_data[i][1],
                                                                        label_dic=label_dic, raisha=training_data[i][2])
        else:
            alpha, beta, z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
        total_logZ += log(z) + \
                      sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
        for t in range(len(X_features)):
            potential = potential_table[t]
            for (prev_y, y), feature_ids in X_features[t]:
                # Adds p(prev_y, y | X, t)
                if prev_y == -1:
                    if t in scaling_dic.keys():
                        prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/z
                    else:
                        prob = (alpha[t, y] * beta[t, y])/z
                elif t == 0:
                    if prev_y is not STARTING_LABEL_INDEX:
                        continue
                    else:
                        prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/z
                else:
                    if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                        continue
                    else:
                        prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / z
                for fid in feature_ids:
                    if prob == np.inf:
                        print(f'prob is inf')
                    expected_counts[fid] += prob

    likelihood = np.dot(empirical_counts, params) - total_logZ - \
                 np.sum(np.dot(params, params))/(squared_sigma*2)  # log conditional likelihood - regularization(L2)

    gradients = empirical_counts - expected_counts - params/squared_sigma  # gradient of the log likelihood
    global GRADIENT
    GRADIENT = gradients

    global SUB_ITERATION_NUM
    sub_iteration_str = '    '
    if SUB_ITERATION_NUM > 0:
        sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION_NUM) + ')'

    if sub_iteration_str == '(19)':
        reut = 1
    print('  ', '{0:03d}'.format(ITERATION_NUM), sub_iteration_str, ':', likelihood * -1)
    logging.info(f'{ITERATION_NUM}  {sub_iteration_str} : {likelihood * -1}')

    SUB_ITERATION_NUM += 1

    return likelihood * -1


def _gradient(params, *args):
    return GRADIENT * -1


def read_corpus(filename, pair_ids: list=None, predict_future: bool=False):
    """
    Read the data
    :param filename: the data file name
    :param pair_ids: list of pair ids in the case of cross validation
    :param predict_future: if we want to get  all sequence data but to predict only the future
    :return:
    """
    if 'txt' in filename:
        return read_conll_corpus(filename)
    else:
        return read_verbal_exp_data(filename, pair_ids, predict_future)


class LinearChainCRF():
    """
    Linear-chain Conditional Random Field
    """

    training_data = None
    feature_set = None

    label_dic = None
    label_array = None
    num_labels = None

    params = None

    def __init__(self, squared_sigma: float=0.0005, predict_future: bool=False):
        """
        :param squared_sigma:
        :param predict_future: if we want to get  all sequence data but to predict only the future
        """
        self.squared_sigma = squared_sigma  # For L-BFGS
        self.predict_future = predict_future
        pass

    def _get_training_feature_data(self):
        # x=data_list[0]
        return [[self.feature_set.get_feature_list(data_list[0], t) for t in range(len(data_list[0]))]
                for data_list in self.training_data]

    def _estimate_parameters(self, use_forward_backward_fix_history):
        """
        Estimates parameters using L-BFGS.
        * References:
            - R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization,
            (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
            - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large
            scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4,
            pp. 550 - 560.
            - J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for
            large scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.
        """
        training_feature_data = self._get_training_feature_data()
        print('* Squared sigma:', self.squared_sigma)
        print('* Start L-BGFS')
        print('   ========================')
        print('   iter(sit): likelihood')
        print('   ------------------------')
        logging.info(f'* Squared sigma: {self.squared_sigma}')
        logging.info('* Start L-BGFS')
        logging.info('   ========================')
        logging.info('   iter(sit): likelihood')
        logging.info('   ------------------------')
        self.params, log_likelihood, information = \
                fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
                              x0=np.zeros(len(self.feature_set)),
                              args=(self.training_data, self.feature_set, training_feature_data,
                                    self.feature_set.get_empirical_counts(),
                                    self.label_dic, self.squared_sigma, use_forward_backward_fix_history),
                              callback=_callback)
        print('   ========================')
        print('   (iter: iteration, sit: sub iteration)')
        print('* Training has been finished with %d iterations' % information['nit'])
        logging.info('   ========================')
        logging.info('   (iter: iteration, sit: sub iteration)')
        logging.info('* Training has been finished with %d iterations' % information['nit'])

        if information['warnflag'] != 0:
            print('* Warning (code: %d)' % information['warnflag'])
            if 'task' in information.keys():
                print('* Reason: %s' % (information['task']))
        print('* Likelihood: %s' % str(log_likelihood))
        logging.info('* Likelihood: %s' % str(log_likelihood))

    def train(self, corpus_filename, model_filename, features_filename=None, vector_rep_input: bool=True,
              pair_ids: list=None, use_forward_backward_fix_history: bool=False):
        """
        Estimates parameters using conjugate gradient methods.(L-BFGS-B used)
        :param corpus_filename: the data file name
        :param features_filename: the feature names file name
        :param model_filename: the model file name
        :param vector_rep_input: if we want to use the vector representation input to teh features function
        :param pair_ids: list of pair ids in the case of cross validation
        :param use_forward_backward_fix_history: if we want to use a fix history and predict only the future
        :return:
        """

        start_time = time.time()
        print('[%s] Start training' % datetime.datetime.now())
        logging.info('[%s] Start training' % datetime.datetime.now())

        if use_forward_backward_fix_history:
            print('Use forward_backward_fix_history')
            logging.info('Use forward_backward_fix_history')
        else:
            print('Use forward_backward')
            logging.info('Use forward_backward')

        # Read the training corpus
        print("* Reading training data ... ", end="")
        logging.info("* Reading training data ... ")
        self.training_data = read_corpus(corpus_filename, pair_ids, self.predict_future)
        print("Done")
        logging.info("Done")

        # Generate feature set from the corpus
        if vector_rep_input and features_filename is not None:
            feature_names = read_feature_names(features_filename)
            self.feature_set = FeatureSet(VectorRepresentationInput, feature_names)
        else:
            self.feature_set = FeatureSet()
        self.feature_set.scan(self.training_data)
        self.label_dic, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        print("* Number of labels: %d" % (self.num_labels-1))
        print("* Number of features: %d" % len(self.feature_set))
        logging.info("* Number of labels: %d" % (self.num_labels-1))
        logging.info("* Number of features: %d" % len(self.feature_set))

        # Estimates parameters to maximize log-likelihood of the corpus.
        self._estimate_parameters(use_forward_backward_fix_history)

        self.save_model(model_filename)

        elapsed_time = time.time() - start_time
        print('* Elapsed time: %f' % elapsed_time)
        print('* [%s] Training done' % datetime.datetime.now())
        logging.info('* Elapsed time: %f' % elapsed_time)
        logging.info('* [%s] Training done' % datetime.datetime.now())

    def test(self, test_corpus_filename, predict_only_last=False, pair_ids: list=None, fold_num: int=None,
             use_viterbi_fix_history: bool=None):
        """
        Test the model
        :param test_corpus_filename: the test data filename
        :param predict_only_last: if we want to predict only the last round in the seq
        :param pair_ids: list of pair ids in the case of cross validation
        :param fold_num: if running cross validation this is the fold for test
        :param use_viterbi_fix_history: if we want ot use the viterbi_fix_history function
        :return:
        """
        if self.params is None:
            raise BaseException("You should load a model first!")

        if fold_num is not None:
            logging.info(f'Start testing on fold {fold_num}')
            logging.info(f'Start testing on fold {fold_num}')
        test_data = read_corpus(test_corpus_filename, pair_ids, self.predict_future)

        if use_viterbi_fix_history is None:
            use_viterbi_fix_history = predict_only_last

        if use_viterbi_fix_history:
            print('Use viterbi_fix_history')
            logging.info('Use viterbi_fix_history')
        else:
            print('Use viterbi')
            logging.info('Use viterbi')

        total_count = 0
        correct_count = 0
        for data_list in test_data:  # x=data_list[0], y=data_list[1], raisha=data_list[2]
            x = data_list[0]
            y = data_list[1]
            raisha = data_list[2] if len(data_list) == 3 else None
            y_prime = self.inference(x=x, y=y, raisha=raisha, use_viterbi_fix_history=use_viterbi_fix_history)
            if predict_only_last:
                for t in range(raisha, len(y)):
                    total_count += 1
                    if y[t] == y_prime[t]:
                        correct_count += 1
            else:
                for t in range(len(y)):
                    total_count += 1
                    if y[t] == y_prime[t]:
                        correct_count += 1

        if fold_num is not None:
            print(f'Model Performance for fold: {fold_num}')
            logging.info(f'Model Performance for fold: {fold_num}')
        print('Correct: %d' % correct_count)
        print('Total: %d' % total_count)
        print('Performance: %f' % (correct_count/total_count))
        logging.info('Correct: %d' % correct_count)
        logging.info('Total: %d' % total_count)
        logging.info('Performance: %f' % (correct_count/total_count))

        return correct_count, total_count

    def print_test_result(self, test_corpus_filename, pair_ids: list=None):
        """
        Print the test results of the model
        :param test_corpus_filename: the test data filename
        :param pair_ids: list of pair ids in the case of cross validation
        :return:
        """
        test_data = read_corpus(test_corpus_filename, pair_ids, self.predict_future)

        for x, y in test_data:
            y_prime = self.inference(x)
            for t in range(len(x)):
                print('%s\t%s\t%s' % ('\t'.join(x[t]), y[t], y_prime[t]))
            print()

    def inference(self, x, y=None, raisha=None, use_viterbi_fix_history=None):
        """
        Finds the best label sequence.
        """
        potential_table = _generate_potential_table(self.params, self.num_labels,
                                                    self.feature_set, x, inference=True)
        if use_viterbi_fix_history:
            return self.viterbi_fix_history(x, potential_table, y, raisha)
        else:
            return self.viterbi(x, potential_table)

    def viterbi(self, x, potential_table):
        """
        The Viterbi algorithm with backpointers
        """

        time_length = len(x)
        max_table = np.zeros((time_length, self.num_labels))
        argmax_table = np.zeros((time_length, self.num_labels), dtype='int64')

        t = 0
        for label_id in range(self.num_labels):
            max_table[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
        for t in range(1, time_length):
            for label_id in range(1, self.num_labels):
                max_value = -float('inf')
                max_label_id = None
                for prev_label_id in range(1, self.num_labels):
                    value = max_table[t-1, prev_label_id] * potential_table[t][prev_label_id, label_id]
                    if value > max_value:
                        max_value = value
                        max_label_id = prev_label_id
                max_table[t, label_id] = max_value
                argmax_table[t, label_id] = max_label_id

        sequence = list()
        next_label = max_table[time_length-1].argmax()
        sequence.append(next_label)
        for t in range(time_length-1, -1, -1):
            next_label = argmax_table[t, next_label]
            sequence.append(next_label)
        return [self.label_dic[label_id] for label_id in sequence[::-1][1:]]

    def viterbi_fix_history(self, x, potential_table, y, raisha):
        """
        The Viterbi algorithm with backpointers
        """

        time_length = len(x)
        max_table = np.zeros((time_length, self.num_labels))
        argmax_table = np.zeros((time_length, self.num_labels), dtype='int64')
        reverse_label_dict = {value: key for key, value in self.label_dic.items()}

        max_table[0, reverse_label_dict[y[0]]] = 1.0
        for t in range(1, raisha):  # the history --> the correct label
            max_table[t, reverse_label_dict[y[t]]] = 1.0
                # potential_table[t][reverse_label_dict[y[t-1]], reverse_label_dict[y[t]]]
            argmax_table[t, reverse_label_dict[y[t]]] = reverse_label_dict[y[t-1]]

        for t in range(raisha, time_length):  # the future --> calculate the label
            for label_id in range(1, self.num_labels):
                max_value = -float('inf')
                max_label_id = None
                for prev_label_id in range(1, self.num_labels):
                    value = max_table[t - 1, prev_label_id] * potential_table[t][prev_label_id, label_id]
                    if value > max_value:
                        max_value = value
                        max_label_id = prev_label_id
                max_table[t, label_id] = max_value
                argmax_table[t, label_id] = max_label_id

        sequence = list()
        next_label = max_table[time_length-1].argmax()
        sequence.append(next_label)
        for t in range(time_length-1, -1, -1):
            next_label = argmax_table[t, next_label]
            sequence.append(next_label)
        return [self.label_dic[label_id] for label_id in sequence[::-1][1:]]

    def save_model(self, model_filename):
        model = {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.label_array,
                 "params": list(self.params),
                 'empirical_counts': self.feature_set.empirical_counts,
                 'observation_set': self.feature_set.observation_set}
        joblib.dump(model, model_filename)
        # f = open(model_filename, 'w')
        # json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        # f.close()
        print('* Trained CRF Model has been saved at "%s/%s"' % (os.getcwd(), model_filename))
        logging.info('* Trained CRF Model has been saved at "%s/%s"' % (os.getcwd(), model_filename))

    def load(self, model_filename, features_filename=None, vector_rep_input: bool = True):

        """Load model and create FeatureSet
        :param features_filename: the feature names file name
        :param model_filename: the model file name
        :param vector_rep_input: if we want to use the vector representation input to teh features function
        :return:
        """

        # f = open(model_filename)
        # model = json.load(f)
        # f.close()

        model = joblib.load(model_filename)
        if vector_rep_input and features_filename is not None:
            feature_names = read_feature_names(features_filename)
            self.feature_set = FeatureSet(VectorRepresentationInput, feature_names)
        else:
            self.feature_set = FeatureSet()
        self.feature_set.load(model['feature_dic'], model['num_features'], model['labels'], model['empirical_counts'],
                              model['observation_set'])
        self.label_dic, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        self.params = np.asarray(model['params'])

        print('CRF model loaded')
        logging.info('CRF model loaded')
