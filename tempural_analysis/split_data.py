__all__ = ['get_folds', 'get_data_label_window', 'get_data_label_window_per_folds', 'get_folds_per_participant',
           'get_data_label_personal']
import copy
import pandas as pd
from random import shuffle
import math


def get_folds_per_participant(data: pd.DataFrame, k_folds: int=5, index_to_use: list=None,
                              col_to_group: str='participant_code') -> pd.DataFrame:
    """
    This function split the data to k folds, such that each participant will be in one fold only
    :param data: all the data to use
    :param k_folds: number of folds to split the data
    :param index_to_use: list of indexes to use --> after remove the first trials for window_size > 0
    :param col_to_group: the column to split the samples by (participant_code or pair)
    :return: data with a fold_number column
    """

    participants = pd.DataFrame(data[col_to_group].unique(), columns=[col_to_group])
    # shuffle participants
    participants = participants.sample(frac=1)
    participants = participants.assign(fold_number=0)
    participants_list = participants[col_to_group].unique()
    for k in range(k_folds):
        participants.loc[participants[col_to_group].isin(
            [x for i, x in enumerate(participants_list) if i % k_folds == k]), 'fold_number'] = k

    participants = data.merge(participants)
    participants.index = data.index
    participants = participants[['fold_number']]
    if index_to_use is not None:  # remove index that are not in the new x,y
        participants = participants.loc[[index for index in participants.index if index in index_to_use]]

    return participants


def get_folds(data: pd.DataFrame, k_folds: int=5, index_to_use: list=None, num_rounds_fold: int=None,
              col_to_group: str = 'participant_code') -> pd.DataFrame:
    """
    This function split the data to k folds, such that each participant will be in all folds -->
    split the participant such that each 10 Subsequent rounds will be in the same fold
    :param data: all the data to use
    :param k_folds: number of folds to split the data
    :param index_to_use: list of indexes to use --> after remove the first trials for window_size > 0
    :param num_rounds_fold: the number of rounds to put in each fold, if None- put total_num_rounds/k_folds
    :param col_to_group: the column to split the samples by (participant_code or pair)
    :return: data with a fold_number column
    """

    participants_list = data[col_to_group].unique()
    folds = data[[col_to_group, 'subsession_round_number']]
    first_round = int(folds.subsession_round_number.min())
    last_round = int(folds.subsession_round_number.max())
    if num_rounds_fold is None:
        num_rounds_fold = math.floor(last_round / k_folds)
    folds = folds.assign(fold_number=0)

    # define the list of rounds --> each list in rounds_list is rounds to be in the same fold
    rounds_list = list()
    for k in range(k_folds-1):
        rounds_list.append(list(range(first_round + (num_rounds_fold * k), first_round + (num_rounds_fold*(k+1)))))
    # insert the last rounds
    rounds_list.append(list(range(first_round + (num_rounds_fold*(k_folds-1)), last_round + 1)))

    for participant in participants_list:
        for k in range(1, k_folds+1):
            folds.loc[(folds[col_to_group] == participant) & (folds.subsession_round_number.isin(rounds_list[k-1])),
                      'fold_number'] = k
        shuffle(rounds_list)

    folds = folds[['fold_number']]
    if index_to_use is not None:  # remove index that are not in the new x,y
        folds = folds.loc[[index for index in folds.index if index in index_to_use]]

    return folds


def get_data_label_personal(data: pd.DataFrame, label: str, features: list, personal_features: list) ->\
        (pd.DataFrame, pd.Series, pd.DataFrame):
    """
    This function create the data and label if window size == 0 (non temporary model, predict singel round)
    :param data: all the data to use
    :param label: the label we want to predict
    :param features: the features we use in this run
    :param personal_features: list of personal features such as age and gender --> use them only in t=0
    :return:
    """

    x = copy.deepcopy(data[[feature for feature in features if feature not in personal_features]])
    y = copy.deepcopy(data[label])
    personal_features_data = copy.deepcopy(data[personal_features])

    return x, y, personal_features_data


def get_data_label_window(x: pd.DataFrame, y: pd.Series, label: str, window_size: int, appendix: str,
                          personal_features_data: pd.DataFrame, first_round_features: pd.DataFrame=None,
                          use_first_round: bool=False, candidate=None, subsession_round_number_removed=None) ->\
        (pd.DataFrame, pd.DataFrame):
    """
    This function create the data and label for window size > 0 (temporary model)
    :param x: the features
    :param y: the labels
    :param label: the label we want to predict
    :param window_size: the size of the window to use as features
    :param appendix: the appendix of the columns (group/player)
    :param personal_features_data: personal features such as age and gender --> use them only in t=0
    :param use_first_round: do we want to use the first round in all samples
    :param first_round_features: features of the first round for each participant. index is the participant_code
    :param candidate: the candidate to remove if backward elimination
    :param subsession_round_number_removed: if ew removed subsession_round_number from the features
    :return:
    """

    # TODO: check if I need to drop participant_code
    if window_size == 0:  # return the data as is
        return x, y

    # x = x.sort_values(by=['participant_code', 'subsession_round_number'])
    # y = y.reindex(x.index)

    rows_to_keep = copy.deepcopy(x[['subsession_round_number']])
    rows_to_keep['round-shift_5'] = rows_to_keep.subsession_round_number -\
                                    rows_to_keep.subsession_round_number.shift(window_size)
    rows_to_keep = rows_to_keep.loc[rows_to_keep['round-shift_5'] == window_size]

    # if the round_number is the candidate to remove, or we already removed it - remove it here
    if 'subsession_round_number' == candidate or subsession_round_number_removed:
        x = x.drop('subsession_round_number', axis=1)

    to_shift = copy.deepcopy(x)
    for i in range(1, window_size+1):  # go from 1 to the window_size
        to_shift = pd.concat([x.shift(i), to_shift], axis=1)

    # change columns names for previous rounds
    to_shift.columns = [name+'_t_' + str(i) for i in range(window_size, -1, -1) for name in x.columns]

    # add the first round for all samples:
    if use_first_round:
        # merge to_shift with the first_round_features on the participant_code for round 0
        to_shift = to_shift.merge(first_round_features, left_on='participant_code_t_0', right_index=True)

    # TODO: think if I want to deal with that differently
    to_shift = to_shift.dropna()  # remove rows with NA values
    # remove rounds 1-window_size --> the history of them is from the previous participant --> not really a window
    # to_shift = to_shift.loc[~to_shift['subsession_round_number_t_0'].isin(range(1, window_size+1))]
    to_shift = to_shift.loc[[index for index in to_shift.index if index in rows_to_keep.index]]
    # drop the participant code for all rounds
    for t in range(window_size, -1, -1):
        to_shift = to_shift.drop('participant_code_t_' + str(t), axis=1)
    # remove features from t_0 if it's the label or post treatment features
    for column in [label + '_t_0', appendix + '_lottery_result_t_0', appendix + '_receiver_timeout_t_0',
                   appendix + '_sender_timeout_t_0', 'expert_wrong_0.5_t_0', 'expert_wrong_0.8_t_0', 'payoff_t_0',
                   'best_response_t_0']:
        if column in to_shift.columns:
            to_shift = to_shift.drop(labels=column, axis=1)  # remove from time t

    # add the personal features
    to_shift = to_shift.merge(personal_features_data, right_index=True, left_index=True)

    y.name = 'label'
    y = y.loc[[index for index in y.index if index in to_shift.index]]
    return to_shift, y


def get_data_label_window_per_folds(x: pd.DataFrame, y: pd.Series, folds: pd.DataFrame, label: str, appendix: str,
                                    personal_features_data: pd.DataFrame, window_size: int, k: int,
                                    use_first_round: bool=False, candidate=None,
                                    subsession_round_number_removed=None,
                                    col_to_group: str='participant_code') -> (pd.DataFrame, pd.DataFrame):
    """
    This function create the data and label for window size > 0 (temporary model)
    :param x: the features
    :param y: the labels
    :param folds: the fold number for each sample
    :param personal_features_data: data frame with the personal features of the participants
    :param window_size: the size of the window to use as features
    :param appendix: the appendix of the columns (group/player)
    :param label: the label we want to predict
    :param k: the fold to be test
    :param use_first_round: do we want to use the first round in all samples
    :param candidate: the candidate to remove if backward elimination
    :param subsession_round_number_removed: if we removed subsession_round_number from features
    :param col_to_group: the column to split the samples by (participant_code or pair)
    :return:
    """
    train_x = x.loc[folds.fold_number != k]
    train_y = y[folds.fold_number != k]
    test_x = x.loc[folds.fold_number == k]
    test_y = y[folds.fold_number == k]
    test_folds = folds[folds.fold_number == k]

    # add the first round for all samples:
    if use_first_round:
        first_round_features = copy.deepcopy(x.loc[x.subsession_round_number == 1])
        first_round_features.index = first_round_features[col_to_group]
        first_round_features = first_round_features.drop(col_to_group, axis=1)
        # if we need to remove subsession_round_number from the features
        if 'subsession_round_number' == candidate or subsession_round_number_removed:
            first_round_features = first_round_features.drop('subsession_round_number', axis=1)
        # rename the columns of the first round
        first_round_features.columns = [name + '_first_round' for name in first_round_features.columns]
    else:
        first_round_features = None

    train_x, train_y = get_data_label_window(train_x, train_y, label, window_size,  appendix, personal_features_data,
                                             first_round_features, use_first_round, candidate,
                                             subsession_round_number_removed)
    test_x, test_y = get_data_label_window(test_x, test_y, label, window_size,  appendix, personal_features_data,
                                           first_round_features, use_first_round, candidate,
                                           subsession_round_number_removed)
    test_folds = test_folds.loc[[index for index in test_folds.index if index in test_y.index]]

    return train_x, train_y, test_x, test_y, test_folds
