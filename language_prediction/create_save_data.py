import pandas as pd
import os
from datetime import datetime
import logging
import numpy as np
import random
import scipy.sparse as sparse
import joblib
import copy
import time
from language_prediction.train_test_models import *
import itertools
from collections import defaultdict


base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
cv_framework = 'cv_framework'
data_directory = os.path.join(base_directory, 'data', condition)
save_data_directory = os.path.join(base_directory, 'data', condition, cv_framework)
logs_directory = os.path.join(base_directory, 'logs')

log_file_name = os.path.join(logs_directory, datetime.now().strftime('LogFile_create_save_data_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=log_file_name,
                    level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    )
random.seed(1)

# define the alpha for the weighted average of the history features - global and text features
# if alpha == 0: use average
alpha_text = 0.5
alpha_global = 0.5

# define global raisha and saifa names to prevent typos
global_raisha = 'raisha'
global_saifa = 'saifa'
crf_label_col_name = 'labels'


def rename_review_features_column(review_data: pd.DataFrame, prefix_column_name: str):
    columns_to_rename = review_data.columns.tolist()
    if 'review_id' in columns_to_rename:
        columns_to_rename.remove('review_id')
    if review_data.columns[0] == 'review_id':  # review_id first
        review_data.columns = ['review_id'] + [f'{prefix_column_name}_{i}' for i in columns_to_rename]
    elif review_data.columns[review_data.shape[1]-1] == 'review_id':  # review_if last
        review_data.columns = [f'{prefix_column_name}_{i}' for i in columns_to_rename] + ['review_id']
    else:
        logging.exception(f'in rename_review_features_column with prefix {prefix_column_name}: '
                          f'review_id is not the first or the last column')
        raise Exception(f'in rename_review_features_column with prefix {prefix_column_name}: '
                        f'review_id is not the first or the last column')

    return review_data


def create_average_history_text(rounds: list, temp_reviews: pd.DataFrame, prefix_history_col_name: str='history',
                                prefix_future_col_name: str = 'future'):
    """
    This function get the temp reviews with the review_id, round_number and the features for the reviews as column for
    each feature
    :param rounds: list of the rounds to create history average features for
    :param temp_reviews: pdDataFrame with review_id, round_number, features as columns
    :param prefix_history_col_name: the prefix of the history column names
    :param prefix_future_col_name: the prefix of the future column names
    :return:
    """
    history_reviews = pd.DataFrame()
    future_reviews = pd.DataFrame()
    for round_num in rounds:
        review_id_curr_round = \
            temp_reviews.loc[temp_reviews.subsession_round_number == round_num].review_id
        review_id_curr_round.index = ['review_id']
        history = temp_reviews.loc[temp_reviews.subsession_round_number < round_num]
        future = temp_reviews.loc[temp_reviews.subsession_round_number > round_num]
        # history.shape[1]-2- no need subsession_round_number and review_id
        history_weights = list(pow(alpha_text, round_num - history.subsession_round_number))
        future_weights = list(pow(alpha_text, future.subsession_round_number - round_num))
        history = history.drop(['subsession_round_number', 'review_id'], axis=1)
        future = future.drop(['subsession_round_number', 'review_id'], axis=1)
        if alpha_text == 0:  # if alpha=0 use average
            history_mean = history.mean(axis=0)  # get the average of each column
            future_mean = future.mean(axis=0)
        else:
            history_mean = history.mul(history_weights, axis=0).sum()
            future_mean = future.mul(future_weights, axis=0).sum()
        # concat the review_id of the current round
        history_mean = history_mean.append(review_id_curr_round)
        history_reviews = pd.concat([history_reviews, history_mean], axis=1, ignore_index=True, sort=False)
        future_mean = future_mean.append(review_id_curr_round)
        future_reviews = pd.concat([future_reviews, future_mean], axis=1, ignore_index=True, sort=False)

    history_reviews = history_reviews.T
    history_reviews = rename_review_features_column(history_reviews, f'{prefix_history_col_name}_avg_feature')
    future_reviews = future_reviews.T
    future_reviews = rename_review_features_column(future_reviews, f'{prefix_future_col_name}_avg_feature')

    return history_reviews, future_reviews


def flat_reviews_numbers(data: pd.DataFrame, rounds: list, columns_to_drop: list, last_round_to_use: int,
                         first_round_to_use: int, total_payoff_label: bool=True, text_data: bool=False,
                         crf_raisha: bool=False, no_saifa_text: bool=False):
    """
    This function get data and flat it as for each row there is be the history of the relevant features in the data
    :param data: the data to flat
    :param rounds: the rounds to create history for
    :param columns_to_drop: the columns to drop, the first should be the column to add at last
    :param last_round_to_use: the last round to create history data for: 10 or 9
    :param first_round_to_use: the first round to use for history: the current round or the round before: 0 ot 1
    :param total_payoff_label: if the label is the decision of a single round
    :param text_data: if the data  we flat is the text representation
    :param crf_raisha: if we create data for crf with raisha features
    :param no_saifa_text: if we don't want to use the text of the saifa rounds
    :return:
    """
    all_history = pd.DataFrame()
    all_history_dict = dict()
    for round_num in rounds:
        id_curr_round = data.loc[data.subsession_round_number == round_num][columns_to_drop[0]]
        id_curr_round.index = ['review_id']
        # this features are not relevant for the last round because they are post treatment features
        data_to_flat = data.copy(deep=True)
        data_to_flat = data_to_flat.reset_index(drop=True)
        # the last round is not relevant for the history of any other rounds
        data_to_flat = data_to_flat.loc[data_to_flat.subsession_round_number <= last_round_to_use]
        data_to_flat = data_to_flat.drop(columns_to_drop, axis=1)
        # for current and next rounds put -1 --> use also current if first_round_to_use=0
        # and not use if first_round_to_use=1
        # if we predict all the future payoff, so the future text can be seen
        # if we want the text only from the raisha rounds and the current round - put -1 for the saifa
        if (not total_payoff_label and text_data) or (not text_data) or (text_data and no_saifa_text):
            data_to_flat.iloc[list(range(round_num - first_round_to_use, last_round_to_use))] = -1
        if crf_raisha:  # for saifa review put list of -1
            columns_to_use = data_to_flat.columns.tolist()
            columns_to_use.remove('review_features')
            for i in range(round_num - first_round_to_use, last_round_to_use):
                data_to_flat.at[i, 'review_features'] = [-1] * data.iloc[0]['review_features'].shape[0]
            raisha_data_list = list()
            for index, row in data_to_flat.iterrows():
                review_features = data_to_flat.review_features.iloc[index]
                if type(review_features) != list:
                    review_features = review_features.tolist()
                review_features.extend(data_to_flat[columns_to_use].iloc[index].to_list())
                raisha_data_list.extend(review_features)
            all_history_dict[round_num] = raisha_data_list
        else:
            data_to_flat.index = data_to_flat.index.astype(str)
            # data_to_flat.columns = data_to_flat.columns.astype(str)
            data_to_flat = data_to_flat.unstack().to_frame().sort_index(level=1).T
            data_to_flat.columns = [f'{str(col)}_{i}' for col, i in zip(data_to_flat.columns.get_level_values(0),
                                                                        data_to_flat.columns.get_level_values(1))]
            # concat the review_id of the current round
            data_to_flat = data_to_flat.assign(review_id=id_curr_round.values)
            all_history = pd.concat([all_history, data_to_flat], axis=0, ignore_index=True, sort=False)
    if crf_raisha:
        all_history = pd.Series(all_history_dict)
        all_history = pd.DataFrame(all_history)

    return all_history


def split_pairs_to_data_sets(load_file_name: str, k_folds: int=6):
    """
    Split all the pairs to data sets: train, validation, test for 6 folds
    :param load_file_name: the raw data file name
    :param k_folds: number of folds to split the data
    :return:
    """
    print(f'Start create and save data for file: {os.path.join(data_directory, f"{load_file_name}.csv")}')
    data = pd.read_csv(os.path.join(data_directory, f'{load_file_name}.csv'))
    data = data.loc[(data.status == 'play') & (data.player_id_in_group == 2)]
    data = data.drop_duplicates()
    pairs = pd.DataFrame(data.pair_id.unique(), columns=['pair_id'])
    pairs = pairs.sample(frac=1)
    pairs = pairs.assign(fold_number=0)
    paris_list = pairs.pair_id.unique()
    for k in range(k_folds):
        pairs.loc[pairs.pair_id.isin([x for i, x in enumerate(paris_list) if i % k_folds == k]), 'fold_number'] = k

    # split pairs to folds - train, test, validation in each fold
    for k in range(k_folds):
        pairs.loc[pairs.fold_number == k, f'fold_{k}'] = 'test'
        if k != k_folds-1:
            pairs.loc[pairs.fold_number == k + 1, f'fold_{k}'] = 'validation'
            pairs.loc[~pairs.fold_number.isin([k, k + 1]), f'fold_{k}'] = 'train'
        else:
            pairs.loc[pairs.fold_number == 0, f'fold_{k}'] = 'validation'
            pairs.loc[~pairs.fold_number.isin([k, 0]), f'fold_{k}'] = 'train'

    return pairs


class CreateSaveData:
    """
    This class load the data, create the seq data and save the new data with different range of K
    """
    def __init__(self, load_file_name: str, total_payoff_label: bool=True, label: str='total_payoff',
                 use_seq: bool=True, use_prev_round: bool=False, use_manual_features: bool=False, features_file: str='',
                 features_file_type: str='', use_all_history: bool = False, use_all_history_text_average: bool = False,
                 no_text: bool=False, use_score: bool=False, use_prev_round_text: bool=True,
                 use_all_history_text: bool=False, use_all_history_average: bool = False, use_crf_raisha: bool=False,
                 predict_first_round: bool=False, use_crf: bool=False, features_to_drop: list=None,
                 string_labels: bool=False, saifa_average_text: bool=False, no_saifa_text: bool=False,
                 saifa_only_prev_rounds_text: bool=False, use_prev_round_label: bool=False,
                 non_nn_turn_model: bool=False, transformer_model: bool=False):
        """
        :param load_file_name: the raw data file name
        :param total_payoff_label: if the label is the total payoff of the expert or the next rounds normalized payoff
        :param label: the name of the label
        :param use_seq: if to create a sample which is a seq or a single round
        :param use_prev_round: if to use the previous round data: review, decision, lottery result
        :param use_prev_round_label: if to use the previous round decision
        :param use_manual_features: if we use manual features - need to get the review id
        :param features_file: if using fix features- the name of the features file
        :param features_file_type: the type of file for the fix features
        :param no_text: if not using text as features
        :param use_score: if this data from the numeric condition
        :param use_prev_round_text: if to use the text features of the previous round
        :param use_all_history: if to add some numeric features regarding the history decisions and lottery
        :param use_all_history_average: if to add some numeric features regarding the history decisions and lottery as
        average over the history
        :param use_all_history_text: if to use all the history text features
        :param use_all_history_text_average: if to use the history text as average over all the history
        :param predict_first_round: if we want to predict the data of the first round
        :param use_crf: if we create data for crf model
        :param use_crf_raisha: if we create data for crf model with fixed raisha
        :param features_to_drop: a list of features to drop
        :param string_labels: if the labels are string --> for LSTM model
        :param saifa_average_text:  if we want to add the saifa average text features
        :param no_saifa_text: if we don't want to use the text of the saifa rounds
        :param saifa_only_prev_rounds_text: if we want to use only the previous rounds in the saifa
        :param non_nn_turn_model: non neural networks models that predict a label for each round
        :param transformer_model: create data for transformer model --> create features for raisha rounds too
        """
        print(f'Start create and save data for file: {os.path.join(data_directory, f"{load_file_name}.csv")}')
        logging.info('Start create and save data for file: {}'.
                     format(os.path.join(data_directory, f'{load_file_name}.csv')))

        # load the data and keep only play pairs and only one row for each round for each pair
        # columns_to_use = ['pair_id', 'status', 'subsession_round_number', 'group_sender_answer_reviews',
        #                   'group_receiver_choice', 'group_lottery_result', 'review_id',
        # 'previous_round_lottery_result', 'previous_round_decision', 'previous_review_id', 'group_average_score',
        # 'lottery_result_low', 'lottery_result_med1', 'lottery_result_high', 'previous_round_lottery_result_low',
        #                   'previous_round_lottery_result_med1', 'previous_round_lottery_result_high',
        # 'previous_group_average_score_low', 'previous_group_average_score_high', 'player_id_in_group']
        self.data = pd.read_csv(os.path.join(data_directory, f'{load_file_name}.csv'))  # , usecols=columns_to_use)
        print(f'Number of rows in data: {self.data.shape[0]}')
        self.data = self.data.loc[(self.data.status == 'play') & (self.data.player_id_in_group == 2)]
        print(f'Number of rows in data: {self.data.shape[0]} after keep only play and decision makers')
        self.data = self.data.drop_duplicates()
        print(f'Number of rows in data: {self.data.shape[0]} after drop duplicates')

        # get manual text features
        if use_manual_features:
            print(f'Load features from: {features_file}')
            if features_file_type == 'pkl':
                self.reviews_features = joblib.load(os.path.join(data_directory,
                                                                 f'{features_file}.{features_file_type}'))
            elif features_file_type == 'xlsx':
                self.reviews_features = pd.read_excel(os.path.join(data_directory,
                                                                   f'{features_file}.{features_file_type}'))
            else:
                print('Features file type is has to be pkl or xlsx')
                return
            # get manual text features
            if 'review' in self.reviews_features:
                self.reviews_features = self.reviews_features.drop('review', axis=1)
            if 'score' in self.reviews_features:
                self.reviews_features = self.reviews_features.drop('score', axis=1)

            if self.reviews_features.shape[1] == 2 and not use_seq:  # Bert features -> flat the vectors
                reviews = pd.DataFrame()
                for i in self.reviews_features.index:
                    temp = pd.DataFrame(self.reviews_features.at[i, 'review_features']).append(
                        pd.DataFrame([self.reviews_features.at[i, 'review_id']], index=['review_id']))
                    reviews = pd.concat([reviews, temp], axis=1, ignore_index=True)

                self.reviews_features = reviews.T
            else:  # manual features
                if features_to_drop is not None:
                    self.reviews_features = self.reviews_features.drop(features_to_drop, axis=1)

        # calculate expert total payoff --> the label
        self.data['exp_payoff'] = self.data.group_receiver_choice.map({1: 0, 0: 1})
        total_exp_payoff = self.data.groupby(by='pair_id').agg(
            total_exp_payoff=pd.NamedAgg(column='exp_payoff', aggfunc=sum))
        self.data = self.data.merge(total_exp_payoff, how='left', right_index=True, left_on='pair_id')
        self.data['10_result'] = np.where(self.data.group_lottery_result == 10, 1, 0)
        self.data = self.data[['pair_id', 'total_exp_payoff', 'subsession_round_number', 'group_sender_answer_reviews',
                               'exp_payoff', 'group_lottery_result', 'review_id', 'previous_round_lottery_result',
                               'previous_round_decision', 'previous_review_id', 'group_average_score',
                               'lottery_result_low', 'lottery_result_med1', 'previous_round_lottery_result_low',
                               'previous_round_lottery_result_high', 'previous_average_score_low',
                               'previous_average_score_high', 'previous_round_lottery_result_med1',
                               'group_sender_payoff', 'lottery_result_high',
                               'chose_lose', 'chose_earn', 'not_chose_lose', 'not_chose_earn',
                               'previous_score', 'group_sender_answer_scores', '10_result']]
        # 'time_spent_low', 'time_spent_high',
        self.final_data = pd.DataFrame()
        self.pairs = pd.Series(self.data.pair_id.unique())
        self.total_payoff_label = total_payoff_label
        self.label = label
        self.use_seq = use_seq
        self.use_prev_round = use_prev_round
        self.use_manual_features = use_manual_features
        self.number_of_rounds = 10  # if self.use_seq else 1
        self.features_file = features_file
        self.features_file_type = features_file_type
        self.use_all_history = use_all_history
        self.use_all_history_average = use_all_history_average
        self.use_all_history_text_average = use_all_history_text_average
        self.use_all_history_text = use_all_history_text
        self.no_text = no_text
        self.use_score = use_score
        self.use_prev_round_text = use_prev_round_text
        self.predict_first_round = predict_first_round
        self.saifa_average_text = saifa_average_text
        self.no_saifa_text = no_saifa_text
        self.saifa_only_prev_rounds_text = saifa_only_prev_rounds_text
        self.use_prev_round_label = use_prev_round_label
        self.non_nn_turn_model = non_nn_turn_model
        self.transformer_model = transformer_model
        # if we use the history average -> don't predict the first round because there is no history
        # if self.use_all_history_text_average or self.use_all_history_average:
        #     self.predict_first_round = False
        print(f'Number of pairs in data: {self.pairs.shape}')

        if self.use_all_history_average:
            self.set_all_history_average_measures()

        # create file names:
        file_name_component = [f'{self.label}_label_',
                               'seq_' if self.use_seq else '',
                               'crf_' if use_crf else '',
                               'crf_raisha_' if use_crf_raisha else '',
                               'string_labels_' if string_labels else '',
                               'non_nn_turn_model_' if self.non_nn_turn_model else '',
                               'transformer_model_' if self.transformer_model else '',
                               'prev_round_' if self.use_prev_round else '',
                               'prev_round_text_' if self.use_prev_round_text else '',
                               'prev_round_label_' if self.use_prev_round_label else '',
                               f'all_history_features_' if self.use_all_history else '',
                               f'all_history_features_average_with_global_alpha_{alpha_global}_'
                               if self.use_all_history_average else '',
                               f'all_history_text_average_with_alpha_{alpha_text}_' if
                               self.use_all_history_text_average else '',
                               f'no_saifa_text_' if self.no_saifa_text else '',
                               f'all_saifa_text_average_' if self.saifa_average_text else '',
                               'saifa_only_prev_rounds_text_' if self.saifa_only_prev_rounds_text else '',
                               f'all_history_text_' if self.use_all_history_text else '',
                               'no_text_' if self.no_text else '',
                               f'{self.features_file}_' if self.use_manual_features and not self.no_text else '',
                               'predict_first_round_' if self.predict_first_round else '',
                               f'{condition}_data']
        self.base_file_name = ''.join(file_name_component)
        print(f'Create data for: {self.base_file_name}')
        return

    def set_all_history_average_measures(self):
        """
        This function calculates some measures about all the history per round for each pair
        :return:
        """

        print('Start set_all_history_average_measures')
        columns_to_calc = ['group_lottery_result', 'group_sender_payoff', 'lottery_result_high', 'chose_lose',
                           'chose_earn', 'not_chose_lose', 'not_chose_earn', '10_result']
        rename_columns = ['lottery_result', 'decisions', 'lottery_result_high', 'chose_lose', 'chose_earn',
                          'not_chose_lose', 'not_chose_earn', '10_result']
        # Create only for the experts and then assign to both players
        columns_to_chose = columns_to_calc + ['pair_id', 'subsession_round_number']
        data_to_create = self.data[columns_to_chose]
        data_to_create.columns = rename_columns + ['pair_id', 'subsession_round_number']
        data_to_create = data_to_create.assign(history_decisions=None)
        data_to_create = data_to_create.assign(history_lottery_result_high=None)
        data_to_create = data_to_create.assign(history_lottery_result=None)
        data_to_create = data_to_create.assign(history_chose_lose=None)
        data_to_create = data_to_create.assign(history_chose_earn=None)
        data_to_create = data_to_create.assign(history_not_chose_lose=None)
        data_to_create = data_to_create.assign(history_not_chose_earn=None)
        data_to_create = data_to_create.assign(history_10_result=None)

        pairs = data_to_create.pair_id.unique()
        for pair in pairs:
            pair_data = data_to_create.loc[data_to_create.pair_id == pair]
            for round_num in range(2, 11):
                history = pair_data.loc[pair_data.subsession_round_number < round_num]
                weights = pow(alpha_global, round_num - history.subsession_round_number)
                for column in rename_columns:
                    if column == 'lottery_result':
                        j = 1
                    else:
                        j = 1
                    if alpha_global == 0:  # if alpha == 0: use average
                        data_to_create.loc[(data_to_create.pair_id == pair) &
                                           (data_to_create.subsession_round_number == round_num),
                                           f'history_{column}'] = round(history[column].mean(), 2)
                    else:
                        data_to_create.loc[(data_to_create.pair_id == pair) &
                                           (data_to_create.subsession_round_number == round_num),
                                           f'history_{column}'] = (pow(history[column], j) * weights).sum()
                    # for the first round put -1 for the history
                    data_to_create.loc[(data_to_create.pair_id == pair) &
                                       (data_to_create.subsession_round_number == 1), f'history_{column}'] = -1
        new_columns = [f'history_{column}' for column in rename_columns] + ['pair_id', 'subsession_round_number']
        self.data = self.data.merge(data_to_create[new_columns],  how='left')

        print('Finish set_all_history_average_measures')

        return

    def set_text_average(self, rounds, reviews_features, data, prefix_history_col_name: str='history',
                         prefix_future_col_name: str='future'):
        """
        Create data frame with the history and future average text features and the current round text features
        :param rounds: the rounds to use
        :param reviews_features: the reviews features of this pair
        :param data: the data we want to merge to
        :param prefix_history_col_name: the prefix of the history column names
        :param prefix_future_col_name: the prefix of the future column names
        :return:
        """
        history_reviews, future_reviews = create_average_history_text(rounds, reviews_features, prefix_history_col_name,
                                                                      prefix_future_col_name)
        # add the current round reviews features
        reviews_features = reviews_features.drop('subsession_round_number', axis=1)
        reviews_features = rename_review_features_column(reviews_features, 'curr_round_feature')
        data = data.merge(reviews_features, on='review_id', how='left')
        data = data.merge(history_reviews, on='review_id', how='left')
        if self.saifa_average_text:
            data = data.merge(future_reviews, on='review_id', how='left')

        return data

    def create_manual_features_data(self):
        """
        This function create 10 samples with different length from each pair data raw
        :return:
        """

        print(f'Start creating manual features data')
        logging.info('Start creating manual features data')

        # create the 10 samples for each pair
        meta_data_columns = [global_raisha, 'pair_id', 'sample_id']
        history_features_columns = ['history_lottery_result', 'history_decisions', 'history_lottery_result_high',
                                    'history_chose_lose', 'history_chose_earn', 'history_not_chose_lose']
        if self.use_prev_round:
            columns_to_use = ['review_id', 'previous_round_lottery_result_low',
                              'previous_round_lottery_result_med1', 'previous_round_lottery_result_high',
                              'previous_round_decision']
        else:
            columns_to_use = ['review_id']

        if self.use_prev_round_text:
            columns_to_use = columns_to_use + ['previous_review_id']

        if self.use_all_history_average:
            columns_to_use = columns_to_use + history_features_columns
        if self.no_text:
            columns_to_use.remove('review_id')
            if 'previous_review_id' in columns_to_use:
                columns_to_use.remove('previous_review_id')

        if self.use_score:  # if we use all the history so we will add it later
            columns_to_use.append('group_sender_answer_scores')
        if self.use_prev_round and condition == 'numeric':
            columns_to_use.append('previous_score')

        columns_to_use = columns_to_use + ['subsession_round_number']

        for pair in self.pairs:
            data = self.data.loc[self.data.pair_id == pair][columns_to_use]
            # don't use first round if use_prev_round or all_history unless we want to predict it
            if not self.predict_first_round:
                data = data.loc[data.subsession_round_number > 1]
                rounds = list(range(2, 11))
            else:
                rounds = list(range(1, 11))
            # concat history numbers
            if self.use_all_history:
                temp_numbers = self.data.loc[self.data.pair_id == pair][
                    ['10_result', 'group_sender_payoff', 'lottery_result_high',  # 'group_lottery_result',
                     'chose_lose', 'chose_earn', 'not_chose_lose', 'not_chose_earn', 'subsession_round_number']]
                temp_numbers = temp_numbers.reset_index(drop=True)
                all_history = flat_reviews_numbers(temp_numbers, rounds, columns_to_drop=['subsession_round_number'],
                                                   last_round_to_use=9, first_round_to_use=1)
                # first_round_to_use=1 because the numbers of the current round should no be in the features
                all_history.rename(columns={'review_id': 'subsession_round_number'}, inplace=True)
                data = data.merge(all_history, on='subsession_round_number', how='left')

            if self.use_score and self.use_all_history_text:
                temp_numbers = self.data.loc[self.data.pair_id == pair][
                    ['group_sender_answer_scores', 'subsession_round_number']]
                temp_numbers = temp_numbers.reset_index(drop=True)
                all_history = flat_reviews_numbers(temp_numbers, rounds, columns_to_drop=['subsession_round_number'],
                                                   last_round_to_use=10, first_round_to_use=0)
                all_history.rename(columns={'review_id': 'subsession_round_number'}, inplace=True)
                data = data.merge(all_history, on='subsession_round_number', how='left')
                data = data.drop('group_sender_answer_scores', axis=1)

            # first merge for the review_id for the current round
            if not self.no_text:  # use test features
                temp_reviews = data[['review_id', 'subsession_round_number']].copy(deep=True)
                temp_reviews = temp_reviews.merge(self.reviews_features, on='review_id', how='left')
                if self.use_all_history_text_average:
                    data = self.set_text_average(rounds, temp_reviews, data)
                elif self.use_all_history_text:
                    history_reviews = flat_reviews_numbers(
                        temp_reviews, rounds, columns_to_drop=['review_id', 'subsession_round_number'],
                        last_round_to_use=10, first_round_to_use=0, text_data=True,
                        total_payoff_label=self.total_payoff_label, no_saifa_text=self.no_saifa_text)
                    data = data.merge(history_reviews, on='review_id', how='left')
                else:  # no history text
                    temp_reviews =\
                        self.data.loc[self.data.pair_id == pair][['review_id', 'subsession_round_number']]
                    data = temp_reviews.merge(self.reviews_features, on='review_id', how='left')

                data = data.drop('review_id', axis=1)
                # first merge for the review_id for the previous round
                if self.use_prev_round_text:
                    prev_round_features = copy.deepcopy(self.reviews_features)
                    prev_round_features = rename_review_features_column(prev_round_features,
                                                                        'prev_round_features')
                    data = data.merge(prev_round_features, left_on='previous_review_id', right_on='review_id',
                                      how='left')
                    # drop columns not in used
                    data = data.drop('review_id', axis=1)
                    data = data.drop('previous_review_id', axis=1)

            else:  # no text
                data = data.reset_index(drop=True)

            # add metadata
            # raisha
            data[meta_data_columns[0]] = data['subsession_round_number'] - 1
            self.data['sample_id'] = self.data[meta_data_columns[1]] + '_' + (
                        self.data['subsession_round_number'] - 1).map(str)
            # add sample ID column
            data[meta_data_columns[1]] = pair
            data[meta_data_columns[2]] = data[meta_data_columns[1]] + '_' + data[meta_data_columns[0]].map(str)
            if self.label == 'single_round':
                # the label is the exp_payoff of the current round - 1 or -1
                data = data.merge(self.data[['exp_payoff', 'sample_id']], how='left', on='sample_id')
                data.rename(columns={'exp_payoff': self.label}, inplace=True)
                # if 1 not in data.subsession_round_number.values:
                #     data[self.label] =\
                #         self.data.loc[(self.data.pair_id == pair) & (self.data.subsession_round_number > 1)][
                #             'exp_payoff'].reset_index(drop=True)
                # else:
                #     data[self.label] = self.data.loc[(self.data.pair_id == pair)]['exp_payoff'].reset_index(drop=True)
                data[self.label] = np.where(data[self.label] == 1, 1, -1)

            else:  # the label is the total payoff
                # columns = [f'group_sender_payoff_{i}' for i in range(self.number_of_rounds - 1)]
                # if self.use_all_history:
                #     data[self.label] = (self.data.loc[self.data.pair_id == pair]['total_exp_payoff'].unique()[0] -
                #                         (data[columns] > 0).sum(axis=1)) / (self.number_of_rounds+1 -
                # data[global_raisha])
                # else:
                if self.predict_first_round:
                    rounds = list(range(1, 11))
                else:
                    rounds = list(range(2, 11))
                for i in rounds:  # the raishas are 0-9
                    data.loc[data.raisha == i-1, self.label] =\
                        self.data.loc[(self.data.pair_id == pair) &
                                      (self.data.subsession_round_number >= i)].group_sender_payoff.sum() /\
                        (self.number_of_rounds + 1 - i)
            # concat to all data
            self.final_data = pd.concat([self.final_data, data], axis=0, ignore_index=True)

        if 'subsession_round_number' in self.final_data.columns:
            self.final_data = self.final_data.drop('subsession_round_number', axis=1)
        # sort columns according to the round number
        if self.use_all_history_text or self.use_all_history:
            columns_to_sort = list(self.final_data.columns)
            meta_data_columns.append(self.label)
            columns_not_to_sort = copy.deepcopy(meta_data_columns)
            if self.use_all_history_average or self.use_all_history_text_average:
                columns_not_to_sort.extend(history_features_columns)
                columns = history_features_columns
            else:
                columns = list()
            for column in columns_not_to_sort:
                columns_to_sort.remove(column)
            columns_to_sort = sorted(columns_to_sort, key=lambda x: int(x[-1]))
            columns.extend(columns_to_sort)
            columns.extend(meta_data_columns)
            self.final_data = self.final_data[columns]

        # save column names to get the features later
        file_name = f'features_{self.base_file_name}'
        features_columns = self.final_data.columns.tolist()
        columns_to_drop = meta_data_columns + ['subsession_round_number', self.label]
        for column in columns_to_drop:
            if column in features_columns:
                features_columns.remove(column)
        pd.DataFrame(features_columns).to_excel(os.path.join(save_data_directory, f'{file_name}.xlsx'), index=True)

        # save final data
        file_name = f'all_data_{self.base_file_name}'
        self.final_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'Finish creating manual features data: {file_name}')
        logging.info('Finish creating manual features data')

        return

    def create_manual_features_seq_data(self):
        """
        This function create 10 samples with different length from each pair data raw
        :return:
        """

        print(f'Start creating sequences with different lengths and concat')
        logging.info('Start creating sequences with different lengths and concat')

        # create the 10 samples for each pair
        meta_data_columns = [global_raisha, 'pair_id', 'sample_id']
        if self.use_prev_round:
            columns_to_use = ['review_features', 'previous_round_decision', 'previous_round_lottery_result_low',
                              'previous_round_lottery_result_med1', 'previous_round_lottery_result_high',
                              'previous_average_score_low', 'previous_round_lottery_result_med1',
                              'previous_average_score_high']
        else:
            columns_to_use = ['review_features']

        if self.use_all_history:
            columns_to_use = columns_to_use + ['history_lottery_result', 'history_decisions',
                                               'history_lottery_result_high', 'history_chose_lose',
                                               'history_chose_earn', 'history_not_chose_lose',]
            #  'time_spent_low', 'time_spent_high

        if self.no_text:
            columns_to_use.remove('review_features')

        if self.reviews_features.shape[1] > 2:  # if the review_features is in one column - no need to create it
            # create features as one column with no.array for NN models
            temp_features = self.reviews_features.copy(deep=True).drop('review_id', axis=1)
            self.reviews_features = self.reviews_features.assign(review_features='')
            for i in self.reviews_features.index:
                self.reviews_features.at[i, 'review_features'] = np.array(temp_features.values)[i]
        self.reviews_features = self.reviews_features[['review_id', 'review_features']]
        # first merge for the review_id for the current round
        data_for_pairs = self.data.merge(self.reviews_features, left_on='review_id', right_on='review_id', how='left')
        data_for_pairs = data_for_pairs.drop('review_id', axis=1)

        for pair in self.pairs:
            # get a row with the subsession_round_number and one with the group_sender_answer_reviews
            # self.use_seq: sample use all rounds data
            print(f'{time.asctime(time.localtime(time.time()))}: Start create data for pair {pair}')
            data_column = dict()
            for column in columns_to_use:
                data_pair = data_for_pairs.loc[data_for_pairs.pair_id == pair][['subsession_round_number', column]].\
                    transpose()
                # define the first row as the header
                data_pair.columns = data_pair.loc['subsession_round_number'].astype(int) - 1
                data_pair = data_pair.drop(['subsession_round_number'])
                for col in data_pair.columns:
                    data_column[f'{column}_{col}'] =\
                        np.concatenate([np.repeat(data_pair[col].values, self.number_of_rounds-col),
                                        np.repeat(np.nan, col)])
            data = pd.DataFrame(data_column)

            # this is for 3 losses --> the lottery result and the expected_dm_payoff of the current round
            data['curr_result'] =\
                self.data.loc[self.data.pair_id == pair]['group_lottery_result'].reset_index(drop=True)
            data['curr_expected_dm_payoff'] =\
                self.data.loc[self.data.pair_id == pair]['group_average_score'].reset_index(drop=True)

            # define raisha
            data[meta_data_columns[0]] = data.index
            data[meta_data_columns[0]] = self.number_of_rounds - data[meta_data_columns[0]]
            # add sample ID column
            data[meta_data_columns[1]] = pair
            data[meta_data_columns[2]] = data[meta_data_columns[1]] + '_' + data[meta_data_columns[0]].map(str)

            # add label column
            columns = [f'exp_payoff_{i}' for i in range(self.number_of_rounds)]
            if self.total_payoff_label:
                # Create label that will be the expert's total payoff over the 10 trails for all the pair's data
                # (with different seq lengths)
                data['label'] = self.data.loc[self.data.pair_id == pair]['total_exp_payoff'].unique()[0]
            else:
                if self.label == 'single_round':
                    # the label is the exp_payoff of the current round - 1 or -1
                    data['label'] = self.data.loc[self.data.pair_id == pair]['exp_payoff'].reset_index(drop=True)
                    data.label = np.where(data.label == 1, 1, -1)

                else:  # the label is the total payoff
                    data['label'] = (self.data.loc[self.data.pair_id == pair]['total_exp_payoff'].unique()[0] - data[
                        columns].sum(axis=1)) / (self.number_of_rounds - data[global_raisha])

            if self.label == 'next_percent':
                data['label'] = data['label']*100

            # concat to all data
            self.final_data = pd.concat([self.final_data, data], axis=0, ignore_index=True)

        # rename columns for convenient:
        data_columns = ['text', 'prev_payoff', 'prev_result_low',
                        'prev_result_med1', 'prev_result_high',
                        'prev_expected_dm_payoff_low', 'prev_expected_dm_payoff_med1', 'prev_expected_dm_payoff_high',
                        'history_lottery_result', 'history_decisions', 'history_lottery_result_high',
                        'history_chose_lose', 'history_chose_earn', 'history_not_chose_lose', 'time_spent_low',
                        'time_spent_high']
        columns = list()
        for column in data_columns:
            columns.extend([f'{column}_{i}' for i in range(self.number_of_rounds)])

        columns.extend(['curr_result', 'curr_expected_dm_payoff'])
        columns.extend(meta_data_columns)
        columns.append('label')
        self.final_data.columns = columns

        # TODO: remove prev_... for the first round

        if self.label == 'single_round':  # create label as numpy array
            help_data_np = np.where(self.final_data.label == 1, 1, 0)
            help_np = np.zeros((self.final_data.label.shape[0], int(self.final_data.label.max()) + 1))
            help_np[np.arange(help_data_np.shape[0]), help_data_np] = 1
            help_np = sparse.coo_matrix(help_np)
            self.final_data['single_label'] = self.final_data.label
            self.final_data.label = help_np.toarray().tolist()

        file_name = f'all_data_{self.base_file_name}'
        print(f'{time.asctime(time.localtime(time.time()))}: Save all data')
        self.final_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'{time.asctime(time.localtime(time.time()))}:'
              f'Finish creating sequences with different lengths and concat with manual features for the text')
        logging.info(f'{time.asctime(time.localtime(time.time()))}:'
                     f'Finish creating sequences with different lengths and concat features for the text')

        return

    def create_manual_features_crf_data(self):
        """
        This function create 10 samples with different length from each pair data raw for crf model
        :return:
        """

        print(f'Start creating sequences with different lengths and concat for crf')
        logging.info('Start creating sequences with different lengths and concat for crf')

        columns_to_use = ['previous_round_lottery_result_low', 'previous_round_lottery_result_med1',
                          'previous_round_lottery_result_high', 'previous_average_score_low',
                          'previous_round_lottery_result_med1', 'previous_average_score_high']

        # columns for raisha data
        columns_to_flat = copy.deepcopy(self.reviews_features.columns.to_list())
        columns_to_flat.extend(['exp_payoff', 'lottery_result_high', 'lottery_result_low',
                                'lottery_result_med1', 'chose_lose', 'chose_earn', 'not_chose_lose',
                                'not_chose_earn'])
        columns_to_flat.remove('review_id')
        raisha_data_columns = sorted([f'{column}_{j}' for column, j in
                                      list(itertools.product(*[columns_to_flat, list(range(10))]))],
                                     key=lambda x: int(x[-1]))
        all_columns = self.reviews_features.columns.to_list() + columns_to_use + raisha_data_columns
        all_columns.remove('review_id')
        file_name = f'features_{self.base_file_name}'
        pd.DataFrame(all_columns).to_excel(os.path.join(save_data_directory, f'{file_name}.xlsx'), index=True)

        if self.reviews_features.shape[1] > 2:  # if the review_features is in one column - no need to create it
            # create features as one column with no.array for CRF models
            temp_features = self.reviews_features.copy(deep=True).drop('review_id', axis=1)
            self.reviews_features = self.reviews_features.assign(review_features='')
            for i in self.reviews_features.index:
                self.reviews_features.at[i, 'review_features'] = np.array(temp_features.values)[i]
        self.reviews_features = self.reviews_features[['review_id', 'review_features']]
        # first merge for the review_id for the current round
        data_for_pairs = self.data.merge(self.reviews_features, left_on='review_id', right_on='review_id', how='left')

        for pair in self.pairs:
            # get a row with the subsession_round_number and one with the group_sender_answer_reviews
            # self.use_seq: sample use all rounds data
            print(f'{time.asctime(time.localtime(time.time()))}: Start create data for pair {pair}')
            data_column = dict()
            data_pair = data_for_pairs.loc[data_for_pairs.pair_id == pair]

            if self.use_all_history and self.use_all_history_text:  # create the raisha history
                columns_to_flat = copy.deepcopy(self.reviews_features.columns.to_list())
                columns_to_flat.extend(['exp_payoff', 'lottery_result_high', 'lottery_result_low',
                                        'lottery_result_med1', 'chose_lose', 'chose_earn', 'not_chose_lose',
                                        'not_chose_earn', 'subsession_round_number'])
                data_to_flat = data_pair[columns_to_flat]
                raisha_data = flat_reviews_numbers(data_to_flat, rounds=list(range(1, 11)),
                                                   columns_to_drop=['review_id', 'subsession_round_number'],
                                                   last_round_to_use=10, first_round_to_use=0, text_data=False,
                                                   total_payoff_label=self.total_payoff_label, crf_raisha=True)
                # raisha_data = raisha_data.drop('review_id', axis=1)
                columns_to_flat.remove('review_id')
                columns_to_flat.remove('subsession_round_number')

            data_pair = data_pair.drop('review_id', axis=1)
            pair_id = data_for_pairs.loc[data_for_pairs.pair_id == pair].pair_id.reset_index(drop=True)
            all_features_pair = list()
            for index, row in data_pair.iterrows():
                all_features_pair.append(list(np.append(row.review_features, row[columns_to_use])))
            # create df with list of vectors in each cell
            data_pair_df = pd.DataFrame(pd.Series(all_features_pair)).transpose()
            data_pair_df.columns = list(range(10))
            # duplicate the values to create seq
            for col in data_pair_df.columns:
                data_column[f'features_{col+1}'] =\
                    np.concatenate([np.repeat(data_pair_df[col].values, self.number_of_rounds-col),
                                    np.repeat(np.nan, col)])
            data = pd.DataFrame(data_column)

            # add labels column
            if self.total_payoff_label:  # TODO: change it
                # Create label that will be the expert's total payoff over the 10 trails for all the pair's data
                # (with different seq lengths)
                data['label'] = self.data.loc[self.data.pair_id == pair]['total_exp_payoff'].unique()[0]
            else:
                if self.label == 'single_round':
                    # the label is the exp_payoff of the current round - 1 or -1
                    single_label = pd.Series(np.where(data_pair.exp_payoff == 1, 1, -1))
                    # single_label = data_pair.subsession_round_number
                    labels = pd.DataFrame(pd.Series(
                        [single_label.iloc[:self.number_of_rounds - i].values.tolist() for i in range(10)]),
                        columns=[crf_label_col_name])
                    data = data.merge(labels, right_index=True, left_index=True)

                else:  # the label is the total payoff
                    # TODO: change it
                    columns = [f'exp_payoff_{i}' for i in range(self.number_of_rounds)]
                    data['label'] = (self.data.loc[self.data.pair_id == pair]['total_exp_payoff'].unique()[0] - data[
                        columns].sum(axis=1)) / (self.number_of_rounds - data[global_raisha])

            if self.label == 'next_percent':
                # TODO: change it
                data['label'] = data['label']*100

            # add pair id
            data = data.merge(pair_id, right_index=True, left_index=True)

            if self.use_all_history_text and self.use_all_history:  # create data for raisha
                data_to_use = data.iloc[0]
                for raisha in range(0, 10):
                    # the saifa data is the data of the rounds after the saifa
                    saifa_data_columns = [f'features_{i}' for i in range(raisha+1, 11)]
                    saifa_data_columns.extend([crf_label_col_name, 'pair_id'])
                    relevant_data_for_raisha_saifa = data_to_use[saifa_data_columns]
                    relevant_data_for_raisha_saifa[global_raisha] = raisha
                    relevant_data_for_raisha_saifa[crf_label_col_name] =\
                        relevant_data_for_raisha_saifa[crf_label_col_name][raisha:]
                    if raisha == 0:
                        self.final_data = pd.concat([self.final_data, pd.DataFrame([relevant_data_for_raisha_saifa])],
                                                    axis=0, ignore_index=True)
                    else:
                        relevant_data_for_raisha_saifa.iloc[0].extend(raisha_data.iloc[raisha].to_list()[0])
                        self.final_data = pd.concat([self.final_data, pd.DataFrame([relevant_data_for_raisha_saifa])],
                                                    axis=0, ignore_index=True)

            else:
                # concat to all data
                self.final_data = pd.concat([self.final_data, data], axis=0, ignore_index=True)

        file_name = f'all_data_{self.base_file_name}'
        print(f'{time.asctime(time.localtime(time.time()))}: Save all data')
        # save_data = self.final_data.drop(['pair_id'], axis=1)
        save_data = self.final_data
        save_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(save_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'{time.asctime(time.localtime(time.time()))}:'
              f'Finish creating sequences with different lengths and concat with manual features for the text')
        logging.info(f'{time.asctime(time.localtime(time.time()))}:'
                     f'Finish creating sequences with different lengths and concat features for the text')

        return

    def create_manual_features_crf_raisha_data(self, string_labels=False):
        """
        This function create 10 samples with different length from each pair data raw for crf model.
        The raisha text and decisions and the text of the relevant history in the safia represent in each round data
        :return:
        """

        print(f'Start creating sequences with different lengths and concat for crf')
        logging.info('Start creating sequences with different lengths and concat for crf')

        # only for single_round label
        if self.label != 'single_round':
            print('CRF raisha function works only with single_round label!')
            return

        text_columns = copy.deepcopy(self.reviews_features.columns.to_list())
        text_columns.remove('review_id')
        decisions_payoffs_columns = ['exp_payoff', 'lottery_result_high', 'lottery_result_low',
                                     'lottery_result_med1', 'chose_lose', 'chose_earn', 'not_chose_lose',
                                     'not_chose_earn']
        if self.non_nn_turn_model:
            decisions_payoffs_columns.remove('exp_payoff')
        columns_to_flat = decisions_payoffs_columns + text_columns
        all_columns = sorted([f'{column}_{j}' for column, j in
                              list(itertools.product(*[columns_to_flat, list(range(0, 10))]))],
                             key=lambda x: int(x[-1]))
        # labels columns:
        label_column = ['round_label']
        label_columns = sorted([f'{column}_{j}' for column, j in
                                list(itertools.product(*[label_column, list(range(0, 10))]))], key=lambda x: int(x[-1]))

        # merge the round number features with the review features
        data_for_pairs = self.data.merge(self.reviews_features, on='review_id', how='left')

        prev_round_text_columns = list()
        prev_round_columns = list()
        # if not self.use_all_history and not self.use_all_history_average and \
        #         not self.use_all_history_text and not self.use_all_history_text_average:  # no history at all
        no_history_all_columns = [f'curr_round_{column}' for column in text_columns]
        if self.use_prev_round_text:
            prev_round_text_columns.extend([f'prev_round_{column}' for column in text_columns])
        if self.use_prev_round:
            prev_round_columns.extend([f'prev_round_{column}' for column in decisions_payoffs_columns])
        prev_round_data = data_for_pairs[decisions_payoffs_columns + ['pair_id']].copy(deep=True)

        file_name = f'features_{self.base_file_name}'
        pd.DataFrame(prev_round_columns + prev_round_text_columns + all_columns).to_excel(
            os.path.join(save_data_directory, f'{file_name}.xlsx'), index=True)

        data_for_pairs['round_label'] = np.where(data_for_pairs.exp_payoff == 1, 1, -1)
        data_for_pairs = data_for_pairs[columns_to_flat + label_column + ['pair_id']]
        for pair in self.pairs:
            print(f'{time.asctime(time.localtime(time.time()))}: Start create data for pair {pair}')
            data_pair_label = data_for_pairs.loc[data_for_pairs.pair_id == pair].copy(deep=True)
            prev_round_data_pair = prev_round_data.loc[prev_round_data.pair_id == pair].copy(deep=True)
            prev_round_data_pair = prev_round_data_pair.drop('pair_id', axis=1)

            # create the data for no history:
            data_pair_label_no_history = data_pair_label.copy(deep=True)
            data_pair_label_no_history.columns = [f'curr_round_{column}' for column in
                                                  data_pair_label_no_history.columns]
            data_pair_no_history = data_pair_label_no_history[no_history_all_columns].copy(deep=True)

            # create a vector for each pair with the text of all the rounds and the decision of rounds 1-9
            data_pair_label = data_pair_label.drop('pair_id', axis=1)
            data_pair_label = data_pair_label.reset_index(drop=True)
            data_pair_label = data_pair_label.unstack().to_frame().sort_index(level=1).T
            data_pair_label.columns = \
                [f'{str(col)}_{i}' for col, i in zip(data_pair_label.columns.get_level_values(0),
                                                     data_pair_label.columns.get_level_values(1))]
            label_pair = data_pair_label[label_columns].copy(deep=True)
            # create data for use history
            data_pair = data_pair_label[all_columns].copy(deep=True)

            if not self.use_all_history and not self.use_all_history_average and \
                    not self.use_all_history_text and not self.use_all_history_text_average:  # no history at all
                # create the labels
                label_pair = data_pair_label_no_history[['curr_round_round_label']].copy(deep=True)
                label_pair.columns = ['round_label']
                label_pair = label_pair.reset_index(drop=True)
                label_pair = label_pair.unstack().to_frame().sort_index(level=1).T
                label_pair.columns = [f'{str(col)}_{i}' for col, i in zip(label_pair.columns.get_level_values(0),
                                                                          label_pair.columns.get_level_values(1))]

            # create data per raisha with all data of the raisha and all the text of the history in the saifa
            raisha_data_dict = defaultdict(dict)

            for raisha in range(0, 10):
                # add metadata and labels
                raisha_data_dict[raisha][global_raisha] = raisha
                raisha_data_dict[raisha]['pair_id'] = pair
                raisha_data_dict[raisha]['sample_id'] = f'{pair}_{raisha}'

                if string_labels:
                    raisha_data_dict[raisha][crf_label_col_name] = \
                        np.where(label_pair[[f'round_label_{i}' for i in range(raisha, 10)]] == 1,
                                 'hotel', 'stay_home')[0].tolist()

                else:
                    # only for single_round label
                    raisha_data_dict[raisha][crf_label_col_name] = \
                        label_pair[[f'round_label_{i}' for i in range(raisha, 10)]].astype(int).values.tolist()[0]

                # all the saifa decisions and payoffs (raisha=3, so all the decisions of rounds 4-10)
                raisha_columns_minus_1 = \
                    sorted([f'{column}_{j}' for column, j in
                            list(itertools.product(*[decisions_payoffs_columns, list(range(raisha, 10))]))],
                           key=lambda x: int(x[-1]))
                for round_num in range(0, 10):
                    # the raisha data --> no need to put because we don't predict these rounds
                    if round_num < raisha and not self.transformer_model:
                        raisha_data_dict[raisha][f'features_round_{round_num+1}'] = None
                    elif round_num < raisha and self.transformer_model:
                        # raisha features will be the columns_to_flat of the current round
                        raisha_columns = [f'{column}_{round_num}' for column in columns_to_flat]
                        round_raisha_data = copy.deepcopy(data_pair)
                        round_raisha_data = round_raisha_data[raisha_columns]
                        round_raisha_data = round_raisha_data.values.tolist()[0]
                        raisha_data_dict[raisha][f'features_round_{round_num+1}'] = round_raisha_data

                    else:
                        if not self.use_all_history and not self.use_all_history_average and \
                                not self.use_all_history_text and not self.use_all_history_text_average:  # no -1 need
                            round_raisha_data = copy.deepcopy(data_pair_no_history)
                            round_raisha_data = round_raisha_data.values.tolist()[round_num]

                            if self.use_prev_round_text:
                                if round_num == 0:  # just get list of -1
                                    prev_round_text = [-1] * data_pair.shape[1]
                                else:
                                    prev_round_text = copy.deepcopy(data_pair)
                                    prev_round_text = prev_round_text.values.tolist()[round_num-1]
                                round_raisha_data = prev_round_text + round_raisha_data
                            if self.use_prev_round:
                                if round_num == 0:
                                    prev_round_num = [-1] * prev_round_data_pair.shape[1]
                                else:
                                    prev_round_num = copy.deepcopy(prev_round_data_pair)
                                    prev_round_num = prev_round_num.values.tolist()[round_num - 1]
                                round_raisha_data = prev_round_num + round_raisha_data

                            raisha_data_dict[raisha][f'features_round_{round_num+1}'] = round_raisha_data
                            continue
                        # the saifa data --> put all the raisha data + the history in the saifa text + the curr
                        # round text
                        else:
                            if self.saifa_only_prev_rounds_text:
                                # all the future of the saifa text (round=5 --> so we don't use the text of rounds 6-10)
                                round_columns_minus_1 = \
                                    sorted([f'{column}_{j}' for column, j in
                                            list(itertools.product(*[text_columns, list(range(round_num+1, 10))]))],
                                           key=lambda x: int(x[-1]))
                            # if we want to use only one previous round text in the saifa
                            elif self.use_prev_round_text:
                                round_columns_minus_1 = \
                                    sorted([f'{column}_{j}' for column, j in
                                            list(itertools.product(*[
                                                text_columns,
                                                # range(raisha, round_num-1) --> the rounds in the saifa that are not
                                                # the current round and the one previous round
                                                # range(round_num, 10) --> the future rounds in the saifa
                                                list(range(raisha, round_num-1)) + list(range(round_num+1, 10))]))],
                                           key=lambda x: int(x[-1]))
                            elif self.no_saifa_text:  # use only the raisha rounds and the current round text data
                                round_columns_minus_1 = \
                                    sorted([f'{column}_{j}' for column, j in
                                            list(itertools.product(*[
                                                text_columns,
                                                # range(raisha, round_num) --> the rounds in the saifa that are not
                                                # the current round
                                                # range(round_num+1, 10) --> the future rounds in the saifa
                                                list(range(raisha, round_num)) + list(range(round_num+1, 10))]))],
                                           key=lambda x: int(x[-1]))
                            else:  # use all the saifa rounds text
                                round_columns_minus_1 = list()
                        round_columns_minus_1.extend(raisha_columns_minus_1)
                        # put -1 in these columns
                        round_raisha_data = copy.deepcopy(data_pair)
                        round_raisha_data[round_columns_minus_1] = -1
                        round_raisha_data = round_raisha_data.values.tolist()[0]
                        raisha_data_dict[raisha][f'features_round_{round_num+1}'] = round_raisha_data

            pair_raisha_data = pd.DataFrame.from_dict(raisha_data_dict).T
            self.final_data = pd.concat([self.final_data, pair_raisha_data], axis=0, ignore_index=True)

        file_name = f'all_data_{self.base_file_name}'
        # save_data = self.final_data.drop(['pair_id'], axis=1)
        if not self.non_nn_turn_model:
            print(f'{time.asctime(time.localtime(time.time()))}: Save all data {file_name}.pkl')
            save_data = self.final_data
            save_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
            joblib.dump(save_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'{time.asctime(time.localtime(time.time()))}: '
              f'Finish creating sequences with different lengths and concat with manual features for the text')
        logging.info(f'{time.asctime(time.localtime(time.time()))}: '
                     f'Finish creating sequences with different lengths and concat features for the text')

        print(f'max features num is: '
              f'{max([max(pair_raisha_data[f"features_round_{i}"].str.len()) for i in range(1, 11)])}')

        return

    def create_manual_features_crf_raisha_data_avg_features(self, string_labels=False):
        """
        This function create 10 samples with different length from each pair data raw for crf model.
        The raisha text and decisions and the text of the relevant history in the safia represent in each round data
        :return:
        """

        print(f'Start create_manual_features_crf_raisha_data_avg_features')
        logging.info('Start create_manual_features_crf_raisha_data_avg_features')

        # only for single_round label
        if self.label != 'single_round':
            print('CRF raisha function works only with single_round label!')
            return

        decisions_payoffs_columns = ['exp_payoff', 'lottery_result_high', 'lottery_result_low',
                                     'lottery_result_med1', 'chose_lose', 'chose_earn', 'not_chose_lose',
                                     'not_chose_earn', 'pair_id', 'subsession_round_number']
        decisions_payoffs_data = self.data[decisions_payoffs_columns].copy(deep=True)
        decisions_payoffs_columns.remove('pair_id')
        decisions_payoffs_columns.remove('subsession_round_number')

        # columns for raisha data
        all_columns = [column for column in self.data.columns if 'history' in column]
        all_columns.extend(['pair_id', 'exp_payoff', 'review_id'])
        if self.use_prev_round_text:
            all_columns.append('previous_review_id')
        data_for_pairs = self.data[all_columns].copy(deep=True)

        labels_for_pairs = self.data[['exp_payoff', 'pair_id', 'subsession_round_number']].copy(deep=True)
        labels_for_pairs['round_label'] = np.where(labels_for_pairs.exp_payoff == 1, 1, -1)
        raisha_columns, round_columns, prev_round_columns = list(), list(), list()

        for pair in self.pairs:
            print(f'{time.asctime(time.localtime(time.time()))}: Start create data for pair {pair}')
            data_pair = data_for_pairs.loc[data_for_pairs.pair_id == pair].copy(deep=True)
            label_pair = labels_for_pairs.loc[labels_for_pairs.pair_id == pair].copy(deep=True)

            # use the raisha (and) the saifa average text and the current round text
            data = self.data.loc[self.data.pair_id == pair][['review_id', 'subsession_round_number']].copy(deep=True)
            temp_reviews = data.merge(self.reviews_features, on='review_id', how='left')
            # return for each round the curr round features, the history rounds features and the future
            # rounds history (and the review_id, subsession_round_number columns)
            avg_text_data = self.set_text_average(list(range(1, 11)), temp_reviews, data,
                                                  prefix_history_col_name=global_raisha,
                                                  prefix_future_col_name=global_saifa)
            data_pair = data_pair.merge(avg_text_data, on='review_id', how='left')

            if self.use_prev_round_text:  # add the previous round in saifa text
                prev_round_saifa_text = self.reviews_features.copy(deep=True)
                prev_round_saifa_text = rename_review_features_column(prev_round_saifa_text, 'prev_round_feature')
                data_pair = data_pair.merge(prev_round_saifa_text, left_on='previous_review_id',
                                            right_on='review_id', how='left')
                data_pair.drop('previous_review_id', axis=1)

            # create data per raisha with all data of the raisha and all the text of the history in the saifa
            raisha_data_dict = defaultdict(dict)

            for raisha in range(0, 10):
                # add metadata and labels
                raisha_data_dict[raisha][global_raisha] = raisha
                raisha_data_dict[raisha]['pair_id'] = pair
                raisha_data_dict[raisha]['sample_id'] = f'{pair}_{raisha}'

                # use the raisha (and) the saifa average text and the current round text
                saifa_labels = label_pair.loc[label_pair.subsession_round_number.isin(
                    list(range(raisha+1, 11)))].round_label.copy(deep=True)
                if string_labels:
                    raisha_data_dict[raisha][crf_label_col_name] =\
                        np.where(saifa_labels == 1, 'hotel', 'stay_home').tolist()
                else:
                    # only for single_round label
                    raisha_data_dict[raisha][crf_label_col_name] = saifa_labels.astype(int).values.tolist()

                # the raisha data is the history when subsession_round_number = raisha
                raisha_saifa_data = data_pair.loc[data_pair.subsession_round_number == raisha+1]
                raisha_columns = [column for column in raisha_saifa_data.columns if global_raisha in column or
                                  'history' in column]
                if self.saifa_average_text:
                    saifa_columns = [column for column in raisha_saifa_data.columns if global_saifa in column]
                    raisha_columns.extend(saifa_columns)
                raisha_saifa_data = raisha_saifa_data[raisha_columns]
                raisha_saifa_data_list = raisha_saifa_data.values.tolist()[0]
                for round_num in range(0, 10):
                    # the raisha data --> no need to put because we don't predict these rounds
                    if round_num < raisha and not self.transformer_model:
                        raisha_data_dict[raisha][f'features_round_{round_num+1}'] = None
                    elif round_num < raisha and self.transformer_model:
                        # raisha features will be the columns_to_flat of the current round
                        round_data = data_pair.loc[data_pair.subsession_round_number == round_num+1]
                        raisha_columns = [column for column in round_data.columns if 'curr_round_feature' in column]
                        round_raisha_data = copy.deepcopy(round_data)
                        round_raisha_data = round_raisha_data.merge(
                            decisions_payoffs_data, on=['subsession_round_number', 'pair_id', 'exp_payoff'])
                        round_raisha_data = round_raisha_data[raisha_columns + decisions_payoffs_columns]
                        round_raisha_data = round_raisha_data.values.tolist()[0]
                        raisha_data_dict[raisha][f'features_round_{round_num+1}'] = round_raisha_data
                    else:
                        round_data = data_pair.loc[data_pair.subsession_round_number == round_num+1]
                        round_columns = [column for column in round_data.columns if 'curr_round_feature' in column]
                        curr_round_data = round_data[round_columns].copy(deep=True)
                        round_data_list = curr_round_data.values.tolist()[0]
                        if self.use_prev_round_text:
                            prev_round_columns = [column for column in round_data.columns if 'prev_round' in column]
                            prev_round_data = round_data[prev_round_columns].copy(deep=True)
                            # the first round in the saifa don't have prev round
                            if round_num == raisha:
                                prev_round_data[prev_round_columns] = -1
                            prev_round_data_list = prev_round_data.values.tolist()[0]
                            round_data_list += prev_round_data_list

                        # concat raisha, current round and saifa data
                        all_data = raisha_saifa_data_list + round_data_list
                        raisha_data_dict[raisha][f'features_round_{round_num+1}'] = all_data

            pair_raisha_data = pd.DataFrame.from_dict(raisha_data_dict).T
            self.final_data = pd.concat([self.final_data, pair_raisha_data], axis=0, ignore_index=True)

        # save features for use_all_history_text_average
        use_all_history_text_average_columns = raisha_columns + round_columns + prev_round_columns
        file_name = f'features_{self.base_file_name}'
        pd.DataFrame(use_all_history_text_average_columns).to_excel(
            os.path.join(save_data_directory, f'{file_name}.xlsx'), index=True)

        file_name = f'all_data_{self.base_file_name}'
        # save_data = self.final_data.drop(['pair_id'], axis=1)
        if not self.non_nn_turn_model:
            print(f'{time.asctime(time.localtime(time.time()))}: Save all data {file_name}.pkl')
            save_data = self.final_data
            save_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
            joblib.dump(save_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'{time.asctime(time.localtime(time.time()))}:Finish create_manual_features_crf_raisha_data_avg_features')
        logging.info(f'{time.asctime(time.localtime(time.time()))}:'
                     f'Finish create_manual_features_crf_raisha_data_avg_features')

        print(f'max features num is: '
              f'{max([max(pair_raisha_data[f"features_round_{i}"].str.len()) for i in range(1, 11)])}')

        return

    def create_non_nn_turn_model_features(self):
        """
        This function flat the crf raisha data to match it to the non nn turn model features
        :return:
        """

        print(f'{time.asctime(time.localtime(time.time()))}: Start create_non_nn_turn_model_features')
        logging.info(f'{time.asctime(time.localtime(time.time()))}: Start create_non_nn_turn_model_features')

        if self.use_prev_round_label:
            features_file = pd.read_excel(os.path.join(save_data_directory, f'features_{self.base_file_name}.xlsx'))[0]
            features_file = features_file.append(pd.Series('prev_round_label', index=[len(features_file.index)]))
            features_file.to_excel(os.path.join(save_data_directory, f'features_{self.base_file_name}.xlsx'), index=True)

        data_to_flat = self.final_data.copy(deep=True)
        all_flat_data = defaultdict(dict)
        for index, row in data_to_flat.iterrows():
            labels = row[crf_label_col_name]
            raisha = row[global_raisha]
            for label_index, round_number in enumerate(range(raisha+1, 11)):
                # features columns are features_round_1,..., features_round_10
                row_features = row[f'features_round_{round_number}']
                if self.use_prev_round_label:
                    if label_index == 0:
                        label_to_append = -1
                    elif labels[label_index-1] == -1:
                        label_to_append = 0
                    elif labels[label_index-1] == 1:
                        label_to_append = 1
                    else:
                        print(f'Error: labels[label_index-1] = {labels[label_index-1]}')
                        return
                    row_features.append(label_to_append)
                all_flat_data[f'{index}_{label_index}']['features'] = row_features
                all_flat_data[f'{index}_{label_index}'][crf_label_col_name] = labels[label_index]
                all_flat_data[f'{index}_{label_index}'][global_raisha] = raisha
                all_flat_data[f'{index}_{label_index}']['round_number'] = round_number
                all_flat_data[f'{index}_{label_index}']['pair_id'] = row.pair_id
                all_flat_data[f'{index}_{label_index}']['sample_id'] = row['sample_id'] + '_' + str(round_number)

        all_flat_data_df = pd.DataFrame.from_dict(all_flat_data).T.astype({
            'features': object, crf_label_col_name: int, global_raisha: int, 'round_number': int, 'pair_id': object,
            'sample_id': object})
        self.final_data = all_flat_data_df

        file_name = f'all_data_{self.base_file_name}'
        print(f'{time.asctime(time.localtime(time.time()))}: Save all data {file_name}.pkl')
        # save_data = self.final_data.drop(['pair_id'], axis=1)
        save_data = self.final_data
        save_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(save_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'{time.asctime(time.localtime(time.time()))}: Finish create_non_nn_turn_model_features')
        logging.info(f'{time.asctime(time.localtime(time.time()))}: Finish create_non_nn_turn_model_features')

    def create_seq_data(self):
        """
        This function create 10 samples with different length from each pair data raw
        :return:
        """

        print(f'Start creating sequences with different lengths and concat')
        logging.info('Start creating sequences with different lengths and concat')

        # create the 10 samples for each pair
        meta_data_columns = [global_raisha, 'pair_id', 'sample_id']
        data_columns = ['group_sender_answer_reviews', 'previous_round_decision', 'previous_round_lottery_result_low',
                        'previous_round_lottery_result_high', 'previous_average_score_low',
                        'previous_average_score_high', 'previous_round_lottery_result_med1']

        for pair in self.pairs:
            # get a row with the subsession_round_number and one with the group_sender_answer_reviews
            if self.use_seq:  # sample use all rounds data
                data_column = dict()
                for column in data_columns:
                    data_pair = self.data.loc[self.data.pair_id == pair][['subsession_round_number', column]].transpose()
                    # define the first row as the header
                    data_pair.columns = data_pair.loc['subsession_round_number'].astype(int) - 1
                    data_pair = data_pair.drop(['subsession_round_number'])
                    for col in data_pair.columns:
                        data_column[f'{column}_{col}'] =\
                            np.concatenate([np.repeat(data_pair[col].values, self.number_of_rounds-col), np.repeat(np.nan, col)])
                data = pd.DataFrame(data_column)

                # define raisha
                data[meta_data_columns[0]] = data.index
                data[meta_data_columns[0]] = self.number_of_rounds - data[meta_data_columns[0]]
                # add sample ID column
                data[meta_data_columns[1]] = pair
                data[meta_data_columns[2]] = data[meta_data_columns[1]] + '_' + data[meta_data_columns[0]].map(str)

                # add label column
                columns = [f'exp_payoff_{i}' for i in range(self.number_of_rounds)]
                if self.total_payoff_label:
                    # Create label that will be the expert's total payoff over the 10 trails for all the pair's data
                    # (with different seq lengths)
                    data['label'] = self.data.loc[self.data.pair_id == pair]['total_exp_payoff'].unique()[0]
                else:
                    if self.label == 'single_round':
                        # the label is the exp_payoff of the current round - 1 or -1
                        data['label'] = self.data.loc[self.data.pair_id == pair]['exp_payoff'].reset_index(drop=True)
                        data.label = np.where(data.label == 1, 1, -1)
                    else:  # the label it the total payoff
                        data['label'] = (self.data.loc[self.data.pair_id == pair]['total_exp_payoff'].unique()[0] - data[
                            columns].sum(axis=1)) / (self.number_of_rounds - data[global_raisha])
                if self.label == 'next_percent':
                    data['label'] = data['label']*100

                # this is for 3 losses --> the lottery result and the expected_dm_payoff of the current round
                data['curr_result'] = \
                    self.data.loc[self.data.pair_id == pair]['group_lottery_result'].reset_index(drop=True)
                data['curr_expected_dm_payoff'] = \
                    self.data.loc[self.data.pair_id == pair]['group_average_score'].reset_index(drop=True)

                # rename columns for convenient:
                columns = [f'text_{i}' for i in range(self.number_of_rounds)]
                columns.extend([f'prev_payoff_{i}' for i in range(self.number_of_rounds)])
                columns.extend([f'prev_result_low_{i}' for i in range(self.number_of_rounds)])
                columns.extend([f'prev_result_med1_{i}' for i in range(self.number_of_rounds)])
                columns.extend([f'prev_result_high_{i}' for i in range(self.number_of_rounds)])
                columns.extend([f'prev_expected_dm_payoff_low_{i}' for i in range(self.number_of_rounds)])
                columns.extend([f'prev_expected_dm_payoff_high_{i}' for i in range(self.number_of_rounds)])
                columns.extend(['curr_result', 'curr_expected_dm_payoff'])

            else:  # single round
                columns_to_use = data_columns + ['subsession_round_number']
                data = self.data.loc[self.data.pair_id == pair][columns_to_use]
                if self.use_prev_round:  # add the previous round review, decision and lottery result
                    # if we need previous round so the first round is not relevant
                    data = data.loc[data.subsession_round_number > 1]
                    data = data.sort_values(by='subsession_round_number')
                    # get previous round data
                    data['prev_review'] = data.group_sender_answer_reviews.shift(1)
                    data['prev_lottery_result_low'] = data.lottery_result_low.shift(1)
                    data['prev_round_lottery_result_med1'] = data.lottery_result_med1.shift(1)
                    data['prev_lottery_result_high'] = data.lottery_result_high.shift(1)
                    data['prev_decision'] = data.exp_payoff.shift(1)
                    data['prev_expected_payoff'] = data.group_average_score.shift(1)

                    data.columns = ['review', 'label', 'lottery_result', 'subsession_round_number', 'prev_review',
                                    'prev_lottery_result_low', 'prev_lottery_result_low', 'prev_decision',
                                    'prev_expected_payoff', 'prev_round_lottery_result_med1']
                columns = data.columns

                # raisha
                data[meta_data_columns[0]] = data.subsession_round_number
                # add sample ID column
                data[meta_data_columns[1]] = pair
                data[meta_data_columns[2]] = data[meta_data_columns[1]] + '_' + data[meta_data_columns[0]].map(str)
                data['label'] = self.data.loc[self.data.pair_id == pair]['exp_payoff']

                # concat to all data
            self.final_data = pd.concat([self.final_data, data], axis=0, ignore_index=True)

        columns.extend(meta_data_columns)
        columns.append('label')
        self.final_data.columns = columns

        if self.label == 'single_round':  # create label as numpy array
            help_data_np = np.where(self.final_data.label == 1, 1, 0)
            help_np = np.zeros((self.final_data.label.shape[0], int(self.final_data.label.max()) + 1))
            help_np[np.arange(help_data_np.shape[0]), help_data_np] = 1
            help_np = sparse.coo_matrix(help_np)
            self.final_data['single_label'] = self.final_data.label
            self.final_data.label = help_np.toarray().tolist()

        file_name = f'all_data_{self.base_file_name}'
        self.final_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'Finish creating sequences with different lengths and concat')
        logging.info('Finish creating sequences with different lengths and concat')

        return

    def split_data(self):
        """
        Split the pairs into train-validation-test data
        :return:
        """

        print(f'Start split data to train-test-validation data and save for k=1-10 and k=1-9')
        logging.info('Start split data to train-test-validation data and save for k=1-10 and k=1-9')

        train_pairs, validation_pairs, test_pairs = np.split(self.pairs.sample(frac=1),
                                                           [int(.6 * len(self.pairs)), int(.8 * len(self.pairs))])
        for data_name, pairs in [['train', train_pairs], ['validation', validation_pairs], ['test', test_pairs]]:
            data = self.final_data.loc[self.final_data.pair_id.isin(pairs)]
            # save 10 sequences per pair

            file_name = f'{data_name}_data_1_{self.number_of_rounds}_{self.base_file_name}'
            data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
            joblib.dump(data, os.path.join(save_data_directory, f'{file_name}.pkl'))
            # save 9 sequences per pair
            if self.use_seq and not self.label == 'single_round':
                seq_len_9_data = data.loc[data.raisha != self.number_of_rounds]
                columns_to_drop = [column for column in seq_len_9_data.columns
                                   if str(self.number_of_rounds - 1) in column]
                seq_len_9_data = seq_len_9_data.drop(columns_to_drop, axis=1)
                seq_len_9_data.to_csv(os.path.join(
                    save_data_directory, f'{data_name}_data_1_{self.number_of_rounds-1}_{self.base_file_name}.csv'),
                    index=False)
                joblib.dump(seq_len_9_data,
                            os.path.join(save_data_directory,
                                         f'{data_name}_data_1_{self.number_of_rounds-1}_{self.base_file_name}.pkl'))

        print(f'Finish split data to train-test-validation data and save for k=1-10 and k=1-9')
        logging.info('Finish split data to train-test-validation data and save for k=1-10 and k=1-9')


def main():
    features_files = {
        'manual_binary_features': 'xlsx',
        'manual_features': 'xlsx',
        'bert_embedding': 'pkl',
        'manual_binary_features_minus_1': 'xlsx',
        'manual_features_minus_1': 'xlsx',
    }
    features_to_use = 'manual_binary_features'
    # label can be single_round or future_total_payoff
    conditions_dict = {
        'verbal': {'use_prev_round': False,
                   'use_prev_round_text': False,
                   'use_prev_round_label': True,
                   'use_manual_features': True,
                   'use_all_history_average': True,
                   'use_all_history': False,
                   'use_all_history_text_average': True,
                   'use_all_history_text': False,
                   'saifa_average_text': False,
                   'no_saifa_text': True,
                   'saifa_only_prev_rounds_text': False,
                   'no_text': False,
                   'use_score': False,
                   'predict_first_round': True,
                   'non_nn_turn_model': False,  # non neural networks models that predict a label for each round
                   'transformer_model': True,   # for transformer models-we need to create features also for the raisha
                   'label': 'single_round',
                   },
        'numeric': {'use_prev_round': False,
                    'use_prev_round_text': False,
                    'use_prev_round_label': True,
                    'use_manual_features': False,
                    'use_all_history_average': False,
                    'use_all_history': True,
                    'use_all_history_text_average': False,
                    'use_all_history_text': True,
                    'saifa_average_text': True,
                    'no_saifa_text': True,
                    'saifa_only_prev_rounds_text': True,
                    'no_text': True,
                    'use_score': True,
                    'predict_first_round': True,
                    'non_nn_turn_model': True,  # non neural networks models that predict a label for each round
                    'transformer_model': True,  # for transformer models-we need to create features also for the raisha
                    'label': 'future_total_payoff'
                    }
    }
    use_seq = False
    use_crf = False
    use_crf_raisha = True
    string_labels = True  # labels are string --> for LSTM model
    total_payoff_label = False if conditions_dict[condition]['label'] == 'single_round' else True
    # features_to_drop = ['topic_room_positive', 'list', 'negative_buttom_line_recommendation',
    #                     'topic_location_negative', 'topic_food_positive']
    features_to_drop = []
    only_split_data = False
    if only_split_data:
        pairs_folds = split_pairs_to_data_sets('results_payments_status')
        pairs_folds.to_csv(os.path.join(save_data_directory, 'pairs_folds.csv'))
        return

    create_save_data_obj = CreateSaveData('results_payments_status', total_payoff_label=total_payoff_label,
                                          label=conditions_dict[condition]['label'],
                                          use_seq=use_seq, use_prev_round=conditions_dict[condition]['use_prev_round'],
                                          use_manual_features=conditions_dict[condition]['use_manual_features'],
                                          features_file_type=features_files[features_to_use],
                                          features_file=features_to_use, use_crf=use_crf, use_crf_raisha=use_crf_raisha,
                                          use_all_history=conditions_dict[condition]['use_all_history'],
                                          use_all_history_average=conditions_dict[condition]['use_all_history_average'],
                                          use_all_history_text_average=conditions_dict[condition]
                                          ['use_all_history_text_average'],
                                          use_all_history_text=conditions_dict[condition]['use_all_history_text'],
                                          no_text=conditions_dict[condition]['no_text'],
                                          use_score=conditions_dict[condition]['use_score'],
                                          use_prev_round_text=conditions_dict[condition]['use_prev_round_text'],
                                          predict_first_round=conditions_dict[condition]['predict_first_round'],
                                          features_to_drop=features_to_drop, string_labels=string_labels,
                                          saifa_average_text=conditions_dict[condition]['saifa_average_text'],
                                          no_saifa_text=conditions_dict[condition]['no_saifa_text'],
                                          saifa_only_prev_rounds_text=conditions_dict[condition][
                                              'saifa_only_prev_rounds_text'],
                                          use_prev_round_label=conditions_dict[condition]['use_prev_round_label'],
                                          non_nn_turn_model=conditions_dict[condition]['non_nn_turn_model'],
                                          transformer_model=conditions_dict[condition]['transformer_model'])
    # create_save_data_obj = CreateSaveData('results_payments_status', total_payoff_label=True)
    if create_save_data_obj.use_manual_features:
        if use_seq:
            create_save_data_obj.create_manual_features_seq_data()
        elif use_crf:
            create_save_data_obj.create_manual_features_crf_data()
        elif use_crf_raisha:
            if not create_save_data_obj.use_all_history_text_average:
                create_save_data_obj.create_manual_features_crf_raisha_data(string_labels=string_labels)
            else:
                create_save_data_obj.create_manual_features_crf_raisha_data_avg_features(string_labels=string_labels)

            if create_save_data_obj.non_nn_turn_model:  # flat the crf raisha data for the non_nn_turn_model
                create_save_data_obj.create_non_nn_turn_model_features()
        else:
            create_save_data_obj.create_manual_features_data()
    else:
        if use_seq:
            create_save_data_obj.create_seq_data()
        else:
            create_save_data_obj.create_manual_features_data()

    # if use_seq or use_crf_raisha:  # for not NN models - no need train and test --> use cross validation
    #     create_save_data_obj.split_data()
    # elif use_crf:
    #     return
    # else:
        # train_test_simple_features_model(create_save_data_obj.base_file_name,
        #                                  f'all_data_{create_save_data_obj.base_file_name}.pkl', backward_search=False,
        #                                  inner_data_directory=save_data_directory,
        #                                  label=conditions_dict[condition]['label']),


if __name__ == '__main__':
    main()
