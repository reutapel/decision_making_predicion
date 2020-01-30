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


base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
data_directory = os.path.join(base_directory, 'data', condition)
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
alpha_text = 0.8
alpha_global = 0.9


def flat_reviews_numbers(data: pd.DataFrame, rounds: list, columns_to_drop: list, last_round_to_use: int,
                         first_round_to_use: int):
    """
    This function get data and flat it as for each row there is be the history of the relevant features in the data
    :param data: the data to flat
    :param rounds: the rounds to create history for
    :param columns_to_drop: the columns to drop, the first should be the column to add at last
    :param last_round_to_use: the last round to create history data for: 10 or 9
    :param first_round_to_use: the first round to use for history: the current round or the round before: 0 ot 1
    :return:
    """
    all_history = pd.DataFrame()
    for round_num in rounds:
        id_curr_round = data.loc[data.subsession_round_number == round_num][columns_to_drop[0]]
        id_curr_round.index = ['review_id']
        # this features are not relevant for the last round because they are post treatment features
        data_to_flat = data.copy()
        # the last round is not relevant to for the history of any other rounds
        data_to_flat = data_to_flat.loc[data_to_flat.subsession_round_number <= last_round_to_use]
        data_to_flat = data_to_flat.drop(columns_to_drop, axis=1)
        # for current and next rounds put -1 --> use also current if first_round_to_use=0
        # and not use use if first_round_to_use=1
        data_to_flat.iloc[list(range(round_num - first_round_to_use, last_round_to_use))] = -1
        data_to_flat.index = data_to_flat.index.astype(str)
        # data_to_flat.columns = data_to_flat.columns.astype(str)
        data_to_flat = data_to_flat.unstack().to_frame().sort_index(level=1).T
        data_to_flat.columns = [f'{str(col)}_{i}' for col, i in zip(data_to_flat.columns.get_level_values(0),
                                                                    data_to_flat.columns.get_level_values(1))]
        # concat the review_id of the current round
        data_to_flat = data_to_flat.assign(review_id=id_curr_round.values)
        all_history = pd.concat([all_history, data_to_flat], axis=0, ignore_index=True, sort=False)

    return all_history


class CreateSaveData:
    """
    This class load the data, create the seq data and save the new data with different range of K
    """
    def __init__(self, load_file_name: str, total_payoff_label: bool=True, label: str='total_payoff',
                 use_seq: bool=True, use_prev_round: bool=False, use_manual_features: bool=False, features_file: str='',
                 features_file_type: str='', use_all_history: bool = False, use_all_history_text_average: bool = False,
                 no_text: bool=False, use_score: bool=False, use_prev_round_text: bool=True,
                 use_all_history_text: bool=False, use_all_history_average: bool = False,
                 predict_first_round: bool=False):
        """
        :param load_file_name: the raw data file name
        :param total_payoff_label: if the label is the total payoff of the expert or the next rounds normalized payoff
        :param label: the name of the label
        :param use_seq: if to create a sample which is a seq or a single round
        :param use_prev_round: if to use the previous round data: review, decision, lottery result
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
                self.reviews_features = joblib.load(os.path.join(data_directory, f'{features_file}.{features_file_type}'))
            elif features_file_type == 'xlsx':
                self.reviews_features = pd.read_excel(os.path.join(data_directory, f'{features_file}.{features_file_type}'))
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

                self.reviews_features = reviews

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
        # if we use the history average -> don't predict the first round because there is no history
        if self.use_all_history_text_average or self.use_all_history_average:
            self.predict_first_round = False
        print(f'Number of pairs in data: {self.pairs.shape}')

        if self.use_all_history_average:
            self.set_all_history_average_measures()

        # create file names:
        file_name_component = [f'{self.label}_label_',
                               'seq_' if self.use_seq else '',
                               'prev_round_' if self.use_prev_round else '',
                               'prev_round_text_' if self.use_prev_round_text else '',
                               f'all_history_features_' if self.use_all_history else '',
                               f'global_alpha_{alpha_global}_' if self.use_all_history_average else '',
                               f'all_history_text_average_with_alpha_{alpha_text}_' if
                               self.use_all_history_text_average else '',
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
                                           (data_to_create.subsession_round_number == round_num), f'history_{column}'] =\
                            round(history[column].mean(), 2)
                    else:
                        data_to_create.loc[(data_to_create.pair_id == pair) &
                                           (data_to_create.subsession_round_number == round_num), f'history_{column}'] =\
                            (pow(history[column], j) * weights).sum()
        new_columns = [f'history_{column}' for column in rename_columns] + ['pair_id', 'subsession_round_number']
        self.data = self.data.merge(data_to_create[new_columns],  how='left')

        return

    def create_manual_features_data(self):
        """
        This function create 10 samples with different length from each pair data raw
        :return:
        """

        print(f'Start creating manual features data')
        logging.info('Start creating manual features data')

        # create the 10 samples for each pair
        meta_data_columns = ['k_size', 'pair_id', 'sample_id']
        already_save_column_names = False
        if self.use_prev_round:
            columns_to_use = ['review_id', 'previous_round_lottery_result_low',
                              'previous_round_lottery_result_med1', 'previous_round_lottery_result_high',
                              'previous_round_decision']
        else:
            columns_to_use = ['review_id']

        if self.use_prev_round_text:
            columns_to_use = columns_to_use + ['previous_review_id']

        if self.use_all_history_average:
            columns_to_use = columns_to_use + ['history_lottery_result', 'history_decisions',
                                               'history_lottery_result_high', 'history_chose_lose',
                                               'history_chose_earn', 'history_not_chose_lose']
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
                    ['10_result', 'group_sender_payoff', 'lottery_result_high', 'group_lottery_result',
                     'chose_lose', 'chose_earn', 'not_chose_lose', 'not_chose_earn', 'subsession_round_number']]
                temp_numbers = temp_numbers.reset_index(drop=True)
                all_history = flat_reviews_numbers(temp_numbers, rounds, columns_to_drop=['subsession_round_number'],
                                                   last_round_to_use=9, first_round_to_use=1)
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
                if self.features_file == 'bert_embedding':  # Bert features
                    data_to_flat = data[['review_id', 'subsession_round_number']].copy()
                    data_to_flat = data_to_flat.merge(self.reviews_features.T, left_on='review_id',
                                                      right_on='review_id', how='left')
                    if self.use_all_history_text:
                        history_reviews = flat_reviews_numbers(
                            data_to_flat, rounds, columns_to_drop=['review_id', 'subsession_round_number'],
                            last_round_to_use=10, first_round_to_use=0)
                        data = data.merge(history_reviews, left_on='review_id', right_on='review_id', how='left')
                    else:
                        # TODO: change to add current, previous and average text
                        reut = 1
                else:  # manual features
                    if self.use_all_history_text_average or self.use_all_history_text:
                        history_reviews = pd.DataFrame()
                        temp_reviews =\
                            self.data.loc[self.data.pair_id == pair][['review_id', 'subsession_round_number']]
                        temp_reviews = temp_reviews.merge(self.reviews_features, on='review_id', how='left')
                        if self.use_all_history_text_average:
                            for round_num in rounds:
                                review_id_curr_round =\
                                    temp_reviews.loc[temp_reviews.subsession_round_number == round_num].review_id
                                review_id_curr_round.index = ['review_id']
                                history = temp_reviews.loc[temp_reviews.subsession_round_number < round_num]
                                # history.shape[1]-2- no need subsession_round_number and review_id
                                weights = list(pow(alpha_text, round_num - history.subsession_round_number))
                                history = history.drop(['subsession_round_number', 'review_id'], axis=1)
                                if alpha_text == 0:  # if alpha=0 use average
                                    history_mean = history.mean(axis=0)  # get the average of each column
                                else:
                                    history_mean = history.mul(weights, axis=0).sum()
                                # concat the review_id of the current round
                                history_mean = history_mean.append(review_id_curr_round)
                                history_reviews = pd.concat([history_reviews, history_mean], axis=1, ignore_index=True,
                                                            sort=False)
                        else:  # self.use_all_history_text == True
                            history_reviews = flat_reviews_numbers(
                                temp_reviews, rounds, columns_to_drop=['review_id', 'subsession_round_number'],
                                last_round_to_use=10, first_round_to_use=0)
                        if self.use_all_history_text_average:
                            history_reviews = history_reviews.T
                            # add the current round reviews features
                            data = data.merge(self.reviews_features, on='review_id', how='left')
                        data = data.merge(history_reviews, on='review_id', how='left')

                data = data.drop('review_id', axis=1)
                # first merge for the review_id for the previous round
                if self.use_prev_round_text:
                    if self.features_file == 'bert_embedding':  # Bert features
                        data = data.merge(self.reviews_features.T, left_on='previous_review_id', right_on='review_id',
                                          how='left')
                    else:
                        data = data.merge(self.reviews_features, left_on='previous_review_id', right_on='review_id',
                                          how='left')
                    # drop columns not in used
                    data = data.drop('review_id', axis=1)
                    data = data.drop('previous_review_id', axis=1)

            else:  # no text
                data = data.reset_index(drop=True)

            # add metadata
            # k_size
            data[meta_data_columns[0]] = data['subsession_round_number']
            # add sample ID column
            data[meta_data_columns[1]] = pair
            data[meta_data_columns[2]] = data[meta_data_columns[1]] + '_' + data[meta_data_columns[0]].map(str)
            if self.label == 'single_round':
                # the label is the exp_payoff of the current round - 1 or -1
                if 1 not in data.subsession_round_number.values:
                    data['label'] =\
                        self.data.loc[(self.data.pair_id == pair) & (self.data.subsession_round_number > 1)][
                            'exp_payoff'].reset_index(drop=True)
                else:
                    data['label'] = self.data.loc[(self.data.pair_id == pair)]['exp_payoff'].reset_index(drop=True)
                data.label = np.where(data.label == 1, 1, -1)
                label_column_name = 'label'

            else:  # the label is the total payoff
                columns = [f'group_sender_payoff_{i}' for i in range(self.number_of_rounds - 1)]
                data['total_payoff'] = (self.data.loc[self.data.pair_id == pair]['total_exp_payoff'].unique()[0] -
                                        (data[columns] > 0).sum(axis=1)) / (self.number_of_rounds+1 - data['k_size'])
                label_column_name = 'total_payoff'
            # save column names to get the features later
            if not already_save_column_names:
                file_name = f'features_{self.base_file_name}'
                pd.DataFrame(data.columns).to_excel(os.path.join(data_directory, f'{file_name}.xlsx'), index=True)
                already_save_column_names = True
            # concat to all data
            self.final_data = pd.concat([self.final_data, data], axis=0, ignore_index=True)

        file_name = f'all_data_{self.base_file_name}'
        self.final_data = self.final_data.drop('subsession_round_number', axis=1)
        # sort columns according to the round number
        if self.use_all_history_text or self.use_all_history:
            columns_to_sort = list(self.final_data.columns)
            columns_not_to_sort = ['sample_id', 'pair_id', 'k_size', label_column_name]
            for column in columns_not_to_sort:
                columns_to_sort.remove(column)
            columns_to_sort = sorted(columns_to_sort, key=lambda x: int(x[-1]))
            columns_to_sort += columns_not_to_sort
            self.final_data = self.final_data[columns_to_sort]

        # save final data
        self.final_data.to_csv(os.path.join(data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(data_directory, f'{file_name}.pkl'))

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
        meta_data_columns = ['k_size', 'pair_id', 'sample_id']
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
            temp_features = self.reviews_features.copy().drop('review_id', axis=1)
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

            # define k_size
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
                        columns].sum(axis=1)) / (self.number_of_rounds - data['k_size'])

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
        self.final_data.to_csv(os.path.join(data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(data_directory, f'{file_name}.pkl'))

        print(f'{time.asctime(time.localtime(time.time()))}:'
              f'Finish creating sequences with different lengths and concat with manual features for the text')
        logging.info(f'{time.asctime(time.localtime(time.time()))}:'
                     f'Finish creating sequences with different lengths and concat features for the text')

        return

    def create_seq_data(self):
        """
        This function create 10 samples with different length from each pair data raw
        :return:
        """

        print(f'Start creating sequences with different lengths and concat')
        logging.info('Start creating sequences with different lengths and concat')

        # create the 10 samples for each pair
        meta_data_columns = ['k_size', 'pair_id', 'sample_id']
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

                # define k_size
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
                            columns].sum(axis=1)) / (self.number_of_rounds - data['k_size'])
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

                # k_size
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
        self.final_data.to_csv(os.path.join(data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(data_directory, f'{file_name}.pkl'))

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
            data.to_csv(os.path.join(data_directory, f'{file_name}.csv'), index=False)
            joblib.dump(data, os.path.join(data_directory, f'{file_name}.pkl'))
            # save 9 sequences per pair
            if self.use_seq and not self.label == 'single_round':
                seq_len_9_data = data.loc[data.k_size != self.number_of_rounds]
                columns_to_drop = [column for column in seq_len_9_data.columns
                                   if str(self.number_of_rounds - 1) in column]
                seq_len_9_data = seq_len_9_data.drop(columns_to_drop, axis=1)
                seq_len_9_data.to_csv(os.path.join(
                    data_directory, f'{data_name}_data_1_{self.number_of_rounds-1}_{self.base_file_name}.csv'),
                    index=False)
                joblib.dump(seq_len_9_data,
                            os.path.join(data_directory,
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
    features_to_use = 'bert_embedding'
    # label can be single_round or total_payoff
    conditions_dict = {
        'verbal': {'use_prev_round': True,
                   'use_prev_round_text': True,
                   'use_manual_features': True,
                   'use_all_history_average': True,
                   'use_all_history': False,
                   'use_all_history_text_average': True,
                   'use_all_history_text': False,
                   'no_text': False,
                   'use_score': False,
                   'predict_first_round': False,
                   'label': 'single_round'},
        'numeric': {'use_prev_round': False,
                    'use_prev_round_text': False,
                    'use_manual_features': False,
                    'use_all_history_average': False,
                    'use_all_history': True,
                    'use_all_history_text_average': False,
                    'use_all_history_text': True,
                    'no_text': True,
                    'use_score': True,
                    'predict_first_round': False,
                    'label': 'future_total_payoff'}
    }
    use_seq = False
    create_save_data_obj = CreateSaveData('results_payments_status', total_payoff_label=False,
                                          label=conditions_dict[condition]['label'],
                                          use_seq=use_seq, use_prev_round=conditions_dict[condition]['use_prev_round'],
                                          use_manual_features=conditions_dict[condition]['use_manual_features'],
                                          features_file_type=features_files[features_to_use],
                                          features_file=features_to_use,
                                          use_all_history=conditions_dict[condition]['use_all_history'],
                                          use_all_history_average=conditions_dict[condition]['use_all_history_average'],
                                          use_all_history_text_average=conditions_dict[condition]
                                          ['use_all_history_text_average'],
                                          use_all_history_text=conditions_dict[condition]['use_all_history_text'],
                                          no_text=conditions_dict[condition]['no_text'],
                                          use_score=conditions_dict[condition]['use_score'],
                                          use_prev_round_text=conditions_dict[condition]['use_prev_round_text'],
                                          predict_first_round=conditions_dict[condition]['predict_first_round'],)
    # create_save_data_obj = CreateSaveData('results_payments_status', total_payoff_label=True)
    if create_save_data_obj.use_manual_features:
        if create_save_data_obj.use_seq:
            create_save_data_obj.create_manual_features_seq_data()
        else:
            create_save_data_obj.create_manual_features_data()
    else:
        if use_seq:
            create_save_data_obj.create_seq_data()
        else:
            create_save_data_obj.create_manual_features_data()

    if use_seq:  # for not NN models - no need train and test --> use cross validation
        create_save_data_obj.split_data()
    else:
        train_test_simple_features_model(create_save_data_obj.base_file_name,
                                         f'all_data_{create_save_data_obj.base_file_name}.pkl', backward_search=False,
                                         inner_data_directory=data_directory),


if __name__ == '__main__':
    main()
