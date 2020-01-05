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
data_directory = os.path.join(base_directory, 'data')
logs_directory = os.path.join(base_directory, 'logs')

log_file_name = os.path.join(logs_directory, datetime.now().strftime('LogFile_create_save_data_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=log_file_name,
                    level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    )
random.seed(1)

# define the alpha for the weighted average of the history features - global and text features
alpha_text = 0.8
alpha_global = 0.9


class CreateSaveData:
    """
    This class load the data, create the seq data and save the new data with different range of K
    """
    def __init__(self, load_file_name: str, total_payoff_label: bool=True, label: str='total_payoff',
                 use_seq: bool=True, use_prev_round: bool=False, use_manual_features: bool=False, features_file: str='',
                 features_file_type: str='', use_all_history: bool = False, use_all_history_text: bool = False,
                 no_text: bool=False):
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
        """
        print(f'Start create and save data for file: {load_file_name}')
        logging.info('Start create and save data for file: {}'.format(load_file_name))

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
        print(f'Load features from: {features_file}')
        if features_file_type == 'pkl':
            self.reviews_features = joblib.load(os.path.join(data_directory, f'{features_file}.{features_file_type}'))
        elif features_file_type == 'xlsx':
            self.reviews_features = pd.read_excel(os.path.join(data_directory, f'{features_file}.{features_file_type}'))
        else:
            print('Features file type is has to be pkl or xlsx')
            return
        # get manual text features
        self.reviews_features = self.reviews_features.drop('review', axis=1)

        # calculate expert total payoff --> the label
        self.data['exp_payoff'] = self.data.group_receiver_choice.map({1: 0, 0: 1})
        total_exp_payoff = self.data.groupby(by='pair_id').agg(
            total_exp_payoff=pd.NamedAgg(column='exp_payoff', aggfunc=sum))
        self.data = self.data.merge(total_exp_payoff, how='left', right_index=True, left_on='pair_id')
        self.data = self.data[['pair_id', 'total_exp_payoff', 'subsession_round_number', 'group_sender_answer_reviews',
                               'exp_payoff', 'group_lottery_result', 'review_id', 'previous_round_lottery_result',
                               'previous_round_decision', 'previous_review_id', 'group_average_score',
                               'lottery_result_low', 'lottery_result_med1', 'previous_round_lottery_result_low',
                               'previous_round_lottery_result_high', 'previous_group_average_score_low',
                               'previous_group_average_score_high', 'previous_round_lottery_result_med1',
                               'time_spent_low', 'time_spent_high', 'group_sender_payoff', 'lottery_result_high',
                               'chose_lose', 'chose_earn', 'not_chose_lose', 'not_chose_earn', '10_result']]
        self.final_data = pd.DataFrame()
        self.pairs = pd.Series(self.data.pair_id.unique())
        self.total_payoff_label = total_payoff_label
        self.label = label
        self.use_seq = use_seq
        self.use_prev_round = use_prev_round
        self.use_manual_features = use_manual_features
        self.number_of_rounds = 10 if self.use_seq else 1
        self.features_file = features_file
        self.features_file_type = features_file_type
        self.use_all_history = use_all_history
        self.use_all_history_text = use_all_history_text
        self.no_text = no_text
        print(f'Number of pairs in data: {self.pairs.shape}')

        if self.use_all_history:
            self.set_all_history_measures()

        # create file names:
        file_name_component = [f'{self.label}_label_',
                               'seq_' if self.use_seq else '',
                               'prev_round_' if self.use_prev_round else '',
                               f'global_alpha_{alpha_global}_' if self.use_all_history else '',
                               f'all_history_text_alpha_{alpha_text}_' if self.use_all_history_text else '',
                               'no_text' if self.no_text else '',
                               self.features_file if self.use_manual_features and not self.no_text else '']
        self.base_file_name = ''.join(file_name_component)
        print(f'Create data for: {self.base_file_name}')
        return

    def set_all_history_measures(self):
        """
        This function calculates some measures about all the history per round for each pair
        :return:
        """

        self.data['10_result'] = np.where(self.data.group_lottery_result == 10, 1, 0)
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
                        j = 2
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
                              'previous_round_decision', 'previous_review_id']
        else:
            columns_to_use = ['review_id']

        if self.use_all_history:
            columns_to_use = columns_to_use + ['history_lottery_result', 'history_decisions',
                                               'history_lottery_result_high', 'history_chose_lose',
                                               'history_chose_earn', 'history_not_chose_lose']
        if self.no_text:
            columns_to_use.remove('review_id')
            if 'previous_review_id' in columns_to_use:
                columns_to_use.remove('previous_review_id')

        for pair in self.pairs:
            data = self.data.loc[self.data.pair_id == pair][columns_to_use]

            # first merge for the review_id for the current round
            if not self.no_text:
                if self.reviews_features.shape[1] == 2:  # Bert features
                    reviews = pd.DataFrame()
                    for i in self.reviews_features.index:
                        temp = pd.DataFrame(self.reviews_features.at[i, 'review_features']).append(
                            pd.DataFrame([self.reviews_features.at[i, 'review_id']], index=['review_id']))
                        reviews = pd.concat([reviews, temp], axis=1, ignore_index=True)
                    data = data.merge(reviews.T, left_on='review_id', right_on='review_id', how='left')
                else:  # manual features
                    if self.use_all_history_text:
                        history_reviews = pd.DataFrame()
                        temp_reviews =\
                            self.data.loc[self.data.pair_id == pair][['review_id', 'subsession_round_number']]
                        temp_reviews = temp_reviews.merge(self.reviews_features, on='review_id', how='left')
                        for round_num in range(2, 11):
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
                            history_mean = history_mean.append(review_id_curr_round)
                            history_reviews = pd.concat([history_reviews, history_mean], axis=1, ignore_index=True,
                                                        sort=False)
                        data = data.merge(history_reviews.T, on='review_id', how='left')

                    # add the current round reviews features
                    data = data.merge(self.reviews_features, on='review_id', how='left')

                data = data.drop('review_id', axis=1)
                # first merge for the review_id for the previous round
                if self.use_prev_round:
                    if self.reviews_features.shape[1] == 2:  # Bert features
                        data = data.merge(reviews.T, left_on='previous_review_id', right_on='review_id', how='left')
                    else:
                        data = data.merge(self.reviews_features, left_on='previous_review_id', right_on='review_id',
                                          how='left')
                    # drop columns not in used
                    data = data.drop('review_id', axis=1)
                    data = data.drop('previous_review_id', axis=1)
            else:
                data = data.reset_index(drop=True)
            # save column names to get the features later
            if not already_save_column_names:
                file_name = f'features_{self.base_file_name}'
                pd.DataFrame(data.columns).to_excel(os.path.join(data_directory, f'{file_name}.xlsx'), index=True)
                already_save_column_names = True

            # change column names to be range of numbers
            data.columns = list(range(len(data.columns)))

            # add metadata
            # k_size
            data[meta_data_columns[0]] =\
                self.data.loc[self.data.pair_id == pair]['subsession_round_number'].reset_index(drop=True)
            # add sample ID column
            data[meta_data_columns[1]] = pair
            data[meta_data_columns[2]] = data[meta_data_columns[1]] + '_' + data[meta_data_columns[0]].map(str)
            data['label'] = self.data.loc[self.data.pair_id == pair]['exp_payoff'].reset_index(drop=True)
            data['label'] = np.where(data['label'] == 1, 1, -1)
            # don't use first round if use_prev_round or all_history
            if self.use_prev_round or self.use_all_history:
                data = data.loc[data.k_size > 1]
            # concat to all data
            self.final_data = pd.concat([self.final_data, data], axis=0, ignore_index=True)

        file_name = f'all_data_{self.base_file_name}'
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
        data_columns = ['review_features', 'previous_round_decision', 'previous_round_lottery_result_low',
                        'previous_round_lottery_result_high', 'previous_group_average_score_low',
                        'previous_round_lottery_result_med1', 'previous_group_average_score_high',
                        'history_lottery_result', 'history_decisions', 'history_lottery_result_high',
                        'history_chose_lose', 'history_chose_earn', 'history_not_chose_lose', 'time_spent_low',
                        'time_spent_high']

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
            for column in data_columns:
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
                        'prev_expected_dm_payoff_low', 'prev_expected_dm_payoff_high',
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
                        'previous_round_lottery_result_high', 'previous_group_average_score_low',
                        'previous_group_average_score_high', 'previous_round_lottery_result_med1']

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
    }
    features_to_use = 'bert_embedding'
    use_seq = False
    create_save_data_obj = CreateSaveData('results_payments_status', total_payoff_label=False, label='single_round',
                                          use_seq=use_seq, use_prev_round=True, use_manual_features=True,
                                          features_file_type=features_files[features_to_use],
                                          features_file=features_to_use, use_all_history=False,
                                          use_all_history_text=True, no_text=False)
    # create_save_data_obj = CreateSaveData('results_payments_status', total_payoff_label=True)
    if create_save_data_obj.use_manual_features:
        if create_save_data_obj.use_seq:
            create_save_data_obj.create_manual_features_seq_data()
        else:
            create_save_data_obj.create_manual_features_data()
    else:
        create_save_data_obj.create_seq_data()

    if use_seq:  # for not NN models - no need train and test --> use cross validation
        create_save_data_obj.split_data()
    else:
        train_test_simple_features_model(create_save_data_obj.base_file_name,
                                         f'all_data_{create_save_data_obj.base_file_name}.pkl', backward_search=False),


if __name__ == '__main__':
    main()
