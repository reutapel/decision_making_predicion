import pandas as pd
import os
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
from typing import *
import logging
from matplotlib.font_manager import FontProperties
import math
from data_analysis import create_chart_bars, create_statistics, create_point_plot, create_histogram, create_bar_from_df
from collections import defaultdict


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'results')
orig_data_analysis_directory = os.path.join(base_directory, 'analysis')
date_directory = 'text_exp_2_tests'
condition_directory = 'numeric'
log_file_name = os.path.join(orig_data_analysis_directory, date_directory,
                             datetime.now().strftime('LogFile_data_analysis_%d_%m_%Y_%H_%M_%S.log'))

split_gender = False
split_condition = False
cost = 8

data_analysis_directory = os.path.join(orig_data_analysis_directory, date_directory, condition_directory)
if split_gender:
    data_analysis_directory = os.path.join(data_analysis_directory, 'per_gender')
if split_condition:
    data_analysis_directory = os.path.join(data_analysis_directory, 'per_condition')
if not (os.path.exists(data_analysis_directory)):
    os.makedirs(data_analysis_directory)

logging.basicConfig(filename=log_file_name,
                    level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    )

like_zero_one_list = ['zero_one_accurate_all', 'zero_one_accurate_prob', 'zero_one_accurate_ev',
                      'zero_one_not_accurate', 'zero_one_accurate_ev_prob', 'zero_one_accurate_ev_avg',
                      'zero_one_accurate_avg']

batch_size = 5
number_of_rounds = 10
alpha = 1

score_eval_data = pd.read_excel('/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/'
                                'data_analysis/results/text_exp_2_tests/score evaluation task.xlsx',
                                sheet_name='data_to_plot')


def linear_score(row: pd.Series, condition_column: str='subsession_round_number'):
    """Create LinPay =-9Pay1 -7Pay2 -5Pay3 -3Pay4 -1Pay5 +1Pay6 + 3Pay7 +5Pay8 +7Pay9 +9Pay10"""
    if row[condition_column] == 1:
        return -9
    elif row[condition_column] == 2:
        return -7
    elif row[condition_column] == 3:
        return -5
    elif row[condition_column] == 4:
        return -3
    elif row[condition_column] == 5:
        return -1
    elif row[condition_column] == 6:
        return 9
    elif row[condition_column] == 7:
        return 7
    elif row[condition_column] == 8:
        return 5
    elif row[condition_column] == 9:
        return 3
    elif row[condition_column] == 10:
        return 1
    else:
        raise Exception(f'{condition_column} not in 1-10')


def compare_positive_negative_first(data: pd.DataFrame):
    """
    This function compare the % DM entered per review_id between the same review when it first has the negative or
    positive part
    :param data: pd.DataFrame: must have: review_id and pct participant columns
    :return:
    """
    data['prefix_review_id'] = data.review_id.map(lambda x: x[:-1])
    data = data.assign(positive_first_better_all='')
    data = data.assign(positive_first_better_1='')
    data = data.assign(positive_first_better_2='')

    for column in [['round_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]', 'all'], ['round_[1]', 1], ['round_[2]', 2]]:
        for i in data.prefix_review_id:
            negative_first = data.loc[data.review_id == str(i) + str(0)][column[0]].values
            positive_first = data.loc[data.review_id == str(i) + str(1)][column[0]].values
            if negative_first < positive_first:  # positive_first_better
                data.loc[data.prefix_review_id == i, f'positive_first_better_{column[1]}'] = 1
            elif negative_first > positive_first:  # negative_first_better
                data.loc[data.prefix_review_id == i, f'positive_first_better_{column[1]}'] = 0
            else:  # one of the prefix does not exist for this column
                data.loc[data.prefix_review_id == i, f'positive_first_better_{column[1]}'] = None

    return data


class DataAnalysis:
    def __init__(self, class_gender='all genders', class_condition='all_condition'):
        self.gender = class_gender
        self.condition = class_condition if split_condition else condition_directory
        print(f'Start running data analysis on data folder {date_directory} {condition_directory}')
        logging.info('Start running data analysis on data folder {}, {}'.format(date_directory, condition_directory))
        columns_to_use = ['participant_code', 'participant__current_page_name', 'participant_visited',
                          'participanst_mturk_worker_id', 'participant_mturk_assignment_id', 'participant_payoff',
                          'player_id_in_group', 'player_name', 'player_age', 'player_gender', 'player_is_student',
                          'player_occupation', 'player_residence', 'player_payoff', 'group_id_in_subsession',
                          'group_sender_answer', 'group_receiver_choice', 'group_lottery_result',
                          'group_sender_timeout',
                          'group_receiver_timeout', 'group_sender_payoff', 'group_x_lottery', 'group_y_lottery',
                          'group_p_lottery', 'subsession_round_number', 'session_code']
        self.results = pd.read_excel(os.path.join(data_directory, date_directory, condition_directory,
                                                  'text_exp_data.xlsx'), sheet_name='data_to_use')
        # , usecols=columns_to_use)
        self.results.columns = self.results.columns.str.replace(r".", "_")

        if split_gender:
            gender_participants = self.results.loc[self.results['player_gender'] == self.gender].participant_code
            self.results = self.results.loc[self.results.participant_code.isin(gender_participants)]
        if split_condition:
            condition_participants =\
                self.results.loc[self.results['subsession_condition'] == self.condition].participant_code
            self.results = self.results.loc[self.results.participant_code.isin(condition_participants)]

        # keep only data to use
        # self.results = self.results.loc[self.results.participant_visited == 1]
        self.time_spent = pd.read_csv(os.path.join(data_directory, date_directory, condition_directory,
                                                   'TimeSpent.csv'))
        self.payments = pd.read_excel(os.path.join(data_directory, date_directory, condition_directory,
                                                   'all_Approved_assignments.xlsx'))
        self.hotel_reviews = pd.read_csv(os.path.join(data_directory, date_directory, '10_reviews.csv'))
        # self.all_reviews = pd.read_csv(os.path.join(data_directory, date_directory, 'all_reviews.csv'))
        self.results_payments =\
            self.results.merge(self.payments, left_on=['participant_mturk_worker_id', 'participant_mturk_assignment_id',
                                                       'participant_code'],
                               right_on=['worker_id', 'assignment_id', 'participant_code'], how='left')
        self.results_payments = self.results_payments.assign(status='')
        self.results_payments = self.results_payments.assign(prob_status='')
        self.results_payments = self.results_payments.assign(player_timeout=0)
        self.results_payments = self.results_payments.assign(player_answer=0)
        self.results_payments = self.results_payments.assign(max_num_rounds_timeout=0)
        self.results_payments = self.results_payments.assign(all_review_len=0)
        self.results_payments = self.results_payments.assign(positive_review_len=0)
        self.results_payments = self.results_payments.assign(negative_review_len=0)
        self.results_payments = self.results_payments.assign(positive_negative_review_len_prop=0)
        self.results_payments['group_average_score'] = self.results_payments['group_average_score'].round(2)

        self.results_payments['group_median_score'] = self.results_payments['group_score_3']

        # adding pair_id
        if 'pair_id' not in self.results_payments.columns:
            self.results_payments['pair_id'] = self.results_payments['session_code'] + '_' +\
                                               self.results_payments['group_id_in_subsession'].map(str)
        self.personal_info = self.results.loc[self.results.player_age.notnull(),
                                              ['participant_code', 'participant_mturk_worker_id', 'player_residence',
                                               'participant_mturk_assignment_id', 'player_name', 'player_age',
                                               'player_gender', 'player_is_student', 'player_occupation']]
        participants_insert_personal_info = self.personal_info.participant_code.unique()
        # only the players that started to play
        self.results_payments_play = self.results_payments.loc[
            (self.results_payments['participant__current_page_name'] != 'AfterInstructions') &
            (self.results_payments.participant_code.isin(participants_insert_personal_info))].copy()
        self.participants_started = self.results_payments.loc[~self.results_payments.player_intro_test.isnull()].\
            participant_code.unique()

    def set_expert_answer_scores_reviews_len_verbal_cond(self):
        """
        This function set the expert answer score in the verbal condition
        :return:
        """
        if self.condition == 'verbal':
            # for each row set the group_sender_answer_scores using group_sender_answer_index
            # for index, row in self.results_payments.iterrows():
            #     group_sender_answer_index = row['group_sender_answer_index']
            #     self.results_payments.at[index, 'group_sender_answer_scores'] =\
            #         row[f'group_score_{group_sender_answer_index-1}']

            # set the reviews length
            self.results_payments['all_review_len'] = self.results_payments['group_sender_answer_reviews'].str.len()
            self.results_payments['positive_review_len'] =\
                self.results_payments['group_sender_answer_positive_reviews'].str.len()
            self.results_payments['negative_review_len'] =\
                self.results_payments['group_sender_answer_negative_reviews'].str.len()
            self.results_payments['positive_negative_review_len_prop'] =\
                round(self.results_payments['positive_review_len']/self.results_payments['negative_review_len'], 2)

            return

    def average_most_close_hotel_id(self):
        """
        This function define the hotel ID and the review that most close to the score average
        :return:
        """

        # define hotel ID
        hotels_scores_list = [f'score_{i}' for i in range(0, 7)]
        hotels_scores_list.append('hotel_id')
        hotels_scores_id = self.hotel_reviews[hotels_scores_list]
        score_columns = [f'group_score_{i}' for i in range(0, 7)]
        columns = ['participant_code', 'subsession_round_number'] + score_columns
        participants_rounds = self.results_payments[columns]
        hotels_scores_id['all_scores'] = hotels_scores_id[f'score_0'].map(str) + '_' +\
            hotels_scores_id[f'score_1'].map(str) + '_' + hotels_scores_id[f'score_2'].map(str) + '_' +\
            hotels_scores_id[f'score_3'].map(str) + '_' + hotels_scores_id[f'score_4'].map(str) + '_' + \
            hotels_scores_id[f'score_5'].map(str) + '_' + hotels_scores_id[f'score_6'].map(str)

        participants_rounds['all_scores'] = participants_rounds['group_score_0'].map(str) + '_' +\
            participants_rounds['group_score_1'].map(str) + '_' + participants_rounds['group_score_2'].map(str) + '_' +\
            participants_rounds['group_score_3'].map(str) + '_' + participants_rounds['group_score_4'].map(str) + '_' + \
            participants_rounds['group_score_5'].map(str) + '_' + participants_rounds['group_score_6'].map(str)
        participants_rounds = participants_rounds.merge(hotels_scores_id, on='all_scores')

        # define review with score most close to average
        hotels_scores_list.append('average_score')
        hotels_scores_id = self.hotel_reviews[hotels_scores_list]
        for i in range(0, 7):
            hotels_scores_id[f'diff_{i}'] = abs(hotels_scores_id[f'score_{i}'] - hotels_scores_id['average_score'])

        hotles_score_diff_list = [f'diff_{i}' for i in range(0, 7)]
        hotels_scores_id['min_diff_score'] = hotels_scores_id[hotles_score_diff_list].min(axis=1)
        hotels_scores_id['avg_most_close_review_index'] = np.empty((len(hotels_scores_id), 0)).tolist()
        for index, row in hotels_scores_id.iterrows():
            for i in range(0, 7):
                if row[f'diff_{i}'] == row['min_diff_score']:
                    hotels_scores_id.at[index, 'avg_most_close_review_index'] += [i]

        participants_rounds = participants_rounds.merge(hotels_scores_id[['hotel_id', 'avg_most_close_review_index']],
                                                        on='hotel_id')
        self.results_payments = self.results_payments.merge(
            participants_rounds[['participant_code', 'subsession_round_number', 'hotel_id',
                                 'avg_most_close_review_index']], on=['participant_code', 'subsession_round_number'],
            how='left')

        # create review ids
        reviews_ids = self.results_payments.loc[(~self.results_payments.hotel_id.isnull()) &
                                                (~self.results_payments.group_sender_answer_index.isnull())][
            ['hotel_id', 'group_sender_answer_index', 'group_sender_answer_reviews']]
        reviews_ids.hotel_id = pd.to_numeric(reviews_ids.hotel_id).astype(int)
        reviews_ids.group_sender_answer_index = pd.to_numeric(reviews_ids.group_sender_answer_index).astype(int)
        reviews_ids['positive_first'] = np.where(reviews_ids['group_sender_answer_reviews'].str[0] == 'P', 1, 0)
        reviews_ids['review_id'] = reviews_ids.hotel_id.map(str) + reviews_ids.group_sender_answer_index.map(str) +\
                                   reviews_ids.positive_first.map(str)
        # reviews_ids.review_id = reviews_ids.review_id.astype(int)
        self.results_payments = self.results_payments.merge(reviews_ids.review_id, left_index=True, right_index=True,
                                                            how='left')

        # set the review seen in the previous round
        self.results_payments = self.results_payments.sort_values(by=['pair_id', 'subsession_round_number'])
        self.results_payments['previous_review_id'] = self.results_payments.review_id.shift(2)
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_review_id'] = np.nan
        self.results_payments['previous_score'] = self.results_payments.group_sender_answer_scores.shift(2)
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_score'] = np.nan
        # add a column that check if the expert chose the score which is most close to the average
        # avg_most_close_review_index is 0-6 and group_sender_answer_index is 1-7
        self.results_payments['chose_most_average_close'] = self.results_payments.apply(
            lambda avg_row: np.nan if type(avg_row['avg_most_close_review_index']) != list else
            (1 if avg_row['group_sender_answer_index']-1 in avg_row['avg_most_close_review_index'] else 0), axis=1)
        # add a column that check if the expert chose the median score
        self.results_payments['chose_median_score'] = np.where(
            self.results_payments['group_sender_answer_scores'] == self.results_payments['group_median_score'], 1, 0)
        # check the bias from the most close to average is the expert didn't chose it
        self.results_payments.assign(bias_from_most_close_avg=0)
        for index, bias_row in self.results_payments.iterrows():
            if bias_row['chose_most_average_close'] in [0]:
                most_close_index = bias_row['avg_most_close_review_index'][0]
                most_close_score = bias_row[f'group_score_{most_close_index}']
                self.results_payments.loc[index, 'bias_from_most_close_avg'] =\
                    bias_row['group_sender_answer_scores'] - most_close_score
        # add column that check if the expert didn't choose the closer to average or the median
        self.results_payments['not_chose_closer_average_or_median'] =\
            np.where((self.results_payments.chose_median_score == 0) &
                     (self.results_payments.chose_most_average_close == 0), 1, 0)

        self.results_payments.to_csv(os.path.join(data_analysis_directory, 'results_payments_status.csv'))
        hotels_numerical_values = self.results_payments.groupby(by='hotel_id').agg(
            {'not_chose_closer_average_or_median': 'sum', 'chose_most_average_close': 'sum',
             'chose_median_score': 'sum', 'group_sender_answer_scores': 'mean', 'group_average_score': 'mean',
             'group_median_score': 'mean'})
        hotels_numerical_values.group_sender_answer_scores = hotels_numerical_values.group_sender_answer_scores.round(2)
        hotels_numerical_values.columns = ['number of experts did not choose median or average',
                                           'number of experts chose the score closest to the average',
                                           'number of experts chose the median score', 'experts average advice',
                                           'hotel average score', 'hotel median score']
        hotels_numerical_values['advice bias from average'] = hotels_numerical_values['experts average advice'] -\
                                                              hotels_numerical_values['hotel average score']
        hotels_numerical_values['advice bias from median'] = hotels_numerical_values['experts average advice'] -\
                                                              hotels_numerical_values['hotel median score']
        hotels_numerical_values.to_csv(os.path.join(data_analysis_directory, 'hotels_numerical_values.csv'))

        return

    def set_player_partner_timeout_answer(self):
        """
        This function arrange the player data. It set the player timeout and answer
        :return:
        """
        # if no p_lottery - consider as timeout
        self.results_payments['player_timeout'] = np.where(self.results_payments.player_id_in_group == 1,
                                                           self.results_payments.group_sender_timeout,
                                                           self.results_payments.group_receiver_timeout)
        self.results_payments['partner_timeout'] = np.where(self.results_payments.player_id_in_group == 2,
                                                            self.results_payments.group_sender_timeout,
                                                            self.results_payments.group_receiver_timeout)
        self.results_payments['player_timeout'] = np.where(self.results_payments.group_sender_answer_index.isnull(), 1,
                                                           self.results_payments.player_timeout)
        self.results_payments['partner_timeout'] = np.where(self.results_payments.group_sender_answer_index.isnull(), 1,
                                                            self.results_payments.partner_timeout)
        self.results_payments['player_answer'] = np.where(self.results_payments.player_id_in_group == 1,
                                                          self.results_payments.group_sender_answer_index,
                                                          self.results_payments.group_receiver_choice)
        self.results_payments['partner_answer'] = np.where(self.results_payments.player_id_in_group == 2,
                                                           self.results_payments.group_sender_answer_index,
                                                           self.results_payments.group_receiver_choice)

    def set_previous_round_measures(self):
        """
        This function create the previous round relevant measures
        :return:
        """
        # set the lottery result from the previous round
        # shift(2) because there are 2 rows for each pair in each subsession_round_number': expert and DM
        self.results_payments = self.results_payments.sort_values(by=['pair_id', 'subsession_round_number'])
        self.results_payments['previous_round_lottery_result'] = self.results_payments.group_lottery_result.shift(2)
        self.results_payments['previous_round_lottery_result_low'] =\
            np.where(self.results_payments.previous_round_lottery_result < 3, 1, 0)
        self.results_payments['previous_round_lottery_result_med1'] = \
            np.where(self.results_payments.previous_round_lottery_result.between(3, 5), 1, 0)
        self.results_payments['previous_round_lottery_result_high'] =\
            np.where(self.results_payments.previous_round_lottery_result >= 8, 1, 0)
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_round_lottery_result'] = np.nan
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_round_lottery_result_high'] = np.nan
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_round_lottery_result_low'] = np.nan

        # set the decision from the previous round
        self.results_payments['previous_round_decision'] = self.results_payments.group_sender_payoff.shift(2)
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_round_decision'] = np.nan

        # set the average score from the previous round
        self.results_payments['previous_average_score'] = self.results_payments.group_average_score.shift(2)
        self.results_payments['previous_average_score_low'] = \
            np.where(self.results_payments.previous_average_score < 3, 1, 0)
        self.results_payments['previous_average_score_med1'] = \
            np.where(self.results_payments.previous_average_score.between(3, 5), 1, 0)
        self.results_payments['previous_average_score_high'] = \
            np.where(self.results_payments.previous_average_score >= 8, 1, 0)
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_average_score_low'] = np.nan
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_average_score_high'] = np.nan
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_average_score'] = np.nan

        # set the score from the previous round
        self.results_payments['previous_score'] = self.results_payments.group_sender_answer_scores.shift(2)
        self.results_payments['previous_score_low'] = np.where(self.results_payments.previous_score < 3, 1, 0)
        self.results_payments['previous_score_med1'] = np.where(self.results_payments.previous_score.between(3, 5), 1, 0)
        self.results_payments['previous_score_high'] = np.where(self.results_payments.previous_score >= 8, 1, 0)
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_score_low'] = np.nan
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_score_high'] = np.nan
        self.results_payments.loc[
            self.results_payments.subsession_round_number == 1, 'previous_score_score'] = np.nan

        # create binary column for the decision and the lottery_result_high
        # group_receiver_choice = 1 if DM chose Home, group_sender_payoff = 1 if DM chose Hotel
        self.results_payments['chose_lose'] = \
            self.results_payments.lottery_result_lose * self.results_payments.group_sender_payoff
        self.results_payments['previous_chose_lose'] = self.results_payments.chose_lose.shift(2)

        self.results_payments['chose_earn'] = \
            self.results_payments.lottery_result_high * self.results_payments.group_sender_payoff
        self.results_payments['previous_chose_earn'] = self.results_payments.chose_earn.shift(2)

        self.results_payments['not_chose_lose'] = \
            self.results_payments.lottery_result_lose * self.results_payments.group_receiver_choice
        self.results_payments['previous_not_chose_lose'] = self.results_payments.chose_lose.shift(2)

        self.results_payments['not_chose_earn'] = \
            self.results_payments.lottery_result_high * self.results_payments.group_receiver_choice
        self.results_payments['previous_not_chose_earn'] = self.results_payments.not_chose_earn.shift(2)

    def set_current_round_measures(self):
        """
        This function create some measures for each round
        :return:
        """
        # create bins for the group_lottery_result and group_average_score
        self.results_payments['lottery_result_low'] = np.where(self.results_payments.group_lottery_result < 3, 1, 0)
        self.results_payments['lottery_result_med1'] =\
            np.where(self.results_payments.group_lottery_result.between(3, 5), 1, 0)
        self.results_payments['lottery_result_high'] = np.where(self.results_payments.group_lottery_result >= 8, 1, 0)
        self.results_payments['lottery_result_lose'] = np.where(self.results_payments.group_lottery_result < 8, 1, 0)
        self.results_payments['average_score_low'] = np.where(self.results_payments.group_average_score < 5, 1, 0)
        self.results_payments['average_score_high'] = np.where(self.results_payments.group_average_score >= 8, 1, 0)
        self.results_payments['dm_expected_payoff'] = np.where(self.results_payments.group_sender_payoff == 1,
                                                               self.results_payments.group_average_score-8, 0)

        return

    def set_all_history_measures(self):
        """
        This function calculates some measures about all the history per round for each pair
        :return:
        """

        self.results_payments['10_result'] = np.where(self.results_payments.group_lottery_result == 10, 1, 0)
        columns_to_calc = ['group_lottery_result', 'group_sender_payoff', 'lottery_result_high', 'chose_lose',
                           'chose_earn', 'not_chose_lose', 'not_chose_earn', '10_result']
        rename_columns = ['lottery_result', 'decisions', 'lottery_result_high', 'chose_lose', 'chose_earn',
                          'not_chose_lose', 'not_chose_earn', '10_result']
        # Create only for the experts and then assign to both players
        columns_to_chose = columns_to_calc + ['pair_id', 'subsession_round_number']
        data_to_create = self.results_payments.loc[(self.results_payments.status == 'play') &
                                                   (self.results_payments.player_id_in_group == 1)][columns_to_chose]
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
                weights = pow(alpha, round_num - history.subsession_round_number)
                for column in rename_columns:
                    if column == 'lottery_result':
                        j = 2
                    else:
                        j = 1
                    data_to_create.loc[(data_to_create.pair_id == pair) &
                                       (data_to_create.subsession_round_number == round_num), f'history_{column}'] =\
                        (pow(history[column], j) * weights).sum()  # round(history[column].mean(), 2)
        new_columns = [f'history_{column}' for column in rename_columns] + ['pair_id', 'subsession_round_number']
        self.results_payments = self.results_payments.merge(data_to_create[new_columns],  how='left')

        return

    def time_spent_analysis(self):
        """
        This function analyze the time spent in SenderPage, ReceiverPage, Introduction and PersonalInformation
        :return:
        """

        # arrange time spent
        if 'group_receiver_passed_test' in self.results_payments.columns:
            participant_role = self.results_payments[['participant_code', 'player_id_in_group', 'player_timeout',
                                                      'group_receiver_passed_test']]
        else:
            participant_role = self.results_payments[['participant_code', 'player_id_in_group', 'player_timeout']]
        time_spent_to_merge = pd.DataFrame(columns=['participant_code', 'subsession_round_number', 'seconds_on_page'])
        for role in [[[1], 'SenderPage'], [[2], 'ReceiverPage'], [[1, 2], 'Introduction'],
                     [[1, 2], 'PersonalInformation']]:
            # get data for role and not timeout, take only the role[1] page
            participant_curr_role = self.time_spent.loc[self.time_spent.page_name == role[1]]
            role_data = participant_role.loc[(participant_role.player_id_in_group.isin(role[0])) &
                                             (participant_role.player_timeout == 0)]
            role_data = role_data.drop_duplicates('participant_code')
            role_data = role_data.merge(participant_curr_role, how='left', on='participant_code')
            # create round number
            role_data = role_data.assign(
                subsession_round_number=role_data.sort_values(['participant_code', 'page_index'], ascending=True).
                                            groupby(['participant_code']).cumcount() + 1)

            if len(role[0]) == 1:  # SenderPage or ReceiverPage
                # for ReceiverPage split histogram for passed on not passed test
                if 'group_receiver_passed_test' in role_data.columns and role[1] == 'ReceiverPage':
                    time_spent_to_merge = time_spent_to_merge.append(
                        role_data[['participant_code', 'subsession_round_number', 'seconds_on_page',
                                   'group_receiver_passed_test']])
                    for passed in [0, 1, 0.5]:
                        passed_role_data = role_data.loc[(role_data.seconds_on_page <= 150) &
                                                         (role_data.group_receiver_passed_test == passed)]
                        if not passed_role_data.empty:
                            x = np.array(passed_role_data['seconds_on_page'])
                            create_histogram(title=f'time_spent_in_page_condition_{self.condition}_{role[1]}_passed_'
                                                   f'test_{passed}_all_rounds', x=x, xlabel='seconds_in_page',
                                             ylabel='total_number_of_rounds', add_labels=True,
                                             curr_date_directory=data_analysis_directory, step=10)

                else:
                    time_spent_to_merge = time_spent_to_merge.append(
                        role_data[['participant_code', 'subsession_round_number', 'seconds_on_page']])
                role_data = role_data.loc[role_data.seconds_on_page <= 150]
                x = np.array(role_data['seconds_on_page'])
                create_histogram(title=f'time_spent_in_page_condition_{self.condition}_{role[1]}_all_rounds', x=x,
                                 xlabel='seconds_in_page', ylabel='total_number_of_rounds', add_labels=True,
                                 curr_date_directory=data_analysis_directory, step=10)
                for round_num in range(1, 11):
                    round_data = role_data.loc[role_data.subsession_round_number == round_num]
                    x = np.array(round_data['seconds_on_page'])
                    create_histogram(title=f'time_spent_in_page_condition_{self.condition}_{role[1]}_round_{round_num}',
                                     x=x, xlabel='seconds_in_page', ylabel='total_number_of_participants',
                                     add_labels=True, curr_date_directory=data_analysis_directory, step=10)

            average_time_spent = role_data.seconds_on_page.mean()
            median_time_spent = role_data.seconds_on_page.median()
            min_time_spent = role_data.seconds_on_page.min()
            max_time_spent = role_data.seconds_on_page.max()

            print(time.asctime(time.localtime(time.time())), ': condition:', self.condition,
                  'average time (in seconds) spent on page', role[1], 'is:', average_time_spent,
                  ', median time spent is:', median_time_spent, 'max time:', max_time_spent, ', min_time:',
                  min_time_spent)
            logging.info('{}: condition: {} average time (in seconds) spent on page: {} is {}, '
                         'median time spent is: {}, max time is: {}, min time is: {}'.
                         format(time.asctime(time.localtime(time.time())), self.condition, role[1], average_time_spent,
                                median_time_spent, max_time_spent, min_time_spent))
            if role[1] == 'ReceiverPage':
                for receiver_passed_test in [0, 1, 0.5]:
                    role_data_second_test = role_data.loc[role_data.group_receiver_passed_test == receiver_passed_test]
                    average_time_spent = role_data_second_test.seconds_on_page.mean()
                    median_time_spent = role_data_second_test.seconds_on_page.median()
                    min_time_spent = role_data_second_test.seconds_on_page.min()
                    max_time_spent = role_data_second_test.seconds_on_page.max()
                    print(time.asctime(time.localtime(time.time())), ': condition:', self.condition,
                          'average time (in seconds) spent on page', role[1], 'and receiver_passed_test=',
                          receiver_passed_test, 'is:', average_time_spent, ', median time spent is:', median_time_spent,
                          'max time:', max_time_spent, ', min_time:',
                          min_time_spent)
                    logging.info('{}: condition: {} average time (in seconds) spent on page: {} and '
                                 'receiver_passed_test={} is {}, median time spent is: {}, max time is: {}, '
                                 'min time is: {}'.
                                 format(time.asctime(time.localtime(time.time())), self.condition, role[1],
                                        receiver_passed_test, average_time_spent, median_time_spent, max_time_spent,
                                        min_time_spent))

        self.results_payments =\
            self.results_payments.merge(time_spent_to_merge[['participant_code', 'seconds_on_page',
                                                             'subsession_round_number']],
                                        on=['participant_code', 'subsession_round_number'], how='left')
        self.results_payments['time_spent_low'] = np.where(self.results_payments.seconds_on_page < 15, 1, 0)
        self.results_payments['time_spent_med'] = np.where(self.results_payments.seconds_on_page.between(15, 29), 1, 0)
        self.results_payments['time_spent_high'] = np.where(self.results_payments.seconds_on_page >= 30, 1, 0)

        return

    def how_much_pay(self):
        """
        This function calculate how much we pay
        :return:
        """
        # workers that we pay them for participate - more than $2.5
        participants_paid = self.payments.loc[self.payments.total_pay >= 2.5]
        participants_paid = participants_paid.drop_duplicates('participant_code')
        print(time.asctime(time.localtime(time.time())), ': number of participants we pay:',
              participants_paid.shape[0])
        logging.info('{}: number of participants we pay: {}'.format(time.asctime(time.localtime(time.time())),
                                                                    participants_paid.shape[0]))
        average_payment = participants_paid.total_pay.mean()
        total_payment = participants_paid.total_pay.sum()
        print(time.asctime(time.localtime(time.time())), ': average payment for participants:', average_payment)
        logging.info('{}: average payment for participants: {}'.format(time.asctime(time.localtime(time.time())),
                                                                       average_payment))
        print(time.asctime(time.localtime(time.time())), ': total payment for participants:', total_payment)
        logging.info('{}: total payment for participants: {}'.format(time.asctime(time.localtime(time.time())),
                                                                     total_payment))

        # workers that we pay them for waiting
        waiters_paid = self.payments.loc[self.payments.total_pay < 2.5]
        print(time.asctime(time.localtime(time.time())), ': number of waiters we pay:', waiters_paid.shape[0])
        logging.info('{}: number of waiters we pay: {}'.format(time.asctime(time.localtime(time.time())),
                                                               waiters_paid.shape[0]))

        average_wait = waiters_paid.total_pay.mean()
        print(time.asctime(time.localtime(time.time())), ': average payment for waiters:', average_wait)
        logging.info('{}: average payment for waiters: {}'.format(time.asctime(time.localtime(time.time())),
                                                                  average_wait))

        total_wait = waiters_paid.total_pay.sum()
        print(time.asctime(time.localtime(time.time())), ': total payment for waiters:', total_wait)
        logging.info('{}: total payment for waiters: {}'.format(time.asctime(time.localtime(time.time())),
                                                                  total_wait))

        return

    def define_player_status(self):
        """
        This function assign for each participant its status: wait, play, left, partner_left, both_left,
        partner_left_before_start, left_before_start, wait_but_not_got_paid, unknown
        :return:
        """
        # wait: if we pay less than $1 but more than $0
        didnt_see_intro = self.results_payments.loc[
            (~self.results_payments.participant_code.isin(
                self.time_spent.loc[self.time_spent.page_name == 'Introduction'].participant_code.unique())) &
            (~self.results_payments.participant_code.isin(self.results_payments.loc[self.results_payments.
                                                          participant__current_page_name == 'Introduction'].
                                                          participant_code.unique()))].participant_code.unique()
        participants_wait =\
            self.results_payments.loc[
                (self.results_payments.total_pay < 1) & (self.results_payments.total_pay > 0) &
                (self.results_payments.player_intro_test.isnull()) &
                (self.results_payments.subsession_round_number == 1) &
                (self.results_payments.participant_code.isin(didnt_see_intro)) &
                (~self.results_payments.participant_mturk_worker_id.isnull())].participant_code.unique()
        self.results_payments.loc[self.results_payments.participant_code.isin(participants_wait), 'status'] = 'wait'
        participants_drop_timeout =\
            self.results_payments.loc[
                (~self.results_payments.participant_code.isin(didnt_see_intro)) &
                (self.results_payments.player_intro_test.isnull()) &
                (self.results_payments.status != 'wait') & (self.results_payments.subsession_round_number == 1) &
                (~self.results_payments.participant_mturk_worker_id.isnull())].participant_code.unique()
        self.results_payments.loc[self.results_payments.participant_code.isin(participants_drop_timeout), 'status'] =\
            'drop_timeout'
        # answered correctly on first test
        participants_answered_correct_first_test =\
            self.results_payments.loc[(self.results_payments.player_intro_test.str.lower() == 'sdkot') &
                                      (self.results_payments.subsession_round_number == 1)].participant_code.unique()
        participants_not_answered_correct_first_test =\
            self.results_payments.loc[(self.results_payments.player_intro_test.str.lower() != 'sdkot') &
                                      (~self.results_payments.player_intro_test.isnull()) &
                                      (self.results_payments.subsession_round_number == 1)].participant_code.unique()
        self.results_payments.loc[self.results_payments.participant_code.isin(participants_answered_correct_first_test),
                                  'first_test_status'] = 1
        self.results_payments.loc[self.results_payments.participant_code.isin(
            participants_not_answered_correct_first_test), 'first_test_status'] = 0
        # second test
        self.results_payments.loc[(self.results_payments.group_receiver_passed_test == 0) &
                                  (self.results_payments.player_id_in_group == 2) &
                                  (self.results_payments.first_test_status == 1), 'second_test_status'] = \
            'dm_failed_second_no_pay'
        self.results_payments.loc[(self.results_payments.group_receiver_passed_test == 0.5) &
                                  (self.results_payments.player_id_in_group == 2) &
                                  (self.results_payments.first_test_status == 1), 'second_test_status'] = \
            'dm_half_passed_second_no_bonus'
        self.results_payments.loc[(self.results_payments.group_receiver_passed_test == 1) &
                                  (self.results_payments.player_id_in_group == 2) &
                                  (self.results_payments.first_test_status == 1), 'second_test_status'] = \
            'dm_passed_second'
        self.results_payments.loc[(self.results_payments.participant__current_page_name == 'Test') &
                                  (self.results_payments.player_id_in_group == 2) &
                                  (self.results_payments.first_test_status == 1), 'second_test_status'] =\
            'dm_not_took_second_test'
        # status based on first test
        participants_pass_first_test_partner_not =\
            self.results_payments.loc[(self.results_payments.first_test_status == 1) &
                                      (self.results_payments.player_name.isnull()) &
                                      (self.results_payments.subsession_round_number == 1)].participant_code.unique()
        self.results_payments.loc[
            (self.results_payments.participant_code.isin(participants_pass_first_test_partner_not)) &
            (self.results_payments.participant__current_page_name != 'AfterIntroTest'), 'status'] =\
            'pass_first_test_partner_not'
        self.results_payments.loc[
            (self.results_payments.participant_code.isin(participants_pass_first_test_partner_not)) &
            (self.results_payments.participant__current_page_name == 'AfterIntroTest'), 'status'] =\
            'pass_first_test_partner_drop'
        self.results_payments.loc[(self.results_payments.first_test_status == 0), 'status'] = 'failed_first_test'
        # self.results_payments.loc[
        #     (self.results_payments.first_test_status == 1) &
        #     (~self.results_payments.participant_code.isin(participants_pass_first_test_partner_not)) &
        #     (self.results_payments.player_id_in_group == 1), 'status'] = 'expert_pass_first_test'

        # partner_left_before_start: they inserted personal info and didn't play - probably the partner didn't start
        partner_left_before_start_after_test =\
            self.results_payments.loc[(self.results_payments.first_test_status == 1) &
                                      (~self.results_payments.player_name.isnull()) &
                                      (self.results_payments.status.isin(['', 'expert_pass_first_test'])) &
                                      (self.results_payments.subsession_round_number == 1) &
                                      (self.results_payments.group_lottery_result.isnull())][
                ['participant_code', 'pair_id']]
        left_before_start_after_test = \
            self.results_payments.loc[(self.results_payments.first_test_status == 1) &
                                      (self.results_payments.player_name.isnull()) &
                                      (self.results_payments.subsession_round_number == 1) &
                                      (self.results_payments.pair_id.isin(
                                          partner_left_before_start_after_test.pair_id)) &
                                      (self.results_payments.group_lottery_result.isnull())].participant_code.unique()
        partner_left_before_start_after_test = partner_left_before_start_after_test.participant_code.unique()
        self.results_payments.loc[self.results_payments.participant_code.isin(partner_left_before_start_after_test),
                                  'status'] = 'partner_left_before_start_after_test'
        self.results_payments.loc[self.results_payments.participant_code.isin(left_before_start_after_test),
                                  'status'] = 'left_before_start_after_test'
        # left_before_start: didn't start to play, didn't insert personal info and have not got paid
        self.results_payments.loc[(~self.results_payments['participant_code'].isin(self.participants_started)) &
                                  (self.results_payments['participant__current_page_name'] != 'AfterInstructions') &
                                  (self.results_payments.total_pay.isnull()) &
                                  (self.results_payments.participant_payoff == 0) &
                                  (self.results_payments.status != 'wait') &
                                  (~self.results_payments.first_test_status.isin([0, 1])) &
                                  (~self.results_payments.participant_code.isin(partner_left_before_start_after_test)) &
                                  (~self.results_payments.participant_code.isin(left_before_start_after_test)) &
                                  (~self.results_payments.player_intro_test.isnull()), 'status'] = 'left_before_start'
        # unknown: didn't start to play, didn't insert personal info and have not got paid, but payoff higher than 0
        self.results_payments.loc[(~self.results_payments['participant_code'].isin(self.participants_started)) &
                                  (self.results_payments['participant__current_page_name'] != 'AfterInstructions') &
                                  (self.results_payments.total_pay.isnull()) &
                                  (self.results_payments.participant_payoff > 0), 'status'] = 'unknown'
        # wait_but_not_got_paid: didn't start to play, didn't insert personal info and have not got paid,
        # but payoff is negative
        self.results_payments.loc[(~self.results_payments['participant_code'].isin(self.participants_started)) &
                                  (self.results_payments['participant__current_page_name'] != 'AfterInstructions') &
                                  (self.results_payments.total_pay.isnull()) &
                                  (self.results_payments.participant_payoff < -2.5) &
                                  (self.results_payments.first_test_status != 0), 'status'] = 'wait_but_not_got_paid'
        # left: played and left in the middle. They started, but not got money
        left_participants =\
            self.results_payments.loc[(self.results_payments['participant_code'].isin(self.participants_started)) &
                                      (self.results_payments.total_pay.isnull()) &
                                      (self.results_payments.first_test_status == 1) &
                                      (self.results_payments.group_sender_payoff.isnull()) &
                                      (self.results_payments.status.isin(['', 'expert_pass_first_test'])) &
                                      (~self.results_payments.participant_code.isin(
                                          partner_left_before_start_after_test)) &
                                      (~self.results_payments.participant_code.isin(left_before_start_after_test)) &
                                      (self.results_payments.second_test_status.isnull())].participant_code.unique()
        self.results_payments.loc[self.results_payments.participant_code.isin(left_participants), 'status'] = 'left'

        all_left_participants = np.concatenate((left_participants, partner_left_before_start_after_test,
                                                left_before_start_after_test))
        # if the player left, but in the last 2 rounds he played - no timeout --> so he played
        # played_not_paid = self.results_payments.loc[(self.results_payments.status == 'left') &
        #                                             (self.results_payments.subsession_round_number.isin(
        #                                                 [number_of_rounds-1, number_of_rounds])) &
        #                                             (self.results_payments.player_timeout == 0) &
        #                                             (self.results_payments.group_sender_answer_index.notnull()) &
        #                                             (self.results_payments.first_test_status == 1)].\
        #     participant_code.unique()
        # self.results_payments.loc[self.results_payments.participant_code.isin(played_not_paid), 'status'] = 'play'
        # get the participants that left info
        # pair_id_left = self.results_payments.loc[self.results_payments.status == 'left']['pair_id'].unique()
        # left_participant_code =\
        #     self.results_payments.loc[self.results_payments.status == 'left'].participant_code.unique()
        # number_of_left_in_pair = self.results_payments.loc[self.results_payments.status == 'left'].\
        #     groupby(by='pair_id').participant_code.count()
        # both_left_pair_id = number_of_left_in_pair.loc[number_of_left_in_pair == number_of_rounds*2].index
        # one_left_pair_id = number_of_left_in_pair.loc[number_of_left_in_pair == number_of_rounds].index
        # partner_left: the pair_id is in the list of left participants and the participant_code is not in the list
        # self.results_payments.loc[(~self.results_payments.participant_code.isin(left_participant_code)) &
        #                           (self.results_payments.pair_id.isin(one_left_pair_id) &
        #                            (self.results_payments['participant_code'].isin(self.participants_started))) &
        #                           (self.results_payments.status == ''), 'status'] = 'partner_left'
        # both_left: the pair_id and the participant_code are in the list of left participants
        # self.results_payments.loc[(self.results_payments.participant_code.isin(left_participant_code)) &
        #                           (self.results_payments.pair_id.isin(both_left_pair_id) &
        #                            (self.results_payments['participant_code'].isin(self.participants_started))),
        #                           'status'] = 'both_left'

        # DM didn't pass second test
        pair_id_left = self.results_payments.loc[
            self.results_payments.participant_code.isin(all_left_participants)].pair_id.unique()
        dm_failed_second_pair_id = self.results_payments.loc[self.results_payments.second_test_status ==
                                                             'dm_failed_second_no_pay'].pair_id.unique()
        dm_didnt_take_second_pair_id = self.results_payments.loc[self.results_payments.second_test_status ==
                                                                 'dm_not_took_second_test'].pair_id.unique()
        self.results_payments.loc[(self.results_payments.pair_id.isin(dm_failed_second_pair_id)) &
                                  (self.results_payments.status == ''), 'status'] = 'dm_not_passed_second_test'
        self.results_payments.loc[(self.results_payments.pair_id.isin(dm_didnt_take_second_pair_id)) &
                                  (self.results_payments.status == ''), 'status'] = 'dm_not_took_second_test'

        # play: the pair_id and the participant_code are not in the list of left participants, and they got paid
        # and DM passed second test, and they passed the first test
        self.results_payments.loc[(~self.results_payments.participant_code.isin(all_left_participants)) &
                                  (~self.results_payments.pair_id.isin(pair_id_left) &
                                   (self.results_payments.participant_code.isin(self.participants_started))) &
                                  (self.results_payments.status == '') & (self.results_payments.first_test_status == 1),
                                  'status'] = 'play'
        number_of_rows_per_pair = self.results_payments.groupby(by='pair_id').participant_code.count()
        pairs_one_player_data = list(number_of_rows_per_pair.loc[number_of_rows_per_pair == number_of_rounds].index)
        self.results_payments.loc[(self.results_payments.pair_id.isin(pairs_one_player_data) &
                                   (self.results_payments['participant_code'].isin(self.participants_started))) &
                                  (self.results_payments.status == ''), 'status'] = 'partner_left'
        # play_but_not_paid: the pair_id and the participant_code are not in the list of left participants,
        # but they didn't get money
        # self.results_payments.loc[(~self.results_payments.participant_code.isin(left_participant_code)) &
        #                           (~self.results_payments.pair_id.isin(pair_id_left) &
        #                            (self.results_payments.total_pay.isnull()) &
        #                            (self.results_payments['participant_code'].isin(self.participants_started))),
        #                           'status'] = 'play_but_not_paid'

        # unknown - I didn't assign anything
        self.results_payments.loc[(self.results_payments.status == '') &
                                  (self.results_payments.participant__current_page_name == 'GroupedWaitPage'),
                                  'status'] = 'left_while_wait'
        self.results_payments.loc[(self.results_payments.status == '') &
                                  (self.results_payments.participant_mturk_worker_id.isnull()), 'status'] =\
            'no_turker_assigned'

        self.results_payments.loc[self.results_payments.status == '', 'status'] = 'unknown'

        return

    def print_pairs(self):
        """
        This function calculate the number of pairs of each type : both finished, one left, both left
        :return:
        """
        # second test status
        dm_failed_second_no_pay = self.results_payments.loc[
            (self.results_payments.second_test_status == 'dm_failed_second_no_pay')].participant_code.unique()
        dm_half_passed_second_no_bonus = self.results_payments.loc[
            (self.results_payments.second_test_status == 'dm_half_passed_second_no_bonus') &
            (self.results_payments.status == 'play')].participant_code.unique()
        dm_passed_second = self.results_payments.loc[(self.results_payments.second_test_status == 'dm_passed_second') &
                                                     (self.results_payments.status == 'play')].participant_code.unique()
        expert_pass_first_test_played =\
            self.results_payments.loc[(self.results_payments.first_test_status == 1) &
                                      (self.results_payments.player_id_in_group == 1)].participant_code.unique()
        participants_partner_left_before_start_after_test =\
            self.results_payments.loc[self.results_payments.status == 'partner_left_before_start_after_test'].\
                participant_code.unique()
        participants_left_before_start_after_test = \
            self.results_payments.loc[self.results_payments.status == 'left_before_start_after_test'].\
                participant_code.unique()

        both_finished = self.results_payments.loc[self.results_payments.status == 'play'].pair_id.unique()
        one_left = self.results_payments.loc[self.results_payments.status == 'left'].pair_id.unique()
        both_left = self.results_payments.loc[self.results_payments.status == 'both_left'].pair_id.unique()
        pass_first_test_partner_not = self.results_payments.loc[
            self.results_payments.status == 'pass_first_test_partner_not'].participant_code.unique()
        failed_first_test =\
            self.results_payments.loc[self.results_payments.status == 'failed_first_test'].participant_code.unique()

        print(time.asctime(time.localtime(time.time())), ': number of participants started:',
              self.participants_started.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of pairs finished:', both_finished.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of pairs one left:', one_left.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of pairs both left:', both_left.shape[0])
        print(time.asctime(time.localtime(time.time())),
              ': number of participants partner left before start after test:',
              participants_partner_left_before_start_after_test.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of participants left before start after test:',
              participants_left_before_start_after_test.shape[0])
        print(time.asctime(time.localtime(time.time())),
              ': number of participants passed first test and their partner did not:',
              pass_first_test_partner_not.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of participants failed the first test:',
              failed_first_test.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of dm failed second test - not pay:',
              dm_failed_second_no_pay.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of dm half passed second test - not bonus:',
              dm_half_passed_second_no_bonus.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of dm passed second test:',
              dm_passed_second.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of expert passed first test:',
              expert_pass_first_test_played.shape[0])

        logging.info('{}: number of participants started: {}'.format(time.asctime(time.localtime(time.time())),
                                                                     self.participants_started.shape[0]))
        logging.info('{}: number of pairs finished: {}'.format(time.asctime(time.localtime(time.time())),
                                                               both_finished.shape[0]))
        logging.info('{}: number of pairs one left: {}'.format(time.asctime(time.localtime(time.time())),
                                                               one_left.shape[0]))
        logging.info('{}: number of pairs both left: {}'.format(time.asctime(time.localtime(time.time())),
                                                                both_left.shape[0]))
        logging.info('{}: number of participants passed first test and their partner did not: {}'.format(
            time.asctime(time.localtime(time.time())), pass_first_test_partner_not.shape[0]))
        logging.info('{}: number of participants failed the first test: {}'.format(
            time.asctime(time.localtime(time.time())), failed_first_test.shape[0]))
        logging.info('{}: number of expert passed first test: {}'.format(
            time.asctime(time.localtime(time.time())), expert_pass_first_test_played.shape[0]))
        logging.info('{}: number of participants partner left before start after test: {}'.format(
            time.asctime(time.localtime(time.time())), participants_partner_left_before_start_after_test.shape[0]))
        logging.info('{}: number of participants left before start after test: {}'.format(
            time.asctime(time.localtime(time.time())), participants_left_before_start_after_test.shape[0]))

        partner_left_before_start = self.results_payments.loc[
            self.results_payments.status == 'partner_left_before_start'].participant_code.unique()
        left_before_start = self.results_payments.loc[
            self.results_payments.status == 'left_before_start'].participant_code.unique()

        print(time.asctime(time.localtime(time.time())), ': number of partner_left_before_start:',
              partner_left_before_start.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of left_before_start:', left_before_start.shape[0])
        logging.info('{}: number of partner_left_before_start: {}'.format(time.asctime(time.localtime(time.time())),
                                                                          partner_left_before_start.shape[0]))
        logging.info('{}: number of left_before_start: {}'.format(time.asctime(time.localtime(time.time())),
                                                                  left_before_start.shape[0]))

        # left participants per role
        expert_left = self.results_payments.loc[
            (self.results_payments.status.isin(['left', 'both_left'])) &
            (self.results_payments.player_id_in_group == 1)].participant_code.unique()
        decision_maker_left = self.results_payments.loc[
            (self.results_payments.status.isin(['left', 'both_left'])) &
            (self.results_payments.player_id_in_group == 2)].participant_code.unique()
        print(time.asctime(time.localtime(time.time())), ': number of experts that left:', expert_left.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of decision makers that left:',
              decision_maker_left.shape[0])
        logging.info('{}: umber of experts that left: {}'.format(time.asctime(time.localtime(time.time())),
                                                                 expert_left.shape[0]))
        logging.info('{}: umber of decision makers that left: {}'.format(time.asctime(time.localtime(time.time())),
                                                                         decision_maker_left.shape[0]))

        return

    def sender_answer_analysis(self):
        """
        This function analyze some measures: probability difference, always answer 0/1 or 3 values
        :return:
        """
        data_to_analyze = self.results_payments.loc[self.results_payments.status.isin(['play', 'partner_left'])]
        # take only senders and not timeouts
        data_to_analyze = data_to_analyze.loc[(data_to_analyze.player_id_in_group == 1) &
                                              (data_to_analyze.player_timeout == 0)]
        # keep only one row from each round in each pair
        data_to_analyze = data_to_analyze.drop_duplicates(subset=['subsession_round_number', 'participant_code'])

        # check how many always answer 0/1
        zero_one = data_to_analyze.groupby(by='participant_code').player_answer
        zero_one = zero_one.unique()
        zero_one_len = zero_one.str.len()

        # played with one value only
        only_one = zero_one_len.loc[zero_one_len == 1].index
        only_one = zero_one.loc[zero_one.index.isin(only_one)]
        self.results_payments.loc[self.results_payments.participant_code.isin(only_one.index), 'prob_status'] =\
            'one_value'
        play_one = data_to_analyze.loc[(data_to_analyze.prob_status == 'one_value') &
                                       (data_to_analyze.status == 'play')]
        print(time.asctime(time.localtime(time.time())), ': number of pairs play but with one value of prob:',
              play_one.pair_id.unique().shape[0])
        logging.info('{}: number of pairs play but less than 3 values of prob: {}'.
                     format(time.asctime(time.localtime(time.time())), play_one.pair_id.unique().shape[0]))

        # played with 2 or 3 values
        short = zero_one_len.loc[zero_one_len.isin([2, 3])].index
        short = zero_one.loc[zero_one.index.isin(short)]

        # define 2-3 values prob status
        for participant_code, values in short.items():
            # define expert that always give high estimation: 5-7
            if set(values).issubset(set([5, 6, 7])):
                self.results_payments.loc[self.results_payments.participant_code == participant_code, 'prob_status'] = \
                    '3_high_values'
                data_to_analyze.loc[
                    data_to_analyze.participant_code == participant_code, 'prob_status'] = '3_high_values'

            # define expert that always give low estimation: 1-3
            elif set(values).issubset(set([1, 2, 3])):
                self.results_payments.loc[
                    self.results_payments.participant_code == participant_code, 'prob_status'] = \
                    '3_low_values'
                data_to_analyze.loc[
                    data_to_analyze.participant_code == participant_code, 'prob_status'] = '3_low_values'
            # define expert that always give median estimation: 2-4
            elif set(values).issubset(set([2, 3, 4])):
                self.results_payments.loc[
                    self.results_payments.participant_code == participant_code, 'prob_status'] = \
                    '3_median_values'
                data_to_analyze.loc[
                    data_to_analyze.participant_code == participant_code, 'prob_status'] = '3_median_values'

            # define expert that always give 1/7
            elif 1 in values and 7 in values and len(values) == 2:
                self.results_payments.loc[
                    self.results_payments.participant_code == participant_code, 'prob_status'] = \
                    '2_extreme_values'
                data_to_analyze.loc[
                    data_to_analyze.participant_code == participant_code, 'prob_status'] = '2_extreme_values'

            else:
                self.results_payments.loc[
                    self.results_payments.participant_code == participant_code, 'prob_status'] = \
                    '2_3_values'
                data_to_analyze.loc[
                    data_to_analyze.participant_code == participant_code, 'prob_status'] = '2_3_values'

        # define use_average status- if the expert use the average score as its estimation
        data_to_analyze = data_to_analyze.assign(use_average=0)
        data_to_analyze['use_average'] = data_to_analyze.apply(
            lambda x: int(x.player_answer in x.avg_most_close_review_index), axis=1)
        experts_sum = data_to_analyze.groupby(by='participant_code').agg({'use_average': 'sum',
                                                                          'subsession_round_number': 'count'})
        experts_sum = experts_sum.loc[experts_sum.subsession_round_number >= 9]
        experts_sum['pct_use_average'] = experts_sum.use_average / experts_sum.subsession_round_number
        experts_sum['use_average_prob_status'] = np.where(experts_sum.pct_use_average >= 0.6, 1, 0)
        use_average_participants = experts_sum.loc[experts_sum.use_average_prob_status == 1].index.tolist()
        self.results_payments.loc[self.results_payments.participant_code.isin(use_average_participants), 'prob_status']\
            += 'use_average'
        data_to_analyze.loc[data_to_analyze.participant_code.isin(use_average_participants), 'prob_status'] +=\
            'use_average'

        for prob_status in ['2_extreme_values', '3_median_values', '3_low_values', '3_high_values', '2_3_values',
                            'use_average']:
            play_short = data_to_analyze.loc[(data_to_analyze.prob_status == prob_status) &
                                             (data_to_analyze.status == 'play')]
            print(f'{time.asctime(time.localtime(time.time()))} : number of pairs play with prob status {prob_status}:'
                  f'{play_short.pair_id.unique().shape[0]}')
            logging.info(f'{time.asctime(time.localtime(time.time()))} : number of pairs play with prob status '
                         f'{prob_status}: {play_short.pair_id.unique().shape[0]}')
            partner_left_short = data_to_analyze.loc[(data_to_analyze.prob_status == prob_status) &
                                                     (data_to_analyze.status == 'partner_left')]
            print(f'{time.asctime(time.localtime(time.time()))} : number of pairs one left with prob status '
                  f'{prob_status}: {partner_left_short.pair_id.unique().shape[0]}')
            logging.info(f'{time.asctime(time.localtime(time.time()))} : number of pairs one left with prob status '
                         f'{prob_status}: {partner_left_short.pair_id.unique().shape[0]}')

        return

    def effectiveness_measure(self):
        """
        This function measure the effectiveness of each expert relative to the number of trials with positive EV.
        :return:
        """

        # create average EV column - take not partner timeout rows
        data_for_average = self.results_payments.loc[(self.results_payments.status == 'play') &
                                                     (self.results_payments.player_id_in_group == 1)]

        if data_for_average.empty:
            if split_gender:
                print(f'no data in experts_to_define for gender {self.gender} in function effectiveness_measure')
                return
            if split_condition:
                print(f'no data in experts_to_define for condition {self.condition} in function effectiveness_measure')
                return

        data_for_average['num_trials_positive_avg'] =\
            data_for_average.apply(lambda row: 1 if row['group_average_score']-cost >= 0 else 0, axis=1)
        # count the number of trials with positive EV
        group_data_for_average = pd.DataFrame(data_for_average.groupby(by='participant_code').
                                              num_trials_positive_avg.sum())
        # group_data_for_average['participant_code'] = group_data_for_average.index
        group_data_for_average = group_data_for_average.reset_index()

        # option 1: compute the score relative to the mean number of trial with positive EV
        average_num_positive_ev = group_data_for_average.num_trials_positive_avg.mean()
        data_to_merge = data_for_average[['total_payoff_no_partner_timeout', 'participant_code']].drop_duplicates()
        group_data_for_average = group_data_for_average.merge(data_to_merge, how='left', on='participant_code')

        group_data_for_average['score_average_num_positive'] = group_data_for_average.total_payoff_no_partner_timeout -\
                                                               average_num_positive_ev
        group_data_for_average['score_num_positive'] = group_data_for_average.total_payoff_no_partner_timeout -\
                                                       group_data_for_average.num_trials_positive_avg

        group_data_for_average = group_data_for_average[['num_trials_positive_avg', 'participant_code',
                                                         'score_average_num_positive', 'score_num_positive']]
        self.results_payments = self.results_payments.merge(group_data_for_average, how='left', on='participant_code')

        return

    def round_analysis(self):
        """
        This function analysis the rounds: when participants retired and the number of timeouts
        :return:
        """
        # round of retired
        data_of_left = self.results_payments.loc[self.results_payments.status.isin(['left', 'both_left'])]

        if data_of_left.empty:
            print(f'no data in experts_to_define for gender {self.gender} and condition {self.condition} in function'
                  f'round_analysis because no data of left')
            self.results_payments = self.results_payments.assign(left_round=11)
        else:
            # analyse the participants that left
            data_of_left = data_of_left[['participant_code', 'subsession_round_number', 'player_timeout']]
            pivot = data_of_left.pivot_table(index='participant_code', columns='subsession_round_number')
            pivot.columns = np.arange(1, number_of_rounds+1)
            pivot = pivot.assign(left_round='')
            for i in range(number_of_rounds, 0, -1):
                pivot['sum_' + str(i)] = pivot.iloc[:, i:number_of_rounds].sum(axis=1)
                pivot['sum_' + str(i)] = number_of_rounds - pivot['sum_' + str(i)]
                pivot.loc[(pivot['sum_' + str(i)] == i) & (pivot[i] == 1), 'left_round'] = i

            pivot_to_merge = pivot[['left_round']]
            pivot_to_merge.loc[pivot_to_merge.left_round == '', 'left_round'] = 10
            create_histogram(title='trial_of_left', x=np.array(pivot_to_merge), xlabel='trial_number',
                             ylabel='number_of_participants', add_labels=True, curr_date_directory=data_analysis_directory)
            self.results_payments = self.results_payments.merge(pivot_to_merge, how='left', left_on='participant_code',
                                                                right_index=True)

        # get the max number of timeouts that is not left
        max_num_rounds_timeout = 0
        max_num_rounds_timeout_participant_code = ''
        # get all players that play
        data_max_timeout = self.results_payments.loc[self.results_payments.status.isin(['left', 'both_left', 'play',
                                                                                        'partner_left'])]

        if data_max_timeout.empty:
            print(f'no data in experts_to_define for gender {self.gender} and condition {self.condition} in function'
                  f'round_analysis because no data_max_timeout')
        else:
            data_max_timeout = data_max_timeout[['participant_code', 'subsession_round_number', 'player_timeout']]
            pivot = data_max_timeout.pivot_table(index='participant_code', columns='subsession_round_number')
            pivot.columns = np.arange(1, number_of_rounds+1)
            pivot = pivot.transpose()
            for participant_code in pivot.columns:
                timeout_lst = pivot[[participant_code]]
                timeout_lst = timeout_lst.loc[(timeout_lst[participant_code] == 0) | (timeout_lst.index == 1)]
                timeout_lst = timeout_lst.reset_index()
                timeout_lst['diff'] = timeout_lst['index'].diff()
                max_index = timeout_lst['diff'].idxmax()
                # if the max diff doesn't start in the first round - need to reduce by 1
                if max_index > 1:
                    timeout_lst['diff'] -= 1
                    # if the first round is timeout need to add 1 for the first diff
                    if timeout_lst.iloc[0][participant_code] == 1:
                        timeout_lst.loc[1, 'diff'] += 1
                max_num_round = timeout_lst['diff'].max()
                if timeout_lst.shape[0] == 1:  # always timeout
                    max_num_round = number_of_rounds
                self.results_payments.loc[self.results_payments.participant_code == participant_code,
                                          'max_num_rounds_timeout'] = max_num_round
                if (max_num_round > max_num_rounds_timeout) and (max_num_round < number_of_rounds):  # set the max number of rounds
                    max_num_rounds_timeout = max_num_round
                    max_num_rounds_timeout_participant_code = participant_code

            print(time.asctime(time.localtime(time.time())), ': max number of timeouts:', max_num_rounds_timeout,
                  'by participant:', max_num_rounds_timeout_participant_code)
            logging.info('{}: max number of timeouts: {} by participant: {}'.
                         format(time.asctime(time.localtime(time.time())), max_num_rounds_timeout,
                                max_num_rounds_timeout_participant_code))

        # create bar for the max number of rounds of timeout
        data_to_visual = self.results_payments.loc[
            self.results_payments.status.isin(['left', 'both_left', 'play', 'partner_left'])]

        if data_to_visual.empty:
            print(f'no data in experts_to_define for gender {self.gender} and condition {self.condition} in function'
                  f'round_analysis because no data_to_visual')
        else:
            data_to_visual = data_to_visual.drop_duplicates(subset=['participant_code'])
            data_to_visual = data_to_visual.loc[data_to_visual.left_round != 1]
            data_to_visual = data_to_visual['max_num_rounds_timeout']
            create_histogram(title=f'max_number_of_timeout_trials_before_left for gender {self.gender} and condition '
                                   f'{self.condition}', x=np.array(data_to_visual), xlabel='number_of_timeout_trials',
                             ylabel='number_of_participants', add_labels=True,
                             curr_date_directory=data_analysis_directory)

            # timeouts
            for status in [[['play', 'partner_left'], 'finish'], [['left', 'both_left', 'play', 'partner_left'], 'all']]:
                for role in [[[1], 'expert'], [[2], 'decision_maker'], [[1, 2], 'all']]:
                    data_to_use = self.results_payments.loc[self.results_payments.status.isin(status[0])]

                    if data_to_use.empty:
                        print(f'no data in experts_to_define for condition {self.condition} and {self.gender}'
                              f'in function round_analysis')

                    role_participants = data_to_use.loc[data_to_use.player_id_in_group.isin(role[0])]
                    role_timeout = role_participants.groupby(by='participant_code')['player_timeout'].sum()
                    role_timeout = (role_timeout / number_of_rounds) * 100
                    x = np.array(role_timeout.values)
                    create_histogram(title=role[1] + '_timeout_' + status[1], x=x, xlabel='pct_timeout',
                                     ylabel='number_of_participants', add_labels=True,
                                     curr_date_directory=data_analysis_directory)

            role_timeout = pd.DataFrame(role_timeout)
            role_timeout = role_timeout.reset_index()
            role_timeout.columns = ['participant_code', 'pct_timeout']
            self.results_payments = self.results_payments.merge(role_timeout, on='participant_code', how='left')
            self.results_payments.loc[(self.results_payments.pct_timeout >= 80) &
                                      (self.results_payments.status == 'play'), 'status'] = 'more_than_80_pct_timeout'
            play_always_timeout = self.results_payments.loc[self.results_payments.status == 'more_than_80_pct_timeout']
            participants_always_timeout = play_always_timeout.participant_code.unique()
            print(time.asctime(time.localtime(time.time())),
                  ': number of participants that played and always had timeout:', participants_always_timeout.shape[0])
            logging.info('{}: number of participants that played and always had timeout: {}'.
                         format(time.asctime(time.localtime(time.time())), participants_always_timeout.shape[0]))
            pair_ids_always_timeout = play_always_timeout.pair_id.unique()
            self.results_payments.loc[(~self.results_payments.participant_code.isin(participants_always_timeout)) &
                                      (self.results_payments.pair_id.isin(pair_ids_always_timeout)), 'status'] =\
                'partner_always_timeout'

        return

    def bonus_analysis(self):
        """
        This function analyze the bonus participants got
        :return:
        """
        # number of partner_left workers that got bonus - how much we could save
        partner_left = self.results_payments.loc[self.results_payments.status == 'partner_left']
        partner_left = partner_left.drop_duplicates('participant_code')
        partner_left_bonus = partner_left.loc[partner_left.bonus > 2.5]
        print(time.asctime(time.localtime(time.time())), ': number of participants that their partner has left'
                                                         ' and got bonus:', partner_left_bonus.shape[0])
        logging.info('{}: number of participants that their partner has left and got bonus: {}'.
                     format(time.asctime(time.localtime(time.time())), partner_left_bonus.shape[0]))

        # bonus per role
        for role in [[1, 'expert'], [2, 'decision_maker']]:
            # all participants with role[0] that started to play
            role_started = self.results_payments.loc[(self.results_payments.player_id_in_group == role[0]) &
                                                     (self.results_payments.status.isin(['play', 'partner_left']))]

            if role_started.empty:
                print(f'no data in experts_to_define for gender {self.gender} and condition {self.condition}'
                      f' in function bonus_analysis')

            number_of_started = role_started.participant_code.unique()
            # participants with role[0] that got bonus
            role_data_bonus = role_started.loc[role_started.bonus > 2.5]
            # participants with role[0] that played but didn't got bonus
            role_data_no_bonus = role_started.loc[(role_started.bonus >= 0) & (role_started.bonus <= 2.5)]
            participants_got_bonus = role_data_bonus.participant_code.unique()
            participants_no_bonus = role_data_no_bonus.participant_code.unique()

            print(time.asctime(time.localtime(time.time())), ': number of participants with role', role[1],
                  'that have finished the game (had potential to get bonus): ', number_of_started.shape[0])
            print(time.asctime(time.localtime(time.time())), ': number of participants with role', role[1],
                  'that got bonus:', participants_got_bonus.shape[0])
            print(time.asctime(time.localtime(time.time())), ': number of participants with role', role[1],
                  'that have not got bonus:', participants_no_bonus.shape[0])

            logging.info('{}: number of participants with role {} that have finished the game'
                         ' (had potential to get bonus): '.format(time.asctime(time.localtime(time.time())), role[1],
                                                                  number_of_started.shape[0]))
            logging.info('{}: number of participants with role {} that got bonus: {}'.
                         format(time.asctime(time.localtime(time.time())), role[1], participants_got_bonus.shape[0]))
            logging.info('{}: number of participants with role {} that got bonus: {}'.
                         format(time.asctime(time.localtime(time.time())), role[1], participants_no_bonus.shape[0]))

        return

    def define_total_payoff_no_partner_timeout(self):
        """
        Define the column of total_payoff_no_partner_timeout
        :return:
        """

        type_list = ['total_payoff', 'total_payoff_no_partner_timeout']

        # get the total payoff of each play participant
        play_participants = self.results_payments.loc[self.results_payments.status == 'play']

        if play_participants.empty:
            if split_gender:
                print(f'no data in experts_to_define for gender {self.gender} in function '
                      f'define_total_payoff_no_partner_timeout')
                return
            if split_condition:
                print(f'no data in experts_to_define for condition {self.condition} in function '
                      f'define_total_payoff_no_partner_timeout')
                return

        for total_payoff_type in type_list:
            if total_payoff_type == 'total_payoff_no_partner_timeout':
                play_participants = play_participants.loc[play_participants.partner_timeout == 0]
            total_payoff = pd.DataFrame(play_participants.groupby(by='participant_code').player_payoff.sum())
            total_payoff = total_payoff.reset_index()
            total_payoff.columns = ['participant_code', total_payoff_type]
            self.results_payments = self.results_payments.merge(total_payoff, how='left', on='participant_code')

        return

    def expert_payoff_analysis(self):
        """
        This function analyze the expert payoff (the sum of payoffs in all rounds)
        :return:
        """

        group_list = self.results_payments.prob_status.unique().tolist()
        groups_payoff_statistics_dict = dict()
        type_list = ['total_payoff', 'total_payoff_no_partner_timeout', 'score_average_num_positive',
                     'score_num_positive']
        # get the total payoff of each play participant
        for total_payoff_type in type_list:
            groups_payoff_statistics = pd.DataFrame(
                columns=['min', 'max', 'mean', 'median', 'std', 'count'], index=group_list)
            # accurate_participants = self.results_payments.loc[(self.results_payments.prob_status == 'accurate') &
            #                                                   (self.results_payments.player_id_in_group == 1) &
            #                                                   (self.results_payments.status == 'play') &
            #                                                   (self.results_payments.pct_timeout <= 50)]
            #
            # if accurate_participants.empty:
            #     if split_gender:
            #         print(f'no data in experts_to_define for gender {self.gender} in function expert_payoff_analysis')
            #         continue
            #     if split_condition:
            #         print(f'no data in experts_to_define for condition {self.condition} in function '
            #               f'expert_payoff_analysis')
            #         continue
            #
            # if total_payoff_type == 'total_payoff_no_partner_timeout':
            #     accurate_participants = accurate_participants.loc[accurate_participants.partner_timeout == 0]
            # accurate_participants = accurate_participants.drop_duplicates('participant_code')
            # accurate_average_payoff = round(accurate_participants[total_payoff_type].mean(), 2)

            play_participants_groups = \
                self.results_payments.loc[(self.results_payments.status == 'play') &
                                          (self.results_payments.player_id_in_group == 1) &
                                          (self.results_payments.pct_timeout <= 50)]

            if play_participants_groups.empty:
                if split_gender:
                    print(f'no data in experts_to_define for gender {self.gender} in function expert_payoff_analysis')
                    continue
                if split_condition:
                    print(f'no data in experts_to_define for condition {self.condition} in function '
                          f'expert_payoff_analysis')
                    continue

            play_participants_groups = play_participants_groups.drop_duplicates('participant_code')

            x_points = list()
            y_points = list()
            group_list = play_participants_groups.prob_status.unique().tolist()

            for group in group_list:
                group_data = play_participants_groups.loc[play_participants_groups.prob_status == group]
                groups_payoff_statistics.loc[group] = create_statistics(group_data[total_payoff_type])
                groups_payoff_statistics_dict[total_payoff_type] = groups_payoff_statistics
                group_data = group_data[[total_payoff_type]].assign(count=1)
                group_data = group_data.groupby(total_payoff_type).count()
                x_points.append(group_data.index.values)
                y_points.append(group_data['count'].tolist())

            legend = group_list
            create_chart_bars(title=f'expert {total_payoff_type} with average for gender {self.gender} and condition '
                                    f'{self.condition}', x=x_points, y=y_points, xlabel='expert_total_payoff',
                              ylabel='number_of_participants', legend=legend)

        writer = pd.ExcelWriter(os.path.join(data_analysis_directory,
                                             f'groups_payoff_statistics_{self.gender}_{self.condition}.xlsx'))
        for total_payoff_type in type_list:
            groups_payoff_statistics_dict[total_payoff_type].to_excel(writer, sheet_name=total_payoff_type)
        writer.save()

        return

    def define_prob_status_for_dm(self):
        """assign the prob status of the expert to its decision maker partner"""
        expert_tyeps = self.results_payments.loc[self.results_payments.player_id_in_group == 1][
            ['pair_id', 'prob_status']].drop_duplicates('pair_id')
        expert_tyeps.columns = ['pair_id', 'player_expert_type']
        self.results_payments = self.results_payments.merge(expert_tyeps, on='pair_id', how='left')

        return

    def pct_entered_file(self):
        """
        This function create graphs for the % of DM choose the hotel per review index, per score, per hotel,
        per round number, per condition
        :return:
        """
        data_to_use = self.results_payments.loc[(self.results_payments['group_receiver_timeout'] == 0) &
                                                (self.results_payments.status == 'play') &
                                                (self.results_payments.player_id_in_group == 2)]
        columns_list = [['previous_review_id', 'previous_round_decision'],
                        ['previous_review_id', 'previous_round_decision', 'previous_round_lottery_result'],
                        ['previous_review_id', 'previous_round_decision', 'review_id'],
                        ['group_sender_answer_scores', 'previous_round_decision', 'previous_round_lottery_result']]
        round_number_lists = [[i] for i in range(2, 11)]
        round_number_lists.append(list(range(2, 11)))
        for column in columns_list:
            prev_review_id_data = pd.DataFrame()
            for round_number in round_number_lists:
                curr_data_to_use = data_to_use.loc[data_to_use.subsession_round_number.isin(round_number)]
                data_groupby = curr_data_to_use.groupby(by=column).agg({
                    'participant_code': 'count',
                    'group_sender_payoff': 'sum'
                })
                data_groupby['pct participant'] = 100 * data_groupby.group_sender_payoff / data_groupby.participant_code
                # create prev_review_id_data to save analysis for 'previous_review_id', 'previous_round_decision'
                data_groupby.columns =\
                    [f'round_{round_number}_count', f'round_{round_number}_sum', f'round_{round_number}_pct']
                if prev_review_id_data.empty:
                    prev_review_id_data = data_groupby
                else:
                    prev_review_id_data = prev_review_id_data.merge(
                        data_groupby, left_index=True, right_index=True, how='outer')

            # change prev_review_id_data to be only previous_review_id
            if 'review_id' in column:  # 2 columns of review_id
                second_merge_column = 'review_id'
            else:
                second_merge_column = None

            for i in range(1, len(prev_review_id_data.index.names)):
                prev_review_id_data[prev_review_id_data.index.names[i]] = prev_review_id_data.index.get_level_values(i)
            prev_review_id_data.index = prev_review_id_data.index.get_level_values(column[0])
            self.create_review_compare_file(prev_review_id_data, column=column, after_merge_name=column[0],
                                            add_compare_positive_negative_first=False,
                                            second_merge_column=second_merge_column)

        return

    def pct_entered_graph(self):
        """
        This function create graphs for the % of DM choose the hotel per review index, per score, per hotel,
        per round number, per condition
        :return:
        """
        data_to_use = self.results_payments.loc[(self.results_payments['group_receiver_timeout'] == 0) &
                                                (self.results_payments.status == 'play') &
                                                (self.results_payments.player_id_in_group == 2)]
        columns_list = [['previous_round_lottery_result', 'group_sender_answer_scores'],
                        ['previous_round_lottery_result', 'group_sender_answer_index'],
                        ['previous_round_decision', 'previous_round_lottery_result'],
                        ['previous_round_lottery_result_high', 'previous_round_decision'],
                        ['previous_score', 'group_sender_answer_scores'],
                        ['previous_round_lottery_result_high', 'group_sender_answer_scores'],
                        ['previous_chose_lose', 'group_sender_answer_scores'],
                        ['previous_chose_earn', 'group_sender_answer_scores'],
                        ['previous_not_chose_lose', 'group_sender_answer_scores'],
                        ['previous_not_chose_earn', 'group_sender_answer_scores'],
                        'history_lottery_result', 'history_decisions', 'history_lottery_result_high',
                        'history_chose_lose', 'history_chose_earn', 'history_not_chose_lose', 'history_not_chose_earn',
                        'previous_round_lottery_result', 'previous_round_lottery_result_high', 'time_spent_low',
                        'time_spent_med', 'time_spent_high', 'group_sender_answer_index', 'group_sender_answer_scores',
                        'hotel_id', 'subsession_round_number', 'group_average_score', 'group_median_score',
                        'all_review_len', 'positive_review_len', 'negative_review_len',
                        'positive_negative_review_len_prop', 'previous_round_lottery_result', 'previous_round_decision',
                        'review_id', 'previous_chose_lose', 'previous_chose_earn', 'previous_not_chose_lose',
                        'previous_not_chose_earn']
        review_id_data = pd.DataFrame()
        round_number_lists = [[i] for i in range(1, 11)]
        round_number_lists.append(list(range(1, 11)))
        if not split_condition:
            columns_list.append('subsession_condition')
        for column in columns_list:
            for round_number in round_number_lists:
                if column == 'subsession_round_number' and len(round_number) == 1:
                    continue   # per round number don't create graph for one round
                # for previous round or all history analysis - don't analysis the first round
                if True in ['previous' in col for col in column] or 'previous' in column or 'history' in column:
                    if round_number == [1]:  # for the first round is not relevant
                        continue
                    elif len(round_number) > 1:  # remove the first round from list
                        round_number = round_number.copy()
                        round_number.remove(1)
                curr_data_to_use = data_to_use.loc[data_to_use.subsession_round_number.isin(round_number)]
                step = 5
                x_is_float = False
                if column == 'positive_negative_review_len_prop':
                    curr_data_to_use = curr_data_to_use.loc[curr_data_to_use['positive_negative_review_len_prop'] <= 2]
                    step = 0.1
                else:
                    curr_data_to_use = curr_data_to_use

                if 'score' in column:
                    x_is_float = True
                if 'len' in column:
                    step = 50
                # % of participant entered per column

                data_groupby = curr_data_to_use.groupby(by=column).agg({
                    'participant_code': 'count',
                    'group_sender_payoff': 'sum'
                })
                data_groupby['pct participant'] = 100 * data_groupby.group_sender_payoff / data_groupby.participant_code

                if column == 'positive_negative_review_len_prop':
                    use_all_xticks = True
                else:
                    use_all_xticks = False

                if type(column) is list:
                    data = pd.DataFrame(data_groupby['pct participant'])
                    create_bar_from_df(data=data, xlabel=column, ylabel='% of participants', add_table=False, rot=False,
                                       title=f'% DM entered per {column} for condition {self.condition} and gender '
                                             f'{self.gender} and rounds {round_number}',
                                       curr_date_directory=data_analysis_directory, add_text_label=True)
                elif column in ['review_id', 'positive_review_len', 'negative_review_len', 'all_review_len',
                                'previous_round_lottery_result', 'positive_negative_review_len_prop'] or\
                        'history' in column:
                    data = pd.DataFrame(data_groupby['pct participant'])
                    print(f'column is: {column}')
                    if 'history' not in column:
                        data.index = data.index.astype(int)
                    create_bar_from_df(data=data, xlabel=column, ylabel='% of participants', add_table=False, rot=False,
                                       title=f'% DM entered per {column} for condition {self.condition} and gender '
                                             f'{self.gender} and rounds {round_number}',
                                       curr_date_directory=data_analysis_directory, add_text_label=True)
                    if column == 'review_id':
                        data_groupby = data_groupby[['pct participant']]
                        data_groupby.columns = [f'round_{round_number}']
                        if review_id_data.empty:
                            review_id_data = data_groupby
                        else:
                            review_id_data = review_id_data.merge(data_groupby, left_index=True, right_index=True,
                                                                  how='outer')

                else:
                    if column != 'subsession_condition':
                        create_chart_bars(title=f'% DM entered per {column} for condition {self.condition} and '
                                                f'gender {self.gender} and rounds {round_number}',
                                          x=[data_groupby.index.values], xlabel=column,
                                          y=[data_groupby['pct participant'].values.tolist()],
                                          ylabel='% of participants',
                                          percentage_graph=True, curr_date_directory=data_analysis_directory,
                                          x_is_float=x_is_float, step=step, use_all_xticks=use_all_xticks)
                    else:
                        create_histogram(title=f'% DM entered per {column} for condition {self.condition} and gender '
                                               f'{self.gender} and rounds {round_number}',
                                         x=data_groupby['pct participant'], xlabel=column,
                                         ylabel='% of participants',
                                         add_labels=False, curr_date_directory=data_analysis_directory)

        self.create_review_compare_file(review_id_data, column='review_id', add_compare_positive_negative_first=True)

        return

    def create_review_compare_file(self, review_id_data: pd.DataFrame, column: Union[str, list],
                                   add_compare_positive_negative_first: bool=True, after_merge_name: str= 'review_id',
                                   second_merge_column=None):
        """
        This function create csv file to compare reviews results
        :param review_id_data: the df with the review_id % of DM that entered per round
        :param add_compare_positive_negative_first: whether to run compare_positive_negative_first function
        :param column: the name of columns
        :param after_merge_name: the name of the column after merge
        :param second_merge_column: if we want to merge twice - the second column we want to merge on
        :return:
        """
        reviews_to_merge = self.results_payments[[
            after_merge_name, 'group_sender_answer_reviews', 'group_sender_answer_scores', 'positive_review_len',
            'negative_review_len']].drop_duplicates()
        if 'previous' in after_merge_name:
            prefix = 'previous_'
        else:
            prefix = ''
        reviews_to_merge.columns = [after_merge_name, f'{prefix}review', f'{prefix}score',
                                    f'{prefix}positive_review_len', f'{prefix}negative_review_len']
        review_id_data = review_id_data.merge(reviews_to_merge, left_index=True, right_on=after_merge_name,
                                              how='left').reset_index(drop=True)
        if second_merge_column is not None:
            reviews_to_merge.columns =\
                [second_merge_column, 'review', 'score', 'positive_review_len', 'negative_review_len']
            review_id_data = review_id_data.merge(reviews_to_merge, how='left').reset_index(drop=True)

        review_id_data.to_csv(os.path.join(data_analysis_directory,
                                           f'% DM entered per {column} for condition {self.condition} and gender '
                                           f'{self.gender}.csv'))
        if add_compare_positive_negative_first:
            review_id_data = compare_positive_negative_first(review_id_data)
            review_id_data.to_csv(
                os.path.join(data_analysis_directory, f'% DM entered per positive negative first for condition '
                                                      f'{self.condition} and gender {self.gender}.csv'))

        return

    def analyze_expert_answer(self):
        """
        This function plot the expert answer vs the round number
        :return:
        """
        data_to_use = self.results_payments.loc[(self.results_payments.player_id_in_group == 1) &
                                                (self.results_payments.status == 'play')]
        expert_answer = data_to_use.groupby(by='subsession_round_number').group_sender_answer_scores.mean()
        expert_answer = expert_answer.round(2)

        create_bar_from_df(data=expert_answer, xlabel='Round Number', ylabel='Expert Average Answer',
                           add_table=False, rot=False,
                           title=f'Histogram Expert answer vs round number for {self.condition} condition '
                                 f'and {self.gender}',
                           curr_date_directory=data_analysis_directory, add_text_label=True,
                           max_height=expert_answer.max(), convert_to_int=False, label_rotation='horizontal')

        create_point_plot(points_x=[expert_answer.index], points_y=[expert_answer.values],
                          legend=['Expert Average Answer'],
                          title=f'Expert answer vs round number for {self.condition} condition and {self.gender}',
                          xlabel='Round Number', ylabel='Expert Average Answer', add_truth_telling=False,
                          curr_date_directory=data_analysis_directory, add_line_between_points=True)

        # diff between true average to expert average
        expert_diff_answer = data_to_use.groupby(by='subsession_round_number').\
            agg({'group_average_score': 'mean',
                 'group_sender_answer_scores': 'mean'})
        expert_diff_answer['diff'] = expert_diff_answer.group_sender_answer_scores -\
                                     expert_diff_answer.group_average_score
        expert_diff_answer = expert_diff_answer['diff']
        expert_diff_answer = expert_diff_answer.round(2)

        create_bar_from_df(data=expert_diff_answer, xlabel='Round Number', ylabel='Expert Average Answer',
                           add_table=False, rot=False,
                           title=f'Histogram Expert answer diff vs round number for {self.condition} condition '
                                 f'and {self.gender}',
                           curr_date_directory=data_analysis_directory, add_text_label=True,
                           max_height=expert_answer.max(), convert_to_int=False, label_rotation='horizontal')

        create_point_plot(points_x=[expert_diff_answer.index], points_y=[expert_diff_answer.values],
                          legend=['Expert Average Answer'],
                          title=f'Expert answer diff vs round number for {self.condition} condition and {self.gender}',
                          xlabel='Round Number', ylabel='Expert Average Answer', add_truth_telling=False,
                          curr_date_directory=data_analysis_directory, add_line_between_points=True)

        unique_averages = data_to_use.group_average_score.unique().tolist()
        for hotel_average_score in unique_averages:
            hotel_data = data_to_use.loc[data_to_use.group_average_score == hotel_average_score]
            hotel_group_data = hotel_data.groupby(by='subsession_round_number').group_sender_answer_scores.mean()
            hotel_group_data = hotel_group_data.round(2)
            create_point_plot(points_x=[hotel_group_data.index], points_y=[hotel_group_data.values],
                              legend=['Expert Average Score'],
                              title=f'Expert average score VS the decision maker expected payoff for hotel '
                                    f'average score {hotel_average_score}, {self.condition} condition and {self.gender}',
                              xlabel='Round Number', ylabel='Expert Average Score',
                              curr_date_directory=data_analysis_directory, add_line_between_points=True,
                              add_truth_telling=False)

            create_bar_from_df(data=hotel_group_data, xlabel='Round Number', ylabel='Expert Average Answer',
                               add_table=False, rot=False,
                               title=f'Histogram Expert average score VS the decision maker expected payoff for '
                                     f'average score {hotel_average_score} {self.condition} condition and {self.gender}',
                               curr_date_directory=data_analysis_directory, add_text_label=True,
                               max_height=hotel_group_data.max(), convert_to_int=False, label_rotation='horizontal')

        return

    def analyze_dm_payoff(self):
        """
        This function plot the decision maker payoff vs the round number
        :return:
        """
        data_to_use = self.results_payments.loc[(self.results_payments.player_id_in_group == 2) &
                                                (self.results_payments.status == 'play')]

        for column, column_name in [['player_payoff', ''], ['dm_expected_payoff', 'Expected ']]:
            dm_payoff = data_to_use.groupby(by='subsession_round_number')[column].mean()
            dm_payoff = dm_payoff.round(2)

            create_bar_from_df(data=dm_payoff, xlabel='Round Number',
                               ylabel=f'Decision Maker Average {column_name}Payoff',
                               add_table=False, rot=False,
                               title=f'Decision Maker Average {column_name}Payoff vs round number for '
                                     f'{self.condition} condition and {self.gender}',
                               curr_date_directory=data_analysis_directory, add_text_label=True,
                               max_height=dm_payoff.max(), label_rotation='horizontal')

            for round_num in range(1, 11):
                round_data = data_to_use.loc[data_to_use.subsession_round_number == round_num]
                round_group_data = round_data.groupby(by=column).participant_code.count()
                create_bar_from_df(data=round_group_data, xlabel='DM payoff', ylabel='Number of participants',
                                   add_table=False, rot=False,
                                   title=f'Number of DM get each {column_name}payoff for {self.condition} condition '
                                         f'and {self.gender} and round number {round_num}',
                                   max_height=dm_payoff.max()+10, curr_date_directory=data_analysis_directory,
                                   add_text_label=True, label_rotation='horizontal')

        return

    def compare_experts_choices(self):
        """
        This function compare the index that the experts chose in each hotel between the condition
        :return:
        """

        data_function = self.results_payments.loc[(self.results_payments.status == 'play') &
                                                  (self.results_payments.player_id_in_group == 1)]
        for param in ['hotel_id', 'positive_negative_review_len_prop', 'group_average_score', 'group_median_score']:
            if self.condition != 'verbal' and param == 'positive_negative_review_len_prop':
                continue
            else:
                # create bar with the number of expert chose each index per param
                for column in ['group_sender_answer_index', 'group_sender_answer_scores']:
                    data_to_use = data_function[['participant_code', column, param]]
                    pivot = data_to_use.pivot_table(index=[param], columns=column,
                                                    values=['participant_code'], aggfunc=[np.count_nonzero])
                    if column == 'group_sender_answer_index':
                        pivot.columns = list(range(1, 8))
                        add_table = True
                        pivot.columns.name = 'index'
                    else:
                        pivot.columns = pivot.columns.levels[2]
                        add_table = False
                        pivot.columns.name = 'score'
                    title = f"Number of experts chose each review's {column} per {param} " \
                            f"\nfor condition {self.condition} and gender {self.gender}"
                    xlabel = param
                    ylabel = 'Number of experts'

                    if param == 'positive_negative_review_len_prop':
                        create_bar_from_df(data=pivot, title=title, xlabel=xlabel, ylabel=ylabel,
                                           curr_date_directory=data_analysis_directory, add_table=False, rot=False,
                                           add_point=33.5)
                    else:
                        create_bar_from_df(data=pivot, title=title, xlabel=xlabel, ylabel=ylabel,
                                           curr_date_directory=data_analysis_directory, add_table=add_table)
        return

    def expert_answer_vs_ep(self):
        """
        This function create a point plot for the expert answer vs the DM expected payoff (the average above the scores)
        :return:
        """
        data_to_use = self.results_payments.loc[(self.results_payments.player_id_in_group == 1) &
                                                (self.results_payments.status == 'play')]

        averages = data_to_use.groupby(by='group_average_score').group_sender_answer_scores.mean()
        print_max_min_values = dict()
        for index, row in self.hotel_reviews.iterrows():
            print_max_min_values[round(row['average_score'], 2)] = [row['min_score'], row['score_6']]

        create_point_plot(points_x=[averages.index], points_y=[averages.values], legend=['Expert Average Score'],
                          title=f'Expert average score VS the decision maker expected payoff for {self.condition} and '
                                f'{self.gender}', print_max_min_values=print_max_min_values,
                          xlabel='Decision Maker Expected Payoff', ylabel='Expert Average Score',
                          curr_date_directory=data_analysis_directory, add_line_between_points=True)

        return

    def pct_entered_two_parameters(self):
        """
        This function compare the index that the experts chose in each hotel between the condition
        :return:
        """
        # get only DM

        data_to_use = self.results_payments.loc[(self.results_payments.player_id_in_group == 2) &
                                                (self.results_payments.status == 'play')]
        data_to_use = data_to_use[['group_sender_answer_index', 'subsession_round_number', 'group_sender_payoff']]

        pivot = data_to_use.pivot_table(index=['subsession_round_number'], columns='group_sender_answer_index',
                                        values=['group_sender_payoff'], aggfunc=[np.size, np.sum])
        count_columns = [f'count_index_{i}' for i in range(1, 8)]
        sum_columns = [f'sum_index_{i}' for i in range(1, 8)]
        pivot.columns = count_columns + sum_columns
        for i in range(1, 8):
            pivot[i] = round(100 * pivot[f'sum_index_{i}'] / pivot[f'count_index_{i}'],1)

        columns_to_use = [i for i in range(1, 8)]
        pivot = pivot[columns_to_use]

        title = f"% DM entered per round and review index for condition {self.condition} and gender {self.gender}"
        xlabel = 'subsession_round_number'
        ylabel = '% DM entered'

        create_bar_from_df(data=pivot, title=title, xlabel=xlabel, ylabel=ylabel,
                           curr_date_directory=data_analysis_directory, add_table=True)

        return

    def compare_experts_choices_per_condition(self):
        """
        This function compare the index that the experts chose in each hotel between the condition
        :return:
        """
        if not split_condition:
            data_to_use = self.results_payments[
                ['participant_code', 'hotel_id', 'group_sender_answer_index', 'subsession_condition']]
            pivot = data_to_use.pivot_table(index=['hotel_id', 'subsession_condition'], columns='group_sender_answer_index',
                                            values=['participant_code'], aggfunc=[np.count_nonzero])
            pivot.columns = list(range(1, 8))
            pivot.columns.name = 'review'
            title = "Number of experts chose each review's index"
            xlabel = '(hotel ID, condition)'
            ylabel = 'Number of experts'

            create_bar_from_df(data=pivot, title=title, xlabel=xlabel, ylabel=ylabel,
                               curr_date_directory=data_analysis_directory, add_table=True)

        return

    def expert_answer_vs_average_score(self):
        """
        This function create a bar that show the % of experts that chose the most close average score,
        the median score and the bias from the average
        :return:
        """

        expert_data = self.results_payments.loc[(self.results_payments.player_id_in_group == 1) &
                                                (self.results_payments.status == 'play')]

        median_vs_average = dict()

        for column in [['chose_most_average_close', 'chose the score that is most closer to the average score'],
                       ['chose_median_score', 'chose the median score'],
                       ['not_chose_closer_average_or_median',
                        'did not choose score that is most closer to average or the median score']]:
            print(f'% of experts {column[1]} in all rounds is: 'f'{expert_data[column[0]].mean().round(2)}')
            expert_choices = expert_data.groupby(by='subsession_round_number')[column[0]].mean()
            expert_choices = expert_choices*100
            expert_choices = expert_choices.round(1)
            median_vs_average[column[0]] = expert_choices

            create_bar_from_df(data=expert_choices, xlabel='Round Number', ylabel='% of experts',
                               add_table=False, rot=False, convert_to_int=False,
                               title=f'% of experts {column[1]} per round number for {self.condition} condition '
                                     f'and {self.gender}',
                               curr_date_directory=data_analysis_directory, add_text_label=True,
                               max_height=expert_choices.max(), label_rotation='horizontal')

        median_vs_average = pd.DataFrame(median_vs_average, index=list(range(1, 11)))
        create_bar_from_df(data=median_vs_average, xlabel='Round Number', ylabel='% of experts',
                           add_table=False, rot=False, stacked=False, convert_to_int=False,
                           title=f'% of experts chose the score that is closer to the average score vs the median '
                                 f'score\nper round number for {self.condition} condition and {self.gender}',
                           curr_date_directory=data_analysis_directory, add_text_label=True,
                           y_ticks_list=list(range(0, 101, 10)), figsize=(15, 5), autolabel_fontsize=8,
                           max_height=median_vs_average.max().max(), label_rotation='horizontal')
        print(f'Bias of experts answer from the average score in all rounds is: '
              f'{expert_data.bias_from_most_close_avg.mean().round(2)}')

        for column in ['subsession_round_number', 'hotel_id']:
            expert_choices = expert_data.groupby(by=column).bias_from_most_close_avg.mean()
            expert_choices = expert_choices.round(2)

            create_bar_from_df(data=expert_choices, xlabel=column, ylabel='Average Bias',
                               add_table=False, rot=False, convert_to_int=False,
                               title=f'Bias of experts answer from the average score vs {column} for '
                                     f'{self.condition} condition and {self.gender}',
                               curr_date_directory=data_analysis_directory, add_text_label=True,
                               max_height=expert_choices.max(), label_rotation='horizontal')

    def dm_entered_per_review_features(self):
        """
        This function compare the % of decision makers that chose the hotel option per feature from the
        text features I extracted from the reviws
        :return:
        """
        manual_features = pd.read_excel('/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/'
                                        'language_prediction/data/verbal/manual_binary_features.xlsx')
        data_to_use = self.results_payments.loc[(self.results_payments['group_receiver_timeout'] == 0) &
                                                (self.results_payments.player_id_in_group == 2) &
                                                (self.results_payments.status == 'play')][
            ['subsession_round_number', 'review_id', 'group_sender_payoff']]
        data_to_use.review_id = data_to_use.review_id.astype(int)
        data_to_use = data_to_use.merge(manual_features, how='left', on='review_id')
        columns_to_compare = list(manual_features.columns)
        columns_to_compare.remove('review_id')
        columns_to_compare.remove('review')

        round_number_lists = list()  # [[i] for i in range(1, 11)]
        round_number_lists.append(list(range(1, 11)))
        data_dict = dict()
        for column in columns_to_compare:
            for round_number in round_number_lists:
                curr_data_to_use = data_to_use.loc[data_to_use.subsession_round_number.isin(round_number)]
                curr_data_to_use = curr_data_to_use.groupby(by=column).group_sender_payoff.mean()
                curr_data_to_use = curr_data_to_use.round(2)
                create_bar_from_df(data=curr_data_to_use, xlabel=column, ylabel='% of participants',
                                   add_table=False, rot=False, max_height=curr_data_to_use.max(),
                                   title=f'% DM entered based on {column} for condition {self.condition} and gender '
                                         f'{self.gender} and rounds {round_number}', label_rotation='horizontal',
                                   curr_date_directory=data_analysis_directory, add_text_label=True)
                data_dict[column] = curr_data_to_use
        data_df = pd.DataFrame(data_dict, index=[0, 1])
        data_df.to_csv(os.path.join(data_analysis_directory,
                                    f'% DM entered based on review features for condition {self.condition} and gender '
                                    f'{self.gender}.csv'))

    def eval_real_score_analysis(self):
        """
        This function compare per hotel the % of exert chose each index and the different between the real score
        and the average estimation
        :return:
        """

        data_to_use = self.results_payments.loc[(self.results_payments.player_id_in_group == 1) &
                                                (self.results_payments.status == 'play')]
        unique_averages = data_to_use.group_average_score.unique().tolist()
        unique_averages.sort()
        index_dict = defaultdict(list)
        eval_score_dict = defaultdict(list)
        df_index = list()
        for hotel_average_score in unique_averages:
            hotel_data = data_to_use.loc[data_to_use.group_average_score == hotel_average_score]
            expert_avg_score = hotel_data.group_sender_answer_scores.mean().round(2)
            df_index.append((hotel_average_score, expert_avg_score))
            hotel_eval_data = score_eval_data.loc[score_eval_data.hotel_avg_score == hotel_average_score]
            hotel_group_data = hotel_data.groupby(by='group_sender_answer_index').participant_code.count()
            for idx in range(1, 8):
                eval_score_dict[int(idx)].append((round(hotel_eval_data.average_answer.values[idx-1], 1),
                                                  round(hotel_eval_data.review_real_score.values[idx-1], 1)))
                if idx in hotel_group_data.index:
                    index_dict[int(idx)].append(round(100 * hotel_group_data.loc[idx]/hotel_group_data.sum(), 2))
                else:
                    index_dict[int(idx)].append(0.0)

        data_to_plot = pd.DataFrame(index_dict, index=df_index)
        autolabel_text = [item for sublist in eval_score_dict.values() for item in sublist]

        create_bar_from_df(data=data_to_plot, curr_date_directory=data_analysis_directory, add_table=False,
                           title=f'% Experts chose each index per hotel for condition {self.condition} and gender '
                                 f'{self.gender}\nThe legend are the reviews index, the text is the (estimated score, '
                                 f'original score)', xlabel="Hotel Average Score, Experts' Average Answer",
                           ylabel='% experts chose each index', stacked=False, figsize=(17, 5), add_text_label=True,
                           convert_to_int=False, autolabel_text=autolabel_text, autolabel_fontsize=8,
                           y_ticks_list=list(range(0, 85, 10)))

        return

    def calculate_linear_score(self):
        """
        This function caclculate for each pair the DM EV linear score and the sum (expert+DM EV) linear score
        :return:
        """
        data_to_use = self.results_payments.loc[
            (self.results_payments.player_id_in_group == 1) & (self.results_payments.status == 'play')][
            ['dm_expected_payoff', 'subsession_round_number', 'pair_id', 'group_sender_payoff']]
        data_to_use['sum_payoff'] = data_to_use.dm_expected_payoff + data_to_use.group_sender_payoff
        data_to_use['coefficient'] = data_to_use.apply(linear_score, axis=1)
        data_to_use['linear_score_dm_ev'] = data_to_use.dm_expected_payoff * data_to_use.coefficient
        data_to_use['linear_sum'] = data_to_use.sum_payoff * data_to_use.coefficient
        data_to_use['linear_expert_payoff'] = data_to_use.group_sender_payoff * data_to_use.coefficient

        linear_score_dm_ev = data_to_use.groupby(by='pair_id').linear_score_dm_ev.sum()
        linear_sum = data_to_use.groupby(by='pair_id').linear_sum.sum()
        linear_expert_payoff = data_to_use.groupby(by='pair_id').linear_expert_payoff.sum()

        to_save = pd.concat([linear_score_dm_ev, linear_expert_payoff, linear_sum], axis=1)
        to_save.to_csv(os.path.join(data_analysis_directory, 'linear_scores.csv'))

        return


def main(main_gender='all genders', main_condition='all_condition'):
    data_analysis_obj = DataAnalysis(main_gender, main_condition)
    data_analysis_obj.set_player_partner_timeout_answer()
    data_analysis_obj.set_current_round_measures()
    data_analysis_obj.set_previous_round_measures()
    data_analysis_obj.define_player_status()
    data_analysis_obj.calculate_linear_score()
    data_analysis_obj.eval_real_score_analysis()
    data_analysis_obj.average_most_close_hotel_id()
    data_analysis_obj.dm_entered_per_review_features()
    data_analysis_obj.expert_answer_vs_average_score()
    data_analysis_obj.analyze_dm_payoff()
    data_analysis_obj.expert_answer_vs_ep()
    data_analysis_obj.analyze_expert_answer()
    data_analysis_obj.set_all_history_measures()
    data_analysis_obj.print_pairs()
    data_analysis_obj.set_expert_answer_scores_reviews_len_verbal_cond()
    data_analysis_obj.compare_experts_choices_per_condition()
    data_analysis_obj.compare_experts_choices()
    data_analysis_obj.pct_entered_two_parameters()
    data_analysis_obj.round_analysis()
    data_analysis_obj.how_much_pay()
    data_analysis_obj.sender_answer_analysis()
    data_analysis_obj.bonus_analysis()
    data_analysis_obj.time_spent_analysis()
    data_analysis_obj.define_total_payoff_no_partner_timeout()
    data_analysis_obj.effectiveness_measure()
    data_analysis_obj.expert_payoff_analysis()
    data_analysis_obj.define_prob_status_for_dm()
    data_analysis_obj.pct_entered_file()
    data_analysis_obj.pct_entered_graph()
    # data_analysis_obj.plot_expected_analysis()
    # data_analysis_obj.anaylze_zero_one_accurate()
    # data_analysis_obj.not_zero_one_changed_to_zero_one()
    # data_analysis_obj.probability_for_one()
    # data_analysis_obj.cutoff_analysis()
    # data_analysis_obj.check_payoff_vs_average_ev()
    # data_analysis_obj.experts_always_overdo_reduce()
    # data_analysis_obj.create_table_prob_ev()
    data_analysis_obj.results_payments.to_csv(os.path.join(data_analysis_directory, 'results_payments_status.csv'))


if __name__ == '__main__':
    if split_gender:
        main_data_analysis_directory = data_analysis_directory
        for gender in ['Male', 'Female']:
            print(f'start analyze for gender {gender}')
            data_analysis_directory = os.path.join(main_data_analysis_directory, gender)
            if not (os.path.exists(data_analysis_directory)):
                os.makedirs(data_analysis_directory)
            main(main_gender=gender)

    if split_condition:
        main_data_analysis_directory = data_analysis_directory
        for condition in ['verbal', 'num']:
            print(f'start analyze for condition {condition}')
            data_analysis_directory = os.path.join(main_data_analysis_directory, condition)
            if not (os.path.exists(data_analysis_directory)):
                os.makedirs(data_analysis_directory)
            main(main_condition=condition)

    else:
        main()
