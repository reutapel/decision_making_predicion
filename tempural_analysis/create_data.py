__all__ = ['CreateData']
import pandas as pd
import logging
import copy
import numpy as np


class CreateData:
    """
    This class load the data and create a data frame with all the optional features and labels
    """
    def __init__(self):
        self.features_labels = None
        self.data = None
        self.base_features = None

    def sign_column(self, new_column_name: str, given_column_name: str):
        """
        create a new column which is the sign of a given column:  0 if 0, 1 if positive, -1 if negative
        :param new_column_name: the new column we want to create
        :param given_column_name: the column we want to use to create the new column
        :return:
        """
        self.features_labels[new_column_name] = 0
        self.features_labels[new_column_name] = np.where(self.features_labels[given_column_name] > 0, 1, -1)

    def load_data(self, data_path: str, not_agent_data: bool):
        """
        This function load the data
        :param data_path: the path to the data to load
        :param not_agent_data: False if this is an agent experiment, True otherwise
        :return:
        """

        logging.info(f'Start loading data')
        if 'xlsx' in data_path:
            self.data = pd.read_excel(data_path, sheetname='data_to_use')
        else:  # load csv
            self.data = pd.read_csv(data_path)
        # get only pairs that both participants played, and only experts- to remove duplicates
        if not_agent_data:
            self.data = self.data.loc[(self.data.status == 'play') & (self.data.player_id_in_group == 1)]
            self.data.index = self.data.pair_id + '_' + self.data.subsession_round_number.map(str)
        else:
            self.data.index = self.data.participant_code + '_' + self.data.subsession_round_number.map(str)
        self.data = self.data.sort_values(by=['participant_code', 'subsession_round_number'])

        logging.info(f'Original data size is: {self.data.shape}')

        return

    def create_features_label(self, labels: list, features: list, add_features_list: list, appendix: str,
                              sample_participants: [bool, int]= (False, 0)):
        """
        This function create all the optional features and labels to use during the evaluate grid search
        :param labels: list of optional labels
        :param features: list of optional features
        :param add_features_list: list of features we will add in some of the experiments we are going to run
        :param appendix: appendix of the columns: player/group (according to the experiment type)
        :param sample_participants: whether to use all the participants data or to sample some of them
        :return:
        """
        logging.info(f'Start creating features')
        if sample_participants[0]:  # if we need to sample some of the participants
            participants = self.data.participant_code.unique()
            participants_to_use = np.random.choice(a=participants, size=sample_participants[1], replace=False)
            self.data = self.data.loc[self.data.participant_code.isin(participants_to_use)]
        # insert base features
        base_features = list(set(labels + features + add_features_list))
        self.features_labels = copy.deepcopy(self.data[base_features])
        for label in labels:
            self.features_labels[label] = np.where(self.features_labels[label] == 0, -1, 1)

        # add calculated features:
        # gender:
        self.features_labels['male'] = np.where(self.data[appendix + '_gender'] == 'Male', 1, 0)
        # self.features_labels['female'] = np.where(self.data[appendix + '_gender'] == 'Female', 1, 0)

        # lottery EV and average
        self.features_labels['ev_lottery'] = \
            self.features_labels[appendix + '_x_lottery'] * self.features_labels[appendix + '_p_lottery'] + \
            self.features_labels[appendix + '_y_lottery'] * (1 - self.features_labels[appendix + '_p_lottery'])
        self.sign_column('sign_lottery_ev', 'ev_lottery')

        self.features_labels['average_lottery'] =\
            0.5 * self.features_labels[appendix + '_x_lottery'] + 0.5 * self.features_labels[appendix + '_y_lottery']
        self.sign_column('sign_lottery_average', 'average_lottery')

        # EV based on expert answer
        self.features_labels['expert_ev_lottery'] = \
            self.features_labels[appendix + '_x_lottery'] * self.features_labels[appendix + '_sender_answer'] + \
            self.features_labels[appendix + '_y_lottery'] * (1 - self.features_labels[appendix + '_sender_answer'])
        self.sign_column('sign_expert_ev_lottery', 'expert_ev_lottery')

        # sender answer diffs
        self.features_labels['sender_answer_ev_diff'] = \
            self.features_labels[appendix + '_sender_answer'] - self.features_labels.ev_lottery
        self.sign_column('sign_sender_answer_ev_diff', 'sender_answer_ev_diff')

        self.features_labels['sender_answer_average_diff'] = \
            self.features_labels[appendix + '_sender_answer'] - self.features_labels.average_lottery
        self.sign_column('sign_sender_answer_average_diff', 'sender_answer_average_diff')

        self.features_labels['sender_answer_lottery_p_diff'] = \
            self.features_labels[appendix + '_sender_answer'] - self.features_labels[appendix + '_p_lottery']
        self.sign_column('sign_sender_answer_lottery_p_diff', 'sender_answer_lottery_p_diff')

        # lottery range:
        self.features_labels['lottery_range'] =\
            self.features_labels[appendix + '_x_lottery'] - self.features_labels[appendix + '_y_lottery']

        # ambiguous EV: (lottery_min + lottery_average)/2 + lottery_ev
        self.features_labels['ambiguous_ev'] =\
            (self.features_labels[appendix + '_y_lottery'] + self.features_labels.average_lottery)/2 +\
            self.features_labels.ev_lottery

        # sign_ev: the lottery sign expected value- use only the sign of x and y
        # --> p_lottery - (1-p_lottery) --> 2*p_lottery-1
        self.features_labels['sign_ev'] = 2*self.features_labels[appendix + '_p_lottery'] - 1

        # expert type: zero-one or other
        self.features_labels['zero_one_expert'] = \
            np.where(self.data[appendix + '_expert_type'].str.contains('zero_one'), 1, 0)

        # expert wrong: say higher than 0.8 or 0.5 and the lottery result was negative,
        # or lower than prob and the lottery result was positive
        for prob in [0.5, 0.8]:
            self.features_labels['expert_wrong_' + str(prob)] =\
                np.where(((self.features_labels[appendix + '_sender_answer'] >= prob) &
                         (self.features_labels[appendix + '_lottery_result'] < 0)) |
                         ((self.features_labels[appendix + '_sender_answer'] < prob) &
                         (self.features_labels[appendix + '_lottery_result'] > 0)), 1, 0)

        # the best response hindsight: 1- certainty is lottery_result<0, 0 otherwise
        self.features_labels['best_response'] = np.where(self.features_labels[appendix + '_lottery_result'] < 0, 1, 0)

        # the payoff in this trial- lottery result is chose 0, 0 otherwise
        self.features_labels['payoff'] = np.where(self.features_labels[appendix + '_receiver_choice'] == 0,
                                                  self.features_labels[appendix + '_lottery_result'], 0)

        logging.info(f'Features and labels size is: {self.features_labels.shape}')

        # the base features that will be used in all runs
        self.base_features = [item for item in self.features_labels.columns.tolist() if
                              item not in add_features_list + labels]

        return
