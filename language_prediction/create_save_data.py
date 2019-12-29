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


class CreateSaveData:
    """
    This class load the data, create the seq data and save the new data with different range of K
    """
    def __init__(self, load_file_name: str, total_payoff_label: bool=True, label: str='total_payoff',
                 use_seq: bool=True, use_prev_round: bool=False, use_manual_features: bool=False, features_file: str='',
                 features_file_type: str=''):
        """
        :param load_file_name: the raw data file name
        :param total_payoff_label: if the label is the total payoff of the expert or the next rounds normalized payoff
        :param label: the name of the label
        :param use_seq: if to create a sample which is a seq or a single round
        :param use_prev_round: if to use the previous round data: review, decision, lottery result
        :param use_manual_features: if we use manual features - need to get the review id
        :param features_file: if using fix features- the name of the features file
        :param features_file_type: the type of file for the fix features
        """
        print(f'Start create and save data for file: {load_file_name}')
        logging.info('Start create and save data for file: {}'.format(load_file_name))

        # load the data and keep only play pairs and only one row for each round for each pair
        columns_to_use = ['pair_id', 'status', 'subsession_round_number', 'group_sender_answer_reviews',
                          'group_receiver_choice', 'group_lottery_result', 'review_id', 'previous_round_lottery_result',
                          'previous_round_decision', 'previous_review_id', 'group_average_score', 'lottery_result_low',
                          'lottery_result_med1', 'lottery_result_high', 'previous_round_lottery_result_low',
                          'previous_round_lottery_result_med1',
                          'previous_round_lottery_result_high', 'previous_group_average_score_low',
                          'previous_group_average_score_high', 'player_id_in_group']
        self.data = pd.read_csv(os.path.join(data_directory, f'{load_file_name}.csv'), usecols=columns_to_use)
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
                               'lottery_result_low', 'lottery_result_med1', 'lottery_result_high',
                               'previous_round_lottery_result_low', 'previous_round_lottery_result_high',
                               'previous_group_average_score_low', 'previous_group_average_score_high',
                               'previous_round_lottery_result_med1']]
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
        print(f'Number of pairs in data: {self.pairs.shape}')

    def create_manual_features_data(self):
        """
        This function create 10 samples with different length from each pair data raw
        :return:
        """

        print(f'Start creating manual features data')
        logging.info('Start creating manual features data')

        # create the 10 samples for each pair
        meta_data_columns = ['k_size', 'pair_id', 'sample_id']
        for pair in self.pairs:
            if self.use_prev_round:
                columns_to_use = ['review_id', 'previous_round_lottery_result_low',
                                  'previous_round_lottery_result_med1', 'previous_round_lottery_result_high',
                                  'previous_round_decision', 'previous_review_id']
            else:
                columns_to_use = ['review_id']
                # , 'previous_round_lottery_result_low', 'previous_round_lottery_result_high',
                # 'previous_round_lottery_result', 'previous_round_decision']

            data = self.data.loc[self.data.pair_id == pair][columns_to_use]

            # first merge for the review_id for the current round
            if self.reviews_features.shape[1] == 2:  # Bert features
                reviews = pd.DataFrame()
                for i in self.reviews_features.index:
                    check = pd.DataFrame(self.reviews_features.at[i, 'review_features']).append(
                        pd.DataFrame([self.reviews_features.at[i, 'review_id']], index=['review_id']))
                    reviews = pd.concat([reviews, check], axis=1, ignore_index=True)
                data = data.merge(reviews.T, left_on='review_id', right_on='review_id', how='left')
            else:
                data = data.merge(self.reviews_features, left_on='review_id', right_on='review_id', how='left')

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
            # don't use first round if use_prev_round
            if self.use_prev_round:
                data = data.loc[data.k_size > 1]
            # concat to all data
            self.final_data = pd.concat([self.final_data, data], axis=0, ignore_index=True)

        if self.use_prev_round:
            file_name = f'all_data_{self.label}_label_prev_round_{self.features_file}'
        else:
            file_name = f'all_data_{self.label}_label_{self.features_file}'

        self.final_data.to_csv(os.path.join(data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(data_directory, f'{file_name}.pkl'))

        print(f'Finish creating manual features data')
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
                        'previous_round_lottery_result_med1', 'previous_group_average_score_high']

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
        columns = [f'text_{i}' for i in range(self.number_of_rounds)]
        columns.extend([f'prev_payoff_{i}' for i in range(self.number_of_rounds)])
        columns.extend([f'prev_result_low_{i}' for i in range(self.number_of_rounds)])
        columns.extend([f'prev_result_med1_{i}' for i in range(self.number_of_rounds)])
        columns.extend([f'prev_result_high_{i}' for i in range(self.number_of_rounds)])
        columns.extend([f'prev_expected_dm_payoff_low_{i}' for i in range(self.number_of_rounds)])
        columns.extend([f'prev_expected_dm_payoff_high_{i}' for i in range(self.number_of_rounds)])
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

        file_name = f'all_data_{self.label}_label_seq_{self.features_file}'
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

        if self.use_seq:
            file_name = f'all_data_{self.label}_label_seq'
        elif self.use_prev_round:
            file_name = f'all_data_{self.label}_label_prev_round'
        else:
            file_name = f'all_data_{self.label}_label'
        self.final_data.to_csv(os.path.join(data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(data_directory, f'{file_name}.pkl'))

        print(f'Finish creating sequnces with different lengths and concat')
        logging.info('Finish creating sequnces with different lengths and concat')

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

            if self.use_seq:
                if self.use_manual_features:
                    file_name =\
                        f'{data_name}_data_1_{self.number_of_rounds}_{self.label}_label_seq_{self.features_file}'
                else:
                    file_name = f'{data_name}_data_1_{self.number_of_rounds}_{self.label}_label_seq'
            elif self.use_manual_features:
                if self.use_prev_round:
                    file_name =\
                        f'{data_name}_data_1_{self.number_of_rounds}_{self.label}_label_prev_{self.features_file}'
                else:
                    file_name = f'{data_name}_data_1_{self.number_of_rounds}_label_{self.features_file}'
            elif self.use_prev_round:
                file_name = f'{data_name}_data_1_{self.number_of_rounds}_{self.label}_label_prev_round'
            else:
                file_name = f'{data_name}_data_{self.label}_label'

            data.to_csv(os.path.join(data_directory, f'{file_name}.csv'), index=False)
            joblib.dump(data, os.path.join(data_directory, f'{file_name}.pkl'))
            # save 9 sequences per pair
            if self.use_seq and not self.label == 'single_round':
                seq_len_9_data = data.loc[data.k_size != self.number_of_rounds]
                columns_to_drop = [column for column in seq_len_9_data.columns if str(self.number_of_rounds - 1) in column]
                seq_len_9_data = seq_len_9_data.drop(columns_to_drop, axis=1)
                seq_len_9_data.to_csv(os.path.join(
                    data_directory, f'{data_name}_data_1_{self.number_of_rounds-1}_{self.label}_label.csv'),
                    index=False)
                joblib.dump(seq_len_9_data,
                            os.path.join(data_directory,
                                         f'{data_name}_data_1_{self.number_of_rounds-1}_{self.label}_label.pkl'))

        print(f'Finish split data to train-test-validation data and save for k=1-10 and k=1-9')
        logging.info('Finish split data to train-test-validation data and save for k=1-10 and k=1-9')


def main():
    features_files = {
        'manual_binary_features': 'xlsx',
        'bert_embedding': 'pkl',
    }
    features_to_use = 'bert_embedding'
    create_save_data_obj = CreateSaveData('results_payments_status', total_payoff_label=False, label='single_round',
                                          use_seq=False, use_prev_round=True, use_manual_features=True,
                                          features_file_type=features_files[features_to_use],
                                          features_file=features_to_use)
    # create_save_data_obj = CreateSaveData('results_payments_status', total_payoff_label=True)
    if create_save_data_obj.use_manual_features:
        if create_save_data_obj.use_seq:
            create_save_data_obj.create_manual_features_seq_data()
        else:
            create_save_data_obj.create_manual_features_data()
    else:
        create_save_data_obj.create_seq_data()
    create_save_data_obj.split_data()


if __name__ == '__main__':
    main()
