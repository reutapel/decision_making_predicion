import pandas as pd
import os
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import scipy.stats as stats
from matplotlib.font_manager import FontProperties
import math
from data_analysis import create_chart_bars
from sklearn.utils import resample
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
from collections import defaultdict


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'results')
data_analysis_directory = os.path.join(base_directory, 'analysis')

# dict for the directories file names. dir_name: [file name, if not_agent_data(2 players exp), number_of_trials,
# known_unknown_exp]
directory_files_dict = {
    'agent_mturk': ['all_data_after_4_1_and_data_to_use_before.xlsx', False, 50, False],
    'one_player': ['first_results.xlsx', False, 50, False],
    'agent_no_payment': ['agent_no_payment.xlsx', False, 50, False],
    'all': ['results_payments_status.csv', True, 50, False],
    'second_agent_exp': ['mturk_ag_second_exp.xlsx', False, 60, True],
    'second_agent_exp_p_win': ['mturk_ag_second_exp.xlsx', False, 60, True],
    'fix_prob': ['results_payments_status.csv', True, 50, False],
}
exp_directory = 'all'
fix_prob = True
not_agent_data = directory_files_dict[exp_directory][1]
num_trials = directory_files_dict[exp_directory][2]
known_unknown_exp = directory_files_dict[exp_directory][3]
appendix = 'group' if not_agent_data else 'player'
unknown_appendix_list = ['_u', '_k'] if known_unknown_exp else ['']

split_gender = False

data_analysis_directory = os.path.join(data_analysis_directory, exp_directory, 'agent_analysis') if not_agent_data \
    else os.path.join(data_analysis_directory, exp_directory)
data_analysis_directory = data_analysis_directory if not fix_prob else os.path.join(data_analysis_directory, 'fix_prob')
if split_gender:
    data_analysis_directory = os.path.join(data_analysis_directory, 'per_gender')
if not (os.path.exists(data_analysis_directory)):
    os.makedirs(data_analysis_directory)

log_file_name = os.path.join(data_analysis_directory, exp_directory,
                             datetime.now().strftime('LogFile_agent_data_analysis_%d_%m_%Y_%H_%M_%S.log'))

logging.basicConfig(filename=log_file_name,
                    level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    )

header_dict = {
    'expert_payoff_with_dm_timeout': 'if decision maker had timeout, agent payoff = 0.5',
    'expert_payoff_no_dm_timeouts': 'remove trials with decision maker timeout',
    'expert_normalized_payoff':
        "Normalized the agent payoff to the number of trials without timeouts "
        "(number_of_trials * (payff_no_timeouts_trials/(number_of_trials-number_of_timeout_trials))",
    'receiver_payoff_no_timeout': 'Decision maker payoff without trials with timeout',
    'receiver_total_payoff': 'Decision maker payoff for all trials',
    'percentage of decision maker': 'Percentage of decision maker that got gamble with positive/negative EV/average '
                                    'and decided to entered or not to entered',
    'Analyze gamble average': "Average and STD of the gambles' average values in all the trials"
}


def create_point_plot_per_user(points_x, points_y, title, xlabel, ylabel, add_line_points=True, inner_dir='', step=5,
                               colors_groups=None, legend=None, percentage_graph=True):
    """
    This function create a plot using lists of points
    :param list(list) points_x: list of lists. each list is a list of the x value of the points to add to the plot
    :param list(list) points_y: list of lists. each list is a list of the y value of the points to add to the plot
    :param str title: the name of the plot
    :param str xlabel: the label of the x axis
    :param str ylabel: the label of the y axis
    :param bool add_line_points: whether to add line between the points or not
    :param str inner_dir: if we want to save the plots in an inner directory
    :param int step: the step size in X axis
    :param list colors_groups: each item is a list of the group the x,y belong to,
    this will be the color
    :param list legend: list of legend to add
    :param bool percentage_graph: if this is a graph for %
    :return:
    """

    print('Create point plot for', title)

    fig, ax = plt.subplots()
    x_max, y_max = 0, 0
    x_min, y_min = 1, 10
    for lst in points_x:
        if max(lst) > x_max:
            x_max = max(lst)
        if min(lst) < x_min:
            x_min = min(lst)

    for lst in points_y:
        if max(lst) > y_max:
            y_max = max(lst)
        if min(lst) < y_min:
            y_min = min(lst)

    gap = 0.1
    ax.axis([x_min - gap, x_max + gap, y_min - gap, y_max + gap])

    if percentage_graph:
        plt.ylim(0, 100)

    markers = [".", "x", "+", "1"]
    colors = ['orange', 'seagreen', 'navy', 'crimson']

    for x, y, color in zip(points_x, points_y, colors_groups):
        ax.scatter(x, y, color=colors[color], marker=markers[color])
        if add_line_points:
            ax.plot(x, y, color=colors[color])

    # add some text for labels, title and axes ticks
    patches = [mpatches.Patch(color=colors[i], label=legend[i]) for i in range(len(legend))]
    plt.legend(handles=patches)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    number_of_unique_points = len(set([item for x_list in points_x for item in x_list]))
    all_max = max([max(x_list) for x_list in points_x if len(x_list) > 0])
    all_min = min([min(x_list) for x_list in points_x if len(x_list) > 0])
    if number_of_unique_points <= 20:
        if len(points_x) > 1:
            lst_xticks = np.append(points_x[0], points_x[1])
        else:
            lst_xticks = np.array(points_x[0])
    elif all_max % 5 == 0:
        lst_xticks = np.arange(all_min, all_max + 1, step=step)
    else:
        lst_xticks = np.arange(all_min, all_max, step=step)
        lst_xticks = np.append(lst_xticks, all_max)

    lst_xticks = lst_xticks.astype(int)
    plt.xticks(lst_xticks)

    if not os.path.exists(os.path.join(data_analysis_directory, inner_dir)):
        os.makedirs(os.path.join(data_analysis_directory, inner_dir))

    fig_to_save = fig
    fig_to_save.savefig(os.path.join(data_analysis_directory, inner_dir, title + '.png'),
                        bbox_inches='tight')

    return


def define_receiver_payoff(row):
    if pd.isnull(row['player_receiver_payoff']):
        if row[appendix + '_receiver_choice'] == 0:  # no payoff and chose lottery
            return row[appendix + '_lottery_result']
        else:
            return 0
    else:
        return row['player_receiver_payoff']  # we have payoff


def bootstrap_test(vec1, vec2, num_bootstrap, vec3=None):
    """
    This function draw a sample with replacement with the chosen size, calculate the average on the sample,
    calculate the mean of the calculated sample statistics and the t-test over the new 2 vectors of averages.
    :param vec1: first vector of results
    :param vec2: second vector of results
    :param int num_bootstrap: the number of times to run the bootstrap
    :param vec3: if there is another data set
    :return:
    """
    vec1_average_list = list()
    vec2_average_list = list()
    vec3_average_list = list()

    for i in range(num_bootstrap):
        sample_vec1 = resample(vec1, replace=True, n_samples=vec1.shape[0])
        sample_vec2 = resample(vec2, replace=True, n_samples=vec2.shape[0])
        if vec3 is not None:
            sample_vec3 = resample(vec3, replace=True, n_samples=vec2.shape[0])
            vec3_average_list.append(sample_vec3.mean())

        vec1_average_list.append(sample_vec1.mean())
        vec2_average_list.append(sample_vec2.mean())

    vec1_average = sum(vec1_average_list)/len(vec1_average_list)
    vec2_average = sum(vec2_average_list)/len(vec2_average_list)
    if vec3 is not None:
        vec3_average = sum(vec3_average_list) / len(vec3_average_list)
    else:
        vec3_average = None

    bootstrap_t_value_vec1_vec2, bootstrap_p_value_vec1_vec2 = stats.ttest_ind(vec1_average_list, vec2_average_list)
    if vec3 is not None:
        bootstrap_t_value_vec1_vec3, bootstrap_p_value_vec1_vec3 = stats.ttest_ind(vec1_average_list, vec3_average_list)
        bootstrap_t_value_vec2_vec3, bootstrap_p_value_vec2_vec3 = stats.ttest_ind(vec2_average_list, vec3_average_list)
    else:
        bootstrap_t_value_vec1_vec3, bootstrap_p_value_vec1_vec3, bootstrap_t_value_vec2_vec3,\
            bootstrap_p_value_vec2_vec3 = None, None, None, None

    return [vec1_average, vec2_average, vec3_average, bootstrap_t_value_vec1_vec2, bootstrap_p_value_vec1_vec2,
            bootstrap_t_value_vec1_vec3, bootstrap_p_value_vec1_vec3,
            bootstrap_t_value_vec2_vec3, bootstrap_p_value_vec2_vec3]


def shapiro_wilk_test(data_a, data_b, data_c=None):
    """
    Shapiro-Wilk: Perform the Shapiro-Wilk test for normality.
    :param data_a: first data
    :param data_b: second data
    :param data_c: third data- option
    :return:
    """
    # Shapiro-Wilk: Perform the Shapiro-Wilk test for normality.
    shapiro_results_a = stats.shapiro(data_a)
    shapiro_results_b = stats.shapiro(data_b)
    if data_c is not None:
        shapiro_results_c = stats.shapiro(data_c)
    else:
        shapiro_results_c = [None, None]

    return [shapiro_results_a[1], shapiro_results_b[1], shapiro_results_c[1]]


class DataAnalysis:
    def __init__(self, class_gender=None):
        self.gender = class_gender if class_gender is not None else 'all_genders'
        print('Start running data analysis on data folder', exp_directory)
        logging.info('Start running data analysis on data folder {}'.format(exp_directory))
        columns_to_use = ['participant_code', 'participant__current_page_name', 'participant_visited',
                          'participant_mturk_worker_id', 'participant_mturk_assignment_id', 'participant_payoff',
                          'player_id_in_group', 'player_name', 'player_age', 'player_gender', 'player_is_student',
                          'player_occupation', 'player_residence', 'player_payoff', 'group_id_in_subsession',
                          'player_sender_answer', 'player_receiver_choice', 'player_lottery_result',
                          'player_receiver_timeout', 'player_sender_payoff',
                          'player_x_lottery', 'player_y_lottery', 'player_p_lottery', 'player_ev_lottery',
                          'subsession_round_number', 'session_code', 'player_expert_type', 'player_receiver_payoff']
        if directory_files_dict[exp_directory][1]:  # if analyze not agent experiment - use results_payments_status
            self.results = pd.read_csv(os.path.join(base_directory, 'analysis', exp_directory,
                                                    directory_files_dict[exp_directory][0]))
            if fix_prob:
                self.result = self.results.loc[self.results.participant_current_app_name == 'first_fix_prob_exp']
            # for not agent experiment - keep only the decision makers
            self.results = self.results.loc[self.results.player_id_in_group == 2]
            self.results = self.results.loc[self.results.pct_timeout <= 50]
            self.results['player_expert_type'] = np.where(pd.isnull(self.results.player_expert_type), '',
                                                          self.results.player_expert_type)
            self.results['player_receiver_payoff'] = np.nan
            self.results['group_ev_lottery'] = self.results.group_p_lottery * self.results.group_x_lottery +\
                                               self.results.group_y_lottery * (1 - self.results.group_p_lottery)
        elif not known_unknown_exp:
            self.results = pd.read_excel(os.path.join(data_directory, exp_directory,
                                                      directory_files_dict[exp_directory][0]), usecols=columns_to_use,
                                         sheetname='data_to_use')
        else:
            self.results = pd.read_excel(os.path.join(data_directory, exp_directory,
                                                      directory_files_dict[exp_directory][0]), sheetname='data_to_use')

        if split_gender:
            gender_participants = self.results.loc[self.results['player_gender'] == self.gender].participant_code
            self.results = self.results.loc[self.results.participant_code.isin(gender_participants)]
        # keep only data to use
        self.results = self.results.loc[self.results.participant_visited == 1]
        # self.time_spent = pd.read_csv(os.path.join(data_directory, date_directory, 'TimeSpent_first_results.csv'))
        payment_file_path = os.path.join(data_directory, exp_directory, 'Approved assignments.xlsx')
        if os.path.isfile(payment_file_path):
            self.payments = pd.read_excel(payment_file_path)
            # keep only the participants got paid
            self.results_payments =\
                self.results.merge(self.payments, left_on=['participant_mturk_worker_id',
                                                           'participant_mturk_assignment_id', 'participant_code'],
                                   right_on=['worker_id', 'assignment_id', 'participant_code'], how='left')
        else:  # for no payment participants
            self.results_payments = self.results

        # adding pair_id
        self.personal_info = self.results.loc[self.results.player_age.notnull(),
                                              ['participant_code', 'participant_mturk_worker_id', 'player_residence',
                                               'participant_mturk_assignment_id', 'player_name', 'player_age',
                                               'player_gender', 'player_is_student', 'player_occupation']]

        payoff_writer_file_name = (os.path.join(data_analysis_directory,
                                                'payoff_comparison_' + exp_directory + '.xlsx'))
        if os.path.exists(payoff_writer_file_name):
            os.remove(payoff_writer_file_name)
        self.payoff_writer = pd.ExcelWriter(os.path.join(data_analysis_directory,
                                                         payoff_writer_file_name), engine='xlsxwriter')

    def get_payoff_measure_per_agent_type(self, column_name, orig_data=None):
        """
        This function create a vector of all participants payoff per agent type
        :param str column_name: the column name to calculate the group by
        :param pd.DataFrame orig_data: data to use, is none- use self.results_payments
        :return:
        """
        agents = self.results_payments.player_expert_type.unique().tolist()
        agents = [agent for agent in agents if 'ten' not in agent]
        max_num_participants = 0
        for agent_type in agents:
            num_par = self.results_payments.loc[
                (self.results_payments.player_expert_type == agent_type)].drop_duplicates('participant_code').shape[0]
            if num_par > max_num_participants:
                max_num_participants = num_par

        group_by = pd.DataFrame(index=range(max_num_participants))
        number_of_agents = 0
        for agent_type in agents:
            if orig_data is None:
                data = self.results_payments.loc[(self.results_payments.player_expert_type == agent_type)]
            else:
                data = orig_data.loc[(orig_data.player_expert_type == agent_type)]

            if data.empty:
                continue
            else:
                number_of_agents += 1

            data = data.drop_duplicates('participant_code').reset_index()
            data = data[column_name]

            group_by[agent_type] = data

        group_by_df_measures = pd.DataFrame(columns=agents)
        group_by_df_measures.loc['average'] = group_by.mean()
        group_by_df_measures.loc['median'] = group_by.median()
        group_by_df_measures.loc['STD'] = group_by.std()
        group_by_df_measures.loc['max'] = group_by.max()
        group_by_df_measures.loc['min'] = group_by.min()
        group_by_df_measures.loc['num objects'] = group_by.count()

        if number_of_agents > 1:
            # calculate t-test
            agent1 = group_by[agents[0]].dropna()
            agent1_values = agent1.values

            agent2 = group_by[agents[1]].dropna()
            agent2_values = agent2.values

            if number_of_agents > 2:
                agent3 = group_by[agents[2]].dropna()
                agent3_values = agent3.values
            else:
                agent3 = None
                agent3_values = None

            bootstrap_test_results = bootstrap_test(agent1, agent2, 100, agent3)

            group_by_df_measures.loc['bootstrap average', agents[0]] = bootstrap_test_results[0]
            group_by_df_measures.loc['bootstrap average', agents[1]] = bootstrap_test_results[1]
            if agent3 is not None:
                group_by_df_measures.loc['bootstrap average', agents[2]] = bootstrap_test_results[2]

            group_by_df_measures.loc[group_by_df_measures.index[0], 't value ' + agents[0] + ' and ' + agents[1]],\
                group_by_df_measures.loc[group_by_df_measures.index[0], 'p value ' + agents[0] + ' and ' + agents[1]] =\
                stats.ttest_ind(agent1.values, agent2.values)

            shapiro_wilk_test_results = shapiro_wilk_test(agent1_values, agent2_values, agent3_values)
            group_by_df_measures.loc['shapiro-wilk p value', agents[0]] = shapiro_wilk_test_results[0]
            group_by_df_measures.loc['shapiro-wilk p value', agents[1]] = shapiro_wilk_test_results[1]
            if agent3 is not None:
                group_by_df_measures.loc['shapiro-wilk p value', agents[2]] = shapiro_wilk_test_results[2]

            # bootstrap agents 0 and 1
            group_by_df_measures.loc[group_by_df_measures.index[0],
                                     'bootstrap t value ' + agents[0] + ' and ' + agents[1]] = bootstrap_test_results[3]
            group_by_df_measures.loc[group_by_df_measures.index[0],
                                     'bootstrap p value ' + agents[0] + ' and ' + agents[1]] = bootstrap_test_results[4]

            if agent3 is not None:
                # bootstrap agents 0 and 2
                group_by_df_measures.loc[group_by_df_measures.index[0],
                                         'bootstrap t value ' + agents[0] + ' and ' + agents[2]] =\
                    bootstrap_test_results[5]
                group_by_df_measures.loc[group_by_df_measures.index[0],
                                         'bootstrap p value ' + agents[0] + ' and ' + agents[2]] =\
                    bootstrap_test_results[6]

                # bootstrap agents 1 and 2
                group_by_df_measures.loc[group_by_df_measures.index[0],
                                         'bootstrap t value ' + agents[1] + ' and ' + agents[2]] =\
                    bootstrap_test_results[7]
                group_by_df_measures.loc[group_by_df_measures.index[0],
                                         'bootstrap p value ' + agents[1] + ' and ' + agents[2]] =\
                    bootstrap_test_results[8]

        group_by_df_measures = group_by_df_measures.append(group_by)[group_by_df_measures.columns.tolist()]
        group_by_df_measures[agents] = group_by_df_measures[agents].round(2)

        return group_by_df_measures

    def agent_payoff_analysis(self):
        """
        This function analysis the agent payoff
        :return:
        """
        columns_groupby_creation = ['expert_payoff_with_dm_timeout', 'expert_payoff_no_dm_timeouts',
                                    'expert_normalized_payoff']
        # assign expert payoff to be 0.5 in case of receiver timeout
        self.results_payments['with_dm_timeout_payoff'] =\
            np.where(self.results_payments[appendix + '_receiver_timeout'] == 1, 0.5,
                     self.results_payments[appendix + '_sender_payoff'])
        with_dm_timeout_expert_payoff =\
            pd.DataFrame(self.results_payments.groupby(by='participant_code').with_dm_timeout_payoff.sum())
        with_dm_timeout_expert_payoff.columns = ['expert_payoff_with_dm_timeout']

        self.results_payments = self.results_payments.merge(with_dm_timeout_expert_payoff, left_on='participant_code',
                                                            right_index=True, how='left')

        # normalized the payoff according to the number of timeouts
        # get the payoff without the timeouts
        no_timeouts = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        no_timeouts = pd.DataFrame(no_timeouts.groupby(by='participant_code')[appendix + '_sender_payoff'].sum())
        no_timeouts.columns = ['expert_payoff_no_dm_timeouts']

        self.results_payments = self.results_payments.merge(no_timeouts, left_on='participant_code', right_index=True,
                                                            how='left')

        # get the number of timeouts per participant
        num_timeouts = pd.DataFrame(self.results_payments.groupby(by='participant_code')[
                                        appendix + '_receiver_timeout'].sum())
        num_timeouts.columns = ['dm_num_timeouts']

        self.results_payments = self.results_payments.merge(num_timeouts, left_on='participant_code', right_index=True,
                                                            how='left')

        self.results_payments['expert_normalized_payoff'] =\
            num_trials * self.results_payments.expert_payoff_no_dm_timeouts/\
            (num_trials - self.results_payments.dm_num_timeouts)

        for column in columns_groupby_creation:
            payoff = self.get_payoff_measure_per_agent_type(column)
            pivot_header = pd.DataFrame(columns=[header_dict[column]])
            pivot_header.to_excel(self.payoff_writer, sheet_name=column, startrow=0, startcol=0)
            payoff.to_excel(self.payoff_writer, sheet_name=column, startrow=1, startcol=0)

    def dm_payoff(self):
        """
        This function calculate the DM payoff per agent
        :return:
        """
        self.results_payments['player_receiver_payoff'] = self.results_payments.apply(define_receiver_payoff, axis=1)
        receiver_total_payoff = pd.DataFrame(self.results_payments.groupby(by=['participant_code']).
                                             player_receiver_payoff.sum())
        receiver_total_payoff.columns = ['receiver_total_payoff']
        receiver_total_payoff = receiver_total_payoff.merge(
            self.results_payments[['participant_code', 'player_expert_type']], left_index=True,
            right_on='participant_code', how='left')
        receiver_total_payoff = receiver_total_payoff.drop_duplicates('participant_code')

        no_timeout = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        no_timeout = pd.DataFrame(no_timeout.groupby(by=['participant_code']).player_receiver_payoff.sum())
        no_timeout.columns = ['receiver_payoff_no_timeout']

        receiver_total_payoff = receiver_total_payoff.merge(no_timeout, left_on='participant_code', right_index=True,
                                                            how='left')

        columns_groupby_creation = ['receiver_payoff_no_timeout', 'receiver_total_payoff']
        for column in columns_groupby_creation:
            payoff = self.get_payoff_measure_per_agent_type(column, receiver_total_payoff)
            pivot_header = pd.DataFrame(columns=[header_dict[column]])
            pivot_header.to_excel(self.payoff_writer, sheet_name=column, startrow=0, startcol=0)
            payoff.to_excel(self.payoff_writer, sheet_name=column, startrow=1, startcol=0)

        return

    def create_analysis_columns(self):
        """
        This function create columns for advanced analysis: positive_ev, positive_average, prob_higher_half
        :return:
        """

        for unknown_appendix in unknown_appendix_list:
            self.results_payments['positive_average' + unknown_appendix] =\
                np.where(0.5*(self.results_payments[appendix + '_x' + unknown_appendix + '_lottery'] +
                              self.results_payments[appendix + '_y' + unknown_appendix + '_lottery']) >= 0, 1, 0)
            self.results_payments['prob_higher_half' + unknown_appendix] =\
                np.where(self.results_payments[appendix + '_sender_answer'] >= 0.5, 1, 0)
            if unknown_appendix == '_k':
                # average for the known option is for x,y,z
                self.results_payments['average' + unknown_appendix] = \
                    self.results_payments[[appendix + '_x' + unknown_appendix + '_lottery',
                                           appendix + '_y' + unknown_appendix + '_lottery',
                                           appendix + '_z' + unknown_appendix + '_lottery']].mean(axis=1)
            else:
                self.results_payments['average' + unknown_appendix] =\
                    self.results_payments[[appendix + '_x' + unknown_appendix + '_lottery',
                                           appendix + '_y' + unknown_appendix + '_lottery']].mean(axis=1)
            self.results_payments['X-Y' + unknown_appendix] =\
                self.results_payments[appendix + '_x' + unknown_appendix + '_lottery'] -\
                abs(self.results_payments[appendix + '_y' + unknown_appendix + '_lottery'])
            # with average = 1 if DM choose lottery (player_sender_payoff == 1) and positive_average == 1 or
            # the DM choose status quo (player_sender_payoff == 0) and positive_average == 0
            # for the known option, the with_average = 1 if (player_sender_payoff == 0) and positive_average == 1
            self.results_payments['with_average' + unknown_appendix] = np.where(
                (self.results_payments['positive_average' + unknown_appendix] ==
                 self.results_payments[appendix + '_sender_payoff'] * 0 if unknown_appendix == '_k' else 1), 1, 0)

        # run only once for all experiments
        self.results_payments['positive_ev' + unknown_appendix_list[0]] = \
            np.where(self.results_payments[appendix + '_ev' + unknown_appendix_list[0] + '_lottery'] >= 0, 1, 0)
        # define if the expert recommended to take the lottery. for zero-one: answer=1,
        # for accurate: the EV based on its probability is positive
        self.results_payments['expert_EV'] = self.results[appendix + '_sender_answer'] *\
                                             self.results[appendix + '_x' + unknown_appendix_list[0] + '_lottery'] +\
                                             self.results[appendix + '_y' + unknown_appendix_list[0] + '_lottery'] *\
                                             (1 - self.results[appendix + '_sender_answer'])

        self.results_payments['expert_recommended'] = np.where(self.results_payments.expert_EV > 0, 1, 0)
        # define expert_recommended for p_win_agent --> when say 0.9- recommend, otherwise don't
        self.results_payments.loc[(self.results_payments[f'player_expert_type'] == 'p_win_agent') &
                                  (self.results_payments[f'{appendix}_sender_answer'] == 0.9), 'expert_recommended'] = 1
        self.results_payments.loc[(self.results_payments[f'player_expert_type'] == 'p_win_agent') &
                                  (self.results_payments[f'{appendix}_sender_answer'] < 0.9), 'expert_recommended'] = 0

        # with expert = 1:
        # if the DM choose lottery (player_sender_payoff == 1) and expert_recommended == 1 or
        # the DM choose status quo (player_sender_payoff == 0) and expert_recommended == 0
        self.results_payments['with_expert'] = np.where(
            (self.results_payments.expert_recommended == self.results_payments[appendix + '_sender_payoff']), 1, 0)

        if known_unknown_exp:
            self.results_payments['diff_avg_k_avg_u'] = self.results_payments.average_k - self.results_payments.average_u
            self.results_payments['diff_x_u_x_k'] = self.results_payments[appendix + '_x_u_lottery'] -\
                                                    self.results_payments[appendix + '_x_k_lottery']
            self.results_payments['diff_x_u_y_k'] = self.results_payments[appendix + '_x_u_lottery'] -\
                                                    self.results_payments[appendix + '_y_k_lottery']
            self.results_payments['diff_y_u_z_k'] = self.results_payments[appendix + '_y_u_lottery'] -\
                                                    self.results_payments[appendix + '_z_k_lottery']

        return

    def analyze_dm_entrances(self):
        """
        Analyze when the decision maker chose the lottery and not the status quo
        :return:
        """
        pivot = pd.pivot_table(self.results_payments, values=appendix + '_sender_payoff',
                               index=['positive_ev' + unknown_appendix_list[0]],
                               columns=['positive_average' + unknown_appendix_list[0], appendix + '_receiver_choice'],
                               aggfunc=len)
        if 1 in pivot.index:
            pivot.index = ['negative ev', 'positive ev']
        else:
            pivot.index = ['negative ev']
        pivot.columns.levels = [['negative average', 'positive average'], ['dm chose unknown', 'dm chose known']]

        pivot.to_csv(os.path.join(data_analysis_directory, 'dm_entrances_analysis.csv'))

        return

    def analyze_dm_choices_per_round(self):
        """
        This function analyze the decision maker choices through the rounds
        :return:
        """

        # if key[1] == True: check when the DM entered (sum of player_sender_payoff),
        # if false: when DM didn't enter (sum of player_receiver_choice - 1 when choose status quo)
        # key[2]: if positive is first in the list of positive and negative average
        # values are: experts list to check, list of queries based on the expert type
        colors_lists = {'negative_first': ['purple', 'skyblue'],
                        'positive_first': ['skyblue', 'purple']}
        expert_legend_dict = {'zero_one': ['p=1', 'p=0'],
                              'accurate': ['expert EV positive', 'expert EV negative'],
                              'p_win': ['unknown win', 'known unknown equal']
                              }
        all_groups = defaultdict(list)
        all_expert_type = {expert: [expert] for expert in self.results_payments.player_expert_type.unique()}
        zero_one_experts = [expert for expert in self.results_payments.player_expert_type.unique()
                            if 'zero_one' in expert and 'not' not in expert]
        if len(zero_one_experts) > 1:
            all_expert_type['all_zero_one'] = zero_one_experts
        accurate_experts = [expert for expert in self.results_payments.player_expert_type.unique()
                            if 'accurate' in expert]
        if len(accurate_experts) > 1:
            all_expert_type['all_accurate'] = accurate_experts

        for expert_name, expert_list in all_expert_type.items():
            if 'zero_one' in expert_name:
                expert_legend = expert_legend_dict['zero_one']
            elif 'p_win' in expert_name:
                expert_legend = expert_legend_dict['p_win']
            else:
                expert_legend = expert_legend_dict['accurate']

            all_groups[(f'% of decision maker that entered when {expert_name} agent recommended to enter',
                        True, 'positive_first')] = \
                [expert_list, [['expert_recommended == 1', expert_legend[0]],
                               ['positive_average' + unknown_appendix_list[0] + ' == 1', 'positive average'],
                               ['positive_average' + unknown_appendix_list[0] + ' == 0', 'negative average']]]

            all_groups[(f"% of decision maker that didn't enter when {expert_name} agent recommended to enter",
                        False, 'negative_first')] = \
                [expert_list, [['expert_recommended == 1', expert_legend[0]],
                               ['positive_average' + unknown_appendix_list[0] + ' == 0', 'negative average'],
                               ['positive_average' + unknown_appendix_list[0] + ' == 1', 'positive average']]]

            all_groups[(f'% of decision maker that entered when {expert_name} agent recommended not to enter',
                        True, 'positive_first')] = \
                [expert_list, [['expert_recommended == 0', expert_legend[1]],
                               ['positive_average' + unknown_appendix_list[0] + ' == 1', 'positive average'],
                               ['positive_average' + unknown_appendix_list[0] + ' == 0', 'negative average']]]

            all_groups[(f"% of decision maker that didn't enter when {expert_name} agent recommended not to enter",
                        False, 'negative_first')] = \
                [expert_list, [['expert_recommended == 0', expert_legend[1]],
                               ['positive_average' + unknown_appendix_list[0] + ' == 0', 'negative average'],
                               ['positive_average' + unknown_appendix_list[0] + ' == 1', 'positive average']]]

        data_to_use = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        for group_name, groups in all_groups.items():
            expert_type = groups[0]
            query_list = groups[1]
            data = data_to_use.loc[data_to_use.player_expert_type.isin(expert_type)]
            if data.empty:
                continue
            for split_average in [False, True]:
                colors = colors_lists[group_name[2]] if split_average else None
                legend = list()
                x_points = list()
                y_points = list()
                for query_index, queries in enumerate(query_list):
                    curr_data = data.copy()
                    if split_average:
                        if query_index == 0:  # for the first query in split average - don't create data for it
                            continue
                        else:  # if this is not the first query: get specific data to work on based on the 1st query
                            curr_data = curr_data.query(query_list[0][0]).copy()
                    else:  # if not split average- don't use the last query
                        if query_index == len(query_list)-1:
                            continue

                    if curr_data.empty:
                        continue

                    curr_data = curr_data.query(queries[0])
                    legend.append(queries[1])
                    count = pd.DataFrame(curr_data.groupby('subsession_round_number').participant_code.count())
                    if group_name[1]:
                        sum_participant =\
                            pd.DataFrame(curr_data.groupby('subsession_round_number')[
                                             appendix + '_sender_payoff'].sum())
                        pct = count.merge(sum_participant, left_index=True, right_index=True)
                        pct['pct'] = 100 * pct[appendix + '_sender_payoff'] / pct.participant_code
                    else:
                        sum_participant =\
                            pd.DataFrame(curr_data.groupby('subsession_round_number')[
                                             appendix + '_receiver_choice'].sum())
                        pct = count.merge(sum_participant, left_index=True, right_index=True)
                        pct['pct'] = 100 * pct[appendix + '_receiver_choice'] / pct.participant_code

                    x_points.append(pct.index.values)
                    y_points.append(pct['pct'].tolist())

                create_chart_bars(title=f'{group_name[0]} {self.gender} (split_average {str(split_average)})',
                                  x=x_points, y=y_points, xlabel='trial number', ylabel='percentage of participants',
                                  legend=legend, curr_colors=colors, percentage_graph=True,
                                  curr_date_directory=os.path.join(data_analysis_directory,
                                                                   'analyze_dm_choices_per_round'))

        return

    def create_2_2_tables(self):
        """
        This function create 2X2 tables for each agent: DM entered/didn't enter and positive/negative EV.
        Each table will show different measures
        :return:
        """
        payoff_writer_file_name = (os.path.join(data_analysis_directory,
                                                '2_2_tables_' + exp_directory + '.xlsx'))
        if os.path.exists(payoff_writer_file_name):
            os.remove(payoff_writer_file_name)
        table_writer = pd.ExcelWriter(payoff_writer_file_name, engine='xlsxwriter')

        workbook = table_writer.book
        worksheet = workbook.add_worksheet('percentage of decision maker')
        table_writer.sheets['percentage of decision maker'] = worksheet

        worksheet_header = pd.DataFrame(columns=[header_dict['percentage of decision maker']])
        worksheet_header.to_excel(table_writer, sheet_name='percentage of decision maker', startrow=0, startcol=0)

        workbook = table_writer.book
        worksheet = workbook.add_worksheet('Analyze gamble average')
        table_writer.sheets['Analyze gamble average'] = worksheet

        worksheet_header = pd.DataFrame(columns=[header_dict['Analyze gamble average']])
        worksheet_header.to_excel(table_writer, sheet_name='Analyze gamble average', startrow=0, startcol=0)

        data_to_analyze = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        columns_to_analyze = ['positive_ev' + unknown_appendix_list[0], 'positive_average' + unknown_appendix_list[0]]
        agents_list = self.results_payments.player_expert_type.unique().tolist()
        agents_list = [agent for agent in agents_list if 'ten' not in agent]
        agents = list()
        for agent in agents_list:
            agents.append([agent, columns_to_analyze])

        writing_counter = 0
        for agent, columns in agents:
            agent_data = data_to_analyze.loc[data_to_analyze.player_expert_type == agent]
            for column in columns:
                group_by_ev = agent_data.groupby(by=column).participant_code.count()
                group_by_both = agent_data.groupby(by=[column, appendix + '_receiver_choice']).participant_code.count()
                merge_group_by = pd.merge(group_by_ev.reset_index(), group_by_both.reset_index(), on=column)
                merge_group_by['percentage_of_decision_maker'] = 100 * merge_group_by.participant_code_y /\
                                                                 merge_group_by.participant_code_x
                pivot = pd.pivot_table(merge_group_by, values='percentage_of_decision_maker', index=column,
                                       columns=appendix + '_receiver_choice', aggfunc=sum)
                if 1 in pivot.index:
                    pivot.index = ['negative ' + column.split('_')[1], 'positive ' + column.split('_')[1]]
                else:
                    pivot.index = ['negative ' + column.split('_')[1]]
                pivot.columns = ['% DM chose unknown', '% DM chose known']
                pivot.columns.name = agent
                pivot = pivot.round(2)
                pivot_header = pd.DataFrame(columns=[agent + ' analyze ' + column.split('_')[1]])
                pivot_header.to_excel(table_writer, sheet_name='percentage of decision maker',
                                      startrow=1+(writing_counter*5), startcol=0)
                pivot.to_excel(table_writer, sheet_name='percentage of decision maker',
                               startrow=1+(writing_counter*5)+1, startcol=0)
                writing_counter += 1

        # second 2X2 table
        writing_counter = 0
        for agent, columns in agents:
            agent_data = data_to_analyze.loc[data_to_analyze.player_expert_type == agent]
            group_by_mean = agent_data.groupby(
                by=['positive_ev' + unknown_appendix_list[0], appendix + '_receiver_choice'])[
                'average' + unknown_appendix_list[0]].mean()
            group_by_std = agent_data.groupby(
                by=['positive_ev' + unknown_appendix_list[0], appendix + '_receiver_choice'])[
                'average' + unknown_appendix_list[0]].std()

            merge_group_by = pd.merge(group_by_mean.reset_index(), group_by_std.reset_index(),
                                      on=['positive_ev' + unknown_appendix_list[0], appendix + '_receiver_choice'])
            merge_group_by.columns = ['positive_ev' + unknown_appendix_list[0],
                                      appendix + '_receiver_choice', 'average_mean', 'average_std']
            pivot = pd.pivot_table(merge_group_by, values=['average_mean', 'average_std'],
                                   index='positive_ev' + unknown_appendix_list[0],
                                   columns=appendix + '_receiver_choice', aggfunc=sum)

            pivot.columns = pivot.columns.set_levels(
                [['average_mean', 'average_std'], ['DM chose unknown', 'DM chose known']])
            if 1 in pivot.index:
                pivot.index = ['negative EV', 'positive EV']
            else:
                pivot.index = ['negative EV']

            pivot = pivot.round(2)
            pivot_header = pd.DataFrame(columns=[agent + ' analyze gamble average'])
            pivot_header.to_excel(table_writer, sheet_name='Analyze gamble average',
                                  startrow=1+(writing_counter*7), startcol=0)
            pivot.to_excel(table_writer, sheet_name='Analyze gamble average',
                           startrow=1+(writing_counter*7)+1, startcol=0)
            writing_counter += 1

        table_writer.save()
        return

    def average_ev_gap_compare(self):
        """
        Create graph that will compare the % of DMs that chose the gamble vs the gamble average,
        EV and the gap between X and Y
        :return:
        """
        step = 15 if not_agent_data else 1
        legend = ['average' + unknown_appendix_list[0], appendix + '_ev' + unknown_appendix_list[0] + '_lottery',
                  'X-Y' + unknown_appendix_list[0]]
        agents_list = self.results_payments.player_expert_type.unique().tolist()
        agents_list = [agent for agent in agents_list if 'ten' not in agent]

        all_expert_type = {expert: [expert] for expert in agents_list}
        zero_one_experts = [expert for expert in self.results_payments.player_expert_type.unique()
                            if 'zero_one' in expert and 'not' not in expert]
        if len(zero_one_experts) > 1:
            all_expert_type['all_zero_one'] = zero_one_experts
        accurate_experts = [expert for expert in self.results_payments.player_expert_type.unique()
                            if 'accurate' in expert]
        if len(accurate_experts) > 1:
            all_expert_type['all_accurate'] = accurate_experts

        data_to_use = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        for y_axis, description in {appendix + '_sender_payoff': 'chose unknown',
                                    'with_expert': 'chose with expert recommendation'}.items():
            for agent, agent_list in all_expert_type.items():
                agent_data = data_to_use.loc[data_to_use.player_expert_type.isin(agent_list)]
                for column in legend:
                    x_points = list()
                    y_points = list()
                    groupby = agent_data.groupby(by=column).agg({'participant_code': 'count',
                                                                 y_axis: 'sum'})
                    groupby['% DM chose unknown'] = 100 * groupby[y_axis] / groupby.participant_code

                    x_points.append(groupby.index.values)
                    y_points.append(groupby['% DM chose unknown'].tolist())

                    create_chart_bars(title=f'% of decision makers {description} vs the {column} of the unknown gamble '
                                            f'for agent {agent} and {self.gender}',
                                      x=x_points, y=y_points, xlabel=column,
                                      ylabel=f'percentage of participants {description}', legend=legend,
                                      curr_date_directory=os.path.join(data_analysis_directory, 'average_ev_gap_compare'),
                                      percentage_graph=True, step=step)

        if known_unknown_exp:
            legend = ['average' + unknown_appendix_list[1], 'diff_avg_k_avg_u',
                      'diff_x_u_x_k', 'diff_x_u_y_k', 'diff_y_u_z_k']
            for y_axis, description in {appendix + '_sender_payoff': 'chose unknown',
                                        'with_expert': 'chose with expert recommendation'}.items():
                for agent in agents_list:
                    agent_data = data_to_use.loc[data_to_use.player_expert_type == agent]
                    for column in legend:
                        x_points = list()
                        y_points = list()
                        groupby = agent_data.groupby(by=column).agg({'participant_code': 'count',
                                                                     y_axis: 'sum'})
                        groupby['% DM chose gamble'] = 100 * groupby[y_axis] / groupby.participant_code

                        x_points.append(groupby.index.values)
                        y_points.append(groupby['% DM chose gamble'].tolist())

                        create_chart_bars(title=f'% of decision makers {description} vs the {column} of the known '
                                                f'gamble for agent {agent} and {self.gender}',
                                          x=x_points, y=y_points, xlabel=column,
                                          ylabel=f'percentage of participants {description}', legend=legend,
                                          curr_date_directory=os.path.join(data_analysis_directory,
                                                                           'average_ev_gap_compare'),
                                          percentage_graph=True, step=step)

        return

    def pct_decisions_with_expert(self):
        """
        This function analyze the % of times the DMs went with the expert recommendations vs didn't go with it
        :return:
        """

        data_to_use = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        agents_list = self.results_payments.player_expert_type.unique().tolist()
        agents_list = [agent for agent in agents_list if 'ten' not in agent]
        for agent in agents_list:
            agent_data = data_to_use.loc[data_to_use.player_expert_type == agent]
            final_data = pd.DataFrame(index=agent_data.participant_code.unique())
            if num_trials == 50:
                rounds_list = [[list(range(1, 16)), '1-15'], [list(range(16, 36)), '16-35'],
                               [list(range(36, 51)), '36-50']]
            else:
                rounds_list = [[list(range(1, 21)), '1-20'], [list(range(21, 41)), '21-40'],
                               [list(range(41, 61)), '41-60']]
            for rounds in rounds_list:
                rounds_data = agent_data.loc[agent_data.subsession_round_number.isin(rounds[0])]
                group_by = rounds_data.groupby(by='participant_code').agg({'subsession_round_number': 'count',
                                                                           'with_expert': 'sum'})
                group_by['pct with expert rounds ' + rounds[1]] =\
                    100 * group_by.with_expert / group_by.subsession_round_number
                final_data = final_data.merge(group_by[['pct with expert rounds ' + rounds[1]]],
                                              left_index=True, right_index=True)

            x_points = list()
            y_points = list()
            colors_groups = list()

            x_per_use = [1, 2, 3]
            for user in final_data.iterrows():
                x_points.append(x_per_use)
                user_lists = user[1].values.tolist()
                y_points.append(user_lists)
                if user_lists[0] >= user_lists[1] >= user_lists[2]:
                    colors_groups.append(0)
                elif user_lists[0] >= user_lists[1] <= user_lists[2]:
                    colors_groups.append(1)
                elif user_lists[0] <= user_lists[1] <= user_lists[2]:
                    colors_groups.append(2)
                elif user_lists[0] <= user_lists[1] >= user_lists[2]:
                    colors_groups.append(3)
                else:
                    print(f'{user[0]} no in any of the colors groups')

            legend = [f'num of down {colors_groups.count(0)}', f'num of down-up {colors_groups.count(1)}',
                      f'num of up {colors_groups.count(2)}', f'num of up-down {colors_groups.count(3)}']

            create_point_plot_per_user(
                x_points, y_points,
                title=f'% of trials each DM chose the expert recommendation for agent {agent} and {self.gender}',
                xlabel=f'round session ({rounds_list[0][1]}, {rounds_list[1][1]}, {rounds_list[2][1]})',
                colors_groups=colors_groups, ylabel='% of rounds DM chose with expert',
                inner_dir='pct_decisions_with_expert', legend=legend)

        return

    def pct_ev_vs_pct_average(self):
        """
        Calculate the % of trials each DM chose based on the EV (0/1) vs the % of times they chose based on average
        :return:
        """
        data_to_use = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        group_by = data_to_use.groupby(by=['participant_code']).agg({'subsession_round_number': 'count',
                                                                     'with_expert': 'sum',
                                                                     'with_average' + unknown_appendix_list[0]: 'sum'})
        group_by['pct average'] = 100 * group_by['with_average' + unknown_appendix_list[0]] /\
                                  group_by.subsession_round_number
        group_by['pct expert'] = 100 * group_by.with_expert / group_by.subsession_round_number
        group_by = group_by.merge(data_to_use[['participant_code', 'player_expert_type']],
                                  left_index=True, right_on='participant_code')
        group_by = group_by.drop_duplicates('participant_code')
        group_by.to_csv(os.path.join(data_analysis_directory, 'pct_ev_vs_pct_average_' + self.gender + '.csv'))

        return

    def average_gamble_against_recommendations(self):
        """
        Create a graph of the % of participants entered when the agent recommended not to enter and vs,
        per gamble's average
        :return:
        """
        data_to_use = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        agents_list = self.results_payments.player_expert_type.unique().tolist()
        agents_list = [agent for agent in agents_list if 'ten' not in agent]
        columns_list = ['average' + unknown_appendix_list[0]]
        if known_unknown_exp:
            columns_list += ['diff_avg_k_avg_u', 'average' + unknown_appendix_list[1]]
        for agent in agents_list:
            agent_data = data_to_use.loc[data_to_use.player_expert_type == agent]
            for column in columns_list:
                # % of participant entered when the agent recommended not to enter
                agent_didnt_recommend = agent_data.loc[agent_data.expert_recommended == 0].copy()
                agent_didnt_recommend_groupby = agent_didnt_recommend.groupby(by=column).agg({
                    'participant_code': 'count',
                    appendix + '_sender_payoff': 'sum'
                })
                agent_didnt_recommend_groupby['pct participant'] =\
                    100 * agent_didnt_recommend_groupby[appendix + '_sender_payoff'] /\
                    agent_didnt_recommend_groupby.participant_code

                create_chart_bars(title=f'% of participant entered when the agent recommended not to enter per '
                                        f'{column} for agent {agent} and {self.gender}',
                                  x=[agent_didnt_recommend_groupby.index.values],
                                  y=[agent_didnt_recommend_groupby['pct participant'].values.tolist()],
                                  xlabel='gamble average', ylabel='% of participants',
                                  curr_date_directory=os.path.join(data_analysis_directory,
                                                                   'average_gamble_against_recommendations'),
                                  percentage_graph=True)

                # % of participant didn't enter when the agent recommended to enter
                agent_recommended = agent_data.loc[agent_data.expert_recommended == 1].copy()
                agent_recommended_groupby = agent_recommended.groupby(by=column).agg({
                    'participant_code': 'count',
                    appendix + '_receiver_choice': 'sum'
                })
                agent_recommended_groupby['pct participant'] =\
                    100 * agent_recommended_groupby[appendix + '_receiver_choice'] /\
                    agent_recommended_groupby.participant_code

                create_chart_bars(title=f"% of participant didn't enter when the agent recommended to enter "
                                        f"per {column} for agent {agent} and {self.gender}",
                                  x=[agent_recommended_groupby.index.values],
                                  y=[agent_recommended_groupby['pct participant'].values.tolist()], xlabel='gamble average',
                                  ylabel='% of participants',
                                  curr_date_directory=os.path.join(data_analysis_directory,
                                                                   'average_gamble_against_recommendations'),
                                  percentage_graph=True)

        return

    def pct_entered_per_prob(self):
        """
        Create graph of the % of participants entered per probability for the accurate agent
        :return:
        """
        data_to_use = self.results_payments.loc[(self.results_payments[appendix + '_receiver_timeout'] == 0) &
                                                (self.results_payments.player_expert_type.isin(
                                                    ['accurate', 'one_player']))]
        if not data_to_use.empty:
            groupby = data_to_use.groupby(
                by=appendix + '_p' + unknown_appendix_list[0] + '_lottery').agg({'participant_code': 'count',
                                                                                 appendix + '_sender_payoff': 'sum'})
            groupby['pct participant'] = 100 * groupby[appendix + '_sender_payoff'] / groupby.participant_code
            create_chart_bars(
                title=f'% of participant entered per unknown probability for accurate agent and {self.gender}',
                x=[groupby.index.values], y=[groupby['pct participant'].values.tolist()], xlabel='gamble probability',
                ylabel='% of participants', curr_date_directory=data_analysis_directory, percentage_graph=True,
                x_is_float=True)

        return

    def draw_entry_rate(self):
        """
        Draw the entry rate in the first and second half
        :return:
        """

        data_to_use = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        agents_list = self.results_payments.player_expert_type.unique().tolist()
        agents_list = [[agent] for agent in agents_list if 'ten' not in agent]
        agents_list += [[agent[0] for agent in agents_list]]  # add for all agents
        legend = ['first half', 'second half']
        for agent in agents_list:
            x_points = list()
            y_points = list()
            agent_data = data_to_use.loc[data_to_use.player_expert_type.isin(agent)]
            rounds_list = [list(range(1, int(num_trials/2) + 1)), list(range(int(num_trials/2) + 1, num_trials))]
            for i, rounds in enumerate(rounds_list):
                rounds_data = agent_data.loc[agent_data.subsession_round_number.isin(rounds)]
                group_by = rounds_data.groupby(by='participant__current_page_name').agg(
                    {'subsession_round_number': 'count', appendix + '_sender_payoff': 'sum'})
                group_by['Entry rate'] = \
                    round(100 * group_by[appendix + '_sender_payoff'] / group_by.subsession_round_number, 2)
                x_points.append(i + 1)
                y_points.append(group_by['Entry rate'][0])

            if len(agent) > 1:  # for all agent types
                agent = ['all_agent']
            create_chart_bars(title=f'Entry rate in first and second half for agent {agent[0]} for {self.gender}',
                              x=[x_points], y=[y_points], xlabel='first and second half', ylabel='entry rate',
                              curr_date_directory=data_analysis_directory, legend=legend)

        return

    def compare_zero_one_p_win(self):
        """
        Compare the entry rate of DMs between the zero-one and the p_win agents
        :return:
        """
        compare = {(1, 0.9): ['', {'zero_one': [1, 'zero one agent answered 1'],
                                   'p_win_agent': [2, 'p_win agent answered 0.9']}],
                   (0, 0.4, 0.5, 0.6): [' not', {'zero_one': [1, 'zero one agent answered 0'],
                                                 'p_win_agent': [2, 'p_win agent answered 0.4/0.5/0.6']}]
                   }
        data_to_use = self.results_payments.loc[self.results_payments[appendix + '_receiver_timeout'] == 0]
        # run only if we have p_win_agent in the data
        if 'p_win_agent' in data_to_use.player_expert_type.unique():
            for answer_list, details in compare.items():
                # get the relevant player_sender_answer
                answer_data = data_to_use.loc[data_to_use[appendix + '_sender_answer'].isin(answer_list)]
                x_points = list()
                y_points = list()
                legend = list()

                all_rounds_x_points = list()
                all_rounds_y_points = list()
                answer_dict = details[1]
                for agent, agent_list in answer_dict.items():
                    legend.append(agent_list[1])
                    curr_data = answer_data.loc[answer_data.player_expert_type == agent]
                    # per round data
                    count = pd.DataFrame(curr_data.groupby('subsession_round_number').participant_code.count())
                    sum_participant = pd.DataFrame(curr_data.groupby('subsession_round_number')[
                                                       appendix + '_sender_payoff'].sum())
                    pct = count.merge(sum_participant, left_index=True, right_index=True)
                    pct['pct'] = 100 * pct[appendix + '_sender_payoff'] / pct.participant_code

                    x_points.append(pct.index.values)
                    y_points.append(pct['pct'].tolist())

                    # not per round data
                    group_by = curr_data.groupby(by='participant__current_app_name').agg(
                        {'subsession_round_number': 'count', appendix + '_sender_payoff': 'sum'})
                    group_by['Entry rate'] = \
                        round(100 * group_by[appendix + '_sender_payoff'] / group_by.subsession_round_number, 2)
                    all_rounds_x_points.append(np.array([agent_list[0]]))
                    all_rounds_y_points.append(np.array([group_by['Entry rate'][0]]))

                create_chart_bars(title=f'% of DMs entered when zero_one or p_win agents recommended{details[0]} to'
                                        f' enter per round',
                                  x=x_points, y=y_points, xlabel='trial number', ylabel='% of participants',
                                  legend=legend, percentage_graph=True,
                                  curr_date_directory=os.path.join(data_analysis_directory,
                                                                   'compare_zero_one_p_win'))
                create_chart_bars(title=f'% of DMs entered when zero_one and p_win agents recommended{details[0]} to '
                                        f'enter',
                                  x=all_rounds_x_points, y=all_rounds_y_points, xlabel='expert type',
                                  ylabel='% of participants', legend=legend, add_lst_xticks=False,
                                  curr_date_directory=os.path.join(data_analysis_directory, 'compare_zero_one_p_win'))

        return


def main(main_gender='all_genders'):
    data_analysis_obj = DataAnalysis(main_gender)
    data_analysis_obj.compare_zero_one_p_win()
    if not not_agent_data:
        data_analysis_obj.agent_payoff_analysis()
        data_analysis_obj.dm_payoff()
        data_analysis_obj.payoff_writer.save()
    data_analysis_obj.create_analysis_columns()
    data_analysis_obj.pct_decisions_with_expert()
    data_analysis_obj.analyze_dm_entrances()
    data_analysis_obj.analyze_dm_choices_per_round()
    data_analysis_obj.create_2_2_tables()
    data_analysis_obj.average_ev_gap_compare()
    data_analysis_obj.average_gamble_against_recommendations()
    data_analysis_obj.pct_entered_per_prob()
    data_analysis_obj.pct_ev_vs_pct_average()
    data_analysis_obj.draw_entry_rate()
    data_analysis_obj.results_payments.to_csv(os.path.join(data_analysis_directory,
                                                           'results_payments_' + main_gender + '.csv'))


if __name__ == '__main__':
    if split_gender:
        orig_data_analysis_directory = data_analysis_directory
        for gender in ['Male', 'Female']:
            print(f'start analyze for gender {gender}')
            data_analysis_directory = os.path.join(orig_data_analysis_directory, gender)
            if not (os.path.exists(data_analysis_directory)):
                os.makedirs(data_analysis_directory)
            main(gender)

    else:
        main()
