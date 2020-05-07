import scipy.stats
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import matplotlib.pyplot as plt


all_conditions = ['num_only', 'both', 'verbal', 'numeric']
conditions_per_study = {1: ['verbal', 'numeric'], 2: ['num_only', 'both']}
condition_names_per_study = {1: ['Expert-both-DM-Verbal', 'Expert-both-DM-Number'],
                             2: ['Expert-both-DM-both', 'Expert-Number-DM-Number']}
base_directory = os.path.abspath(os.curdir)
analysis_directory = os.path.join(base_directory, 'analysis', 'text_exp_2_tests')
colors = {'verbal': 'pink', 'numeric': 'forestgreen', 'num_only': 'crimson', 'both': 'darkblue'}
markers = {'verbal': 's', 'numeric': '.', 'num_only': 'v', 'both': 'd'}
graph_directory = os.path.join(base_directory, 'per_study_graphs')

rounds_dict = {
    'all rounds': {
        'numeric_p_enter': [0.27, 0.11, 0.11, 0.19, 0.11, 0.6, 0.32, 0.36, 0.53, 0.59, 0.75, 0.72, 0.76, 0.83, 0.84, 0.89],
        'both_p_enter': [0, 0.2, 0, 0, 0, 0, 0.21, 0.23, 0.5, 0.25, 0.5, 0.66, 0.83, 0.85, 0.8, 0.89],
        'numeric_experts': [22, 52, 17, 61, 9, 15, 166, 158, 66, 77, 37, 274, 177, 431, 658, 1669],
        'both_experts': [3, 5, 2, 12, 1, 0, 33, 21, 14, 12, 2, 30, 31, 75, 102, 249],
        'verbal_p_enter': [0.06, 0.08, 0.19, 0.09, 0.09, 0.15, 0.22, 0.19, 0.54, 0.69, 0.34, 0.65, 0.71, 0.8, 0.76, 0.85],
        'verbal_experts': [17, 37, 16, 55, 11, 26, 187, 189, 67, 55, 29, 237, 156, 430, 717, 1835],
        'num_only_p_enter': [0.0, 0.33, 0.29, 0.36, 0.0, 0.0, 0.3, 0.29, 0.5, 0.57, 0.5, 0.73, 0.81, 0.88, 0.85, 0.91],
        'num_only_experts': [2, 9, 7, 11, 1, 2, 10, 35, 16, 30, 18, 52, 42, 77, 126, 148]},
    'round 1': {
        'numeric_p_enter': [0.33, 0.0, 0.0, 0.25, 0.5, 0.8, 0.42, 0.62, 1.0, 0.94, 0.62, 0.94, 1.0, 0.91, 0.93, 0.95],
        'both_p_enter': [None, 1.0, None, None, None, None, None, 0.67, 0.5, 0.33, None, 0.78, 1.0, 0.78, 0.75, 0.9],
        'numeric_experts': [3, 2, 2, 4, 2, 5, 12, 13, 4, 18, 8, 32, 18, 47, 72, 148],
        'both_experts': [0, 1, 0, 0, 0, 0, 0, 3, 2, 3, 0, 9, 4, 9, 8, 20],
        'verbal_p_enter': [0.0, 0.0, 0.4, 0.18, None, 0.0, 0.45, 0.15, 0.33, 1.0, 0.67, 0.7, 0.76, 0.92, 0.86, 0.93],
        'verbal_experts': [2, 3, 5, 11, 0, 1, 20, 26, 3, 6, 3, 20, 17, 49, 85, 153],
        'num_only_p_enter': [None, 0.0, 1.0, 0.5, None, None, None, None, 1.0, 0.88, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'num_only_experts': [0, 1, 1, 2, 0, 0, 0, 0, 2, 8, 1, 5, 5, 5, 14, 15]},
    'round 2': {
        'numeric_p_enter': [0.0, 0.4, 0.0, 0.12, None, 0.5, 0.56, 0.26, 0.67, 1.0, 1.0, 0.78, 0.86, 0.84, 0.89, 0.92],
        'both_p_enter': [None, None, 0.0, 0.0, None, None, 0.0, 0.5, None, 0.33, 1.0, None, 1.0, 0.75, 0.62, 1.0],
        'numeric_experts': [1, 5, 5, 8, 0, 2, 18, 19, 9, 3, 3, 32, 21, 45, 65, 155],
        'both_experts': [0, 0, 1, 4, 0, 0, 3, 2, 0, 3, 1, 0, 4, 8, 8, 26],
        'verbal_p_enter': [0.0, 0.25, 0.0, 0.11, None, None, 0.22, 0.27, 0.5, 0.57, 1.0, 0.74, 0.65, 0.74, 0.81, 0.81],
        'verbal_experts': [2, 4, 3, 9, 0, 0, 32, 33, 12, 7, 1, 34, 20, 31, 78, 143],
        'num_only_p_enter': [None, None, 0.0, 0.67, None, None, 0.33, 0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.83, 0.8, 1.0],
        'num_only_experts': [0, 0, 1, 3, 0, 0, 3, 3, 2, 1, 2, 5, 2, 18, 5, 14]},

    'round 3': {
        'numeric_p_enter': [0.0, 0.5, 0.0, 0.15, None, 0.0, 0.2, 0.18, 0.14, 0.75, 1.0, 0.83, 0.54, 0.87, 0.88, 0.89],
        'both_p_enter': [None, None, None, None, 0.0, None, 0.14, 0.5, None, 0.0, None, 0.57, 0.75, 0.75, 0.62, 0.88],
        'numeric_experts': [3, 4, 1, 13, 0, 1, 15, 11, 7, 4, 3, 18, 13, 45, 57, 196],
        'both_experts': [0, 0, 0, 0, 1, 0, 7, 2, 0, 1, 0, 7, 4, 12, 8, 17],
        'verbal_p_enter': [0.0, 0.0, None, 0.0, None, 0.2, 0.1, 0.21, 0.67, 0.43, None, 0.86, 0.78, 0.8, 0.79, 0.88],
        'verbal_experts': [6, 1, 0, 5, 0, 5, 10, 14, 6, 7, 0, 22, 18, 41, 62, 207],
        'num_only_p_enter': [None, 1.0, None, None, None, None, 0.0, 0.29, 0.0, 0.4, 1.0, 0.71, 1.0, 0.8, 0.64, 0.8],
        'num_only_experts': [0, 1, 0, 0, 0, 0, 2, 7, 1, 5, 1, 7, 4, 5, 11, 15]},

    'round 4': {
        'numeric_p_enter': [0.0, 0.0, 0.5, 0.25, 0.0, 1.0, 0.41, 0.4, 0.67, 0.67, 1.0, 0.62, 0.78, 0.81, 0.85, 0.9],
        'both_p_enter': [None, None, None, 0.0, None, None, 0.5, 0.0, 0.25, None, None, 0.0, 1.0, 0.78, 0.7, 0.96],
        'numeric_experts': [1, 6, 2, 4, 2, 1, 17, 10, 6, 6, 5, 26, 32, 47, 67, 158],
        'both_experts': [0, 0, 0, 3, 0, 0, 4, 2, 4, 0, 0, 2, 1, 9, 10, 24],
        'verbal_p_enter': [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.09, 0.27, 0.43, 1.0, 0.0, 0.6, 0.85, 0.78, 0.78, 0.85],
        'verbal_experts': [2, 5, 2, 5, 2, 1, 22, 15, 7, 2, 3, 15, 13, 32, 79, 202],
        'num_only_p_enter': [None, 0.5, 0.33, 0.0, None, None, 0.5, 0.0, None, 0.5, 1.0, 0.5, 0.5, 1.0, 0.78, 1.0],
        'num_only_experts': [0, 2, 3, 3, 0, 0, 2, 3, 0, 4, 1, 2, 6, 6, 18, 8]},

    'round 5': {
        'numeric_p_enter': [0.25, 0.17, None, 0.0, 0.0, None, 0.25, 0.27, 0.5, 0.5, 1.0, 0.76, 0.7, 0.86, 0.84, 0.9],
        'both_p_enter': [0.0, 0.0, 0.0, 0.0, None, None, 0.0, None, 0.0, 0.0, None, 1.0, 0.67, 0.8, 0.67, 0.81],
        'numeric_experts': [4, 6, 0, 5, 3, 0, 20, 15, 6, 10, 1, 34, 23, 35, 76, 152],
        'both_experts': [1, 2, 1, 2, 0, 0, 1, 0, 2, 1, 0, 4, 3, 5, 12, 26],
        'verbal_p_enter': [0.0, 0.11, 0.0, 0.4, None, 0.0, 0.31, 0.21, 0.75, 0.83, 0.0, 0.62, 0.89, 0.85, 0.76, 0.86],
        'verbal_experts': [1, 9, 1, 5, 0, 2, 16, 19, 4, 6, 4, 24, 9, 53, 71, 180],
        'num_only_p_enter': [None, 0.0, None, 0.0, None, 0.0, 1.0, 0.67, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.88, 0.85],
        'num_only_experts': [0, 3, 0, 2, 0, 1, 1, 3, 2, 1, 2, 5, 3, 7, 8, 20]},

    'round 6': {
        'numeric_p_enter': [0.5, 0.14, 0.0, 0.25, None, None, 0.35, 0.45, 0.71, 0.5, 0.67, 0.7, 0.6, 0.74, 0.85, 0.86],
        'both_p_enter': [0.0, None, None, None, None, None, 0.33, 0.33, 0.67, 1.0, None, 0.0, 1.0, 0.5, 0.85, 0.9],
        'numeric_experts': [4, 7, 1, 4, 0, 0, 31, 22, 7, 4, 3, 27, 15, 38, 71, 156],
        'both_experts': [1, 0, 0, 0, 0, 0, 3, 3, 3, 1, 0, 1, 4, 2, 13, 29],
        'verbal_p_enter': [0.0, 0.2, None, 0.0, 0.0, 0.0, 0.6, 0.1, 0.62, 0.86, 0.8, 0.45, 0.62, 0.79, 0.67, 0.81],
        'verbal_experts': [1, 5, 0, 4, 2, 2, 10, 10, 8, 7, 5, 33, 13, 34, 81, 193],
        'num_only_p_enter': [0.0, None, 0.0, None, None, None, None, 0.0, 0.0, 0.67, 0.0, 0.5, 0.67, 0.83, 0.91, 0.89],
        'num_only_experts': [1, 0, 1, 0, 0, 0, 0, 6, 1, 3, 1, 6, 3, 6, 11, 19]},

    'round 7': {
        'numeric_p_enter': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.27, 0.36, 0.33, 0.0, 1.0, 0.65, 0.94, 0.84, 0.83, 0.92],
        'both_p_enter': [None, None, None, 0.0, None, None, 0.5, 0.0, 1.0, 0.0, None, 1.0, 0.5, 1.0, 1.0, 0.93],
        'numeric_experts': [2, 6, 1, 2, 1, 1, 11, 11, 6, 6, 1, 31, 18, 38, 75, 179],
        'both_experts': [0, 0, 0, 1, 0, 0, 4, 1, 1, 1, 0, 1, 2, 5, 14, 28],
        'verbal_p_enter': [None, 0.0, 0.0, 0.0, 0.0, 0.33, 0.33, 0.18, 0.67, 0.6, 0.0, 0.52, 0.64, 0.91, 0.75, 0.86],
        'verbal_experts': [0, 4, 1, 4, 2, 3, 9, 17, 3, 5, 3, 21, 14, 47, 73, 200],
        'num_only_p_enter': [None, 0.0, 0.0, None, 0.0, None, 0.0, 0.33, 0.0, 1.0, 0.5, 0.8, 0.83, 0.92, 0.9, 0.85],
        'num_only_experts': [0, 1, 1, 0, 1, 0, 1, 3, 1, 1, 4, 5, 6, 12, 10, 13]},

    'round 8': {
        'numeric_p_enter': [None, 0.0, 0.0, 0.29, None, 0.5, 0.25, 0.25, 0.5, 0.69, 0.8, 0.6, 0.58, 0.82, 0.66, 0.91],
        'both_p_enter': [0.0, 0.0, None, 0.0, None, None, 0.0, 0.0, None, 0.0, None, 1.0, 1.0, 1.0, 0.8, 0.88],
        'numeric_experts': [0, 7, 2, 7, 0, 2, 16, 16, 4, 13, 5, 15, 12, 40, 56, 192],
        'both_experts': [1, 1, 0, 2, 0, 0, 6, 1, 0, 1, 0, 1, 4, 12, 5, 25],
        'verbal_p_enter': [1.0, 0.0, 0.0, 0.0, None, 0.14, 0.15, 0.15, 0.56, 0.8, 0.33, 0.64, 0.62, 0.67, 0.76, 0.82],
        'verbal_experts': [1, 1, 1, 7, 0, 7, 26, 13, 9, 5, 3, 22, 16, 46, 58, 191],
        'num_only_p_enter': [None, None, None, 1.0, None, 0.0, None, 0.5, 0.75, 1.0, None, 0.6, 0.83, 1.0, 1.0, 0.87],
        'num_only_experts': [0, 0, 0, 1, 0, 1, 0, 2, 4, 1, 0, 5, 6, 6, 18, 15]},

    'round 9': {
        'numeric_p_enter': [1.0, 0.0, 0.0, 0.17, None, 1.0, 0.21, 0.38, 0.5, 0.17, 0.67, 0.69, 0.82, 0.82, 0.9, 0.88],
        'both_p_enter': [None, None, None, None, None, None, 0.0, 0.0, 1.0, 0.0, 0.0, None, 0.5, 1.0, 1.0, 0.86],
        'numeric_experts': [1, 2, 1, 6, 0, 1, 14, 13, 8, 6, 3, 26, 17, 40, 52, 196],
        'both_experts': [0, 0, 0, 0, 0, 0, 1, 3, 2, 1, 1, 0, 4, 7, 11, 28],
        'verbal_p_enter': [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.08, 0.0, 0.33, 0.67, 0.2, 0.71, 0.65, 0.69, 0.77, 0.84],
        'verbal_experts': [1, 2, 2, 3, 2, 2, 24, 13, 3, 3, 5, 24, 23, 54, 62, 184],
        'num_only_p_enter': [None, 1.0, None, None, None, None, 0.0, 0.33, 0.0, 0.0, 0.2, 0.86, 0.67, 0.75, 0.69, 0.92],
        'num_only_experts': [0, 1, 0, 0, 0, 0, 1, 3, 2, 1, 5, 7, 3, 8, 16, 12]},

    'round 10': {
        'numeric_p_enter': [0.33, 0.0, 0.5, 0.38, 0.0, 0.0, 0.25, 0.43, 0.44, 0.29, 0.4, 0.67, 0.62, 0.84, 0.79, 0.89],
        'both_p_enter': [None, 0.0, None, None, None, None, 0.25, 0.0, None, None, None, 0.6, 1.0, 1.0, 0.85, 0.85],
        'numeric_experts': [3, 7, 2, 8, 1, 2, 12, 28, 9, 7, 5, 33, 8, 56, 67, 137],
        'both_experts': [0, 1, 0, 0, 0, 0, 4, 4, 0, 0, 0, 5, 1, 6, 13, 26],
        'verbal_p_enter': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.17, 0.21, 0.5, 0.43, 0.5, 0.73, 0.77, 0.86, 0.65, 0.84],
        'verbal_experts': [1, 3, 1, 2, 3, 3, 18, 29, 12, 7, 2, 22, 13, 43, 68, 182],
        'num_only_p_enter': [0.0, None, None, None, None, None, None, 0.6, 0.0, 0.2, 0.0, 0.2, 0.75, 0.75, 0.87, 0.94],
        'num_only_experts': [1, 0, 0, 0, 0, 0, 0, 5, 1, 5, 1, 5, 4, 4, 15, 17]},
}


honesty_dict = {'numeric': [5.21, 8.45, 8.87, 9.13, 9.33, 9.59, 9.58, 9.59, 9.8, 9.86],
                'verbal': [5.45, 8.51, 9.0, 9.21, 9.37, 9.56, 9.57, 9.64, 9.82, 9.87],
                'num_only': [5.02, 8.16, 8.45, 8.89, 8.82, 8.95, 9.59, 9.38, 9.65, 9.75],
                'both': [5.24, 8.58, 8.89, 9.24, 9.44, 9.48, 9.29, 9.68, 9.79, 9.85]}


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[(data.status == 'play') & (data.player_id_in_group == 1)]
    data = data[['group_sender_payoff',	'group_receiver_payoff', 'pair_id', 'group_lottery_result',
                 'group_average_score', 'group_sender_answer_scores']]
    data['honesty'] = data['group_sender_answer_scores'] - data['group_average_score']
    data['expert_answer_above_8'] = np.where(data.group_sender_answer_scores > 8, 1, 0)

    return data


def significant_tests(data_1: pd.DataFrame, data_2: pd.DataFrame, criterion: str):
    statistic = scipy.stats.f_oneway(data_1, data_2)
    print(criterion)
    print(f'Groups criterion vars: {np.var(data_1), np.var(data_2)}')
    print(f'Groups criterion means: {np.mean(data_1), np.mean(data_2)}')
    kruskal = scipy.stats.kruskal(data_1, data_2)
    ttest = scipy.stats.ttest_ind(data_1, data_2)
    print(f'ANOVA test: {statistic},\nKruskal test: {kruskal}\nT_test: {ttest}\n')

    return


def trust_graphs(study: int, data_1_p_enter_list_all_rounds: list=None, data_2_p_enter_list_all_rounds: list = None):
    """

    :param study:
    :param data_1_p_enter_list_all_rounds: [average_p_enter_est<8, average_p_enter_est>8] for data 1
    :param data_2_p_enter_list_all_rounds: [average_p_enter_est<8, average_p_enter_est>8] for data 2
    :return:
    """
    x = [2.5, 3.3, 3.8, 4.2, 5, 5.4, 5.8, 6.3, 7.1, 7.5, 7.9, 8.3, 8.8, 9.2, 9.6, 10]
    condition_1 = conditions_per_study[study][0]
    condition_2 = conditions_per_study[study][1]
    for rng, in_title in [[range(1, 6), 'First 5'], [range(6, 11), 'Last'], [range(1, 11), 'All']]:
        fig100, ax100 = plt.subplots()
        data_1_p_enter_list = list()
        data_2_p_enter_list = list()
        data_1_experts_list = list()
        data_2_experts_list = list()

        for idx in range(len(x)):
            values = [rounds_dict[f'round {j}'][f'{condition_1}_p_enter'][idx] for j in rng]
            values = [x for x in values if x is not None]
            data_1_p_enter_list.append(round(np.average(values), 2))
            data_1_experts_list.append(sum([rounds_dict[f'round {j}'][f'{condition_1}_experts'][idx] for j in rng]))

            values = [rounds_dict[f'round {z}'][f'{condition_2}_p_enter'][idx] for z in rng]
            values = [x for x in values if x is not None]
            data_2_p_enter_list.append(round(np.average(values), 2))
            data_2_experts_list.append(sum([rounds_dict[f'round {j}'][f'{condition_2}_experts'][idx] for j in rng]))

        ax100.plot(x, data_1_p_enter_list, color=colors[condition_1], label=condition_names_per_study[study][0],
                   marker=markers[condition_1], linestyle='-')
        ax100.plot(x, data_2_p_enter_list, color=colors[condition_2], label=condition_names_per_study[study][1],
                   marker=markers[condition_2], linestyle='-')

        plt.title(f"P(DM chose hotel) as a function of the experts\nnumerical estimate by Condition in {in_title}"
                  f" rounds for study {study}")
        plt.xlabel('Experts Numerical Estimate', fontsize=15)
        plt.ylabel('P(DM chose hotel)', fontsize=15)
        ax100.legend(loc='upper left', shadow=True, fontsize=8)
        plt.xticks(x)
        # Add a table at the bottom of the axes
        the_table = ax100.table(cellText=[x, data_1_experts_list, data_2_experts_list],
                                rowLabels=['Experts Numerical Estimate', condition_names_per_study[study][0],
                                           condition_names_per_study[study][1]],
                                bbox=(0, -0.5, 1, 0.3))
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        # plt.yticks(range(4, 11))
        plt.show()
        fig100.savefig(os.path.join(
            graph_directory, f'P_enter_as_a_function_of_the_experts_numerical_estimate_by_Condition_in_{in_title}'
                             f'_rounds_for_study_{study}.png'), bbox_inches='tight')

        fig1000, ax1000 = plt.subplots()
        combine_dict = {
            'est<8': {f'{condition_1}_p_enter_list': [], f'{condition_2}_p_enter_list': [],
                      f'{condition_1}_experts_list': [], f'{condition_2}_experts_list': []},
            'est>8': {f'{condition_1}_p_enter_list': [], f'{condition_2}_p_enter_list': [],
                      f'{condition_1}_experts_list': [], f'{condition_2}_experts_list': []}
        }
        # split per est < 8:
        for idx, est in enumerate(x):
            if est < 8:
                if not np.isnan(data_1_p_enter_list[idx]):
                    combine_dict['est<8'][f'{condition_1}_p_enter_list'].append(data_1_p_enter_list[idx])
                if not np.isnan(data_2_p_enter_list[idx]):
                    combine_dict['est<8'][f'{condition_2}_p_enter_list'].append(data_2_p_enter_list[idx])
                combine_dict['est<8'][f'{condition_1}_experts_list'].append(data_1_experts_list[idx])
                combine_dict['est<8'][f'{condition_2}_experts_list'].append(data_2_experts_list[idx])
            else:
                if not np.isnan(data_1_p_enter_list[idx]):
                    combine_dict['est>8'][f'{condition_1}_p_enter_list'].append(data_1_p_enter_list[idx])
                if not np.isnan(data_2_p_enter_list[idx]):
                    combine_dict['est>8'][f'{condition_2}_p_enter_list'].append(data_2_p_enter_list[idx])
                combine_dict['est>8'][f'{condition_1}_experts_list'].append(data_1_experts_list[idx])
                combine_dict['est>8'][f'{condition_2}_experts_list'].append(data_2_experts_list[idx])

        data_1_p_enter_list = [round(np.average(combine_dict['est<8'][f'{condition_1}_p_enter_list']), 2),
                               round(np.average(combine_dict['est>8'][f'{condition_1}_p_enter_list']), 2)]
        data_2_p_enter_list = [round(np.average(combine_dict['est<8'][f'{condition_2}_p_enter_list']), 2),
                               round(np.average(combine_dict['est>8'][f'{condition_2}_p_enter_list']), 2)]
        data_1_experts_list = [np.sum(combine_dict['est<8'][f'{condition_1}_experts_list']),
                               np.sum(combine_dict['est>8'][f'{condition_1}_experts_list'])]
        data_2_experts_list = [np.sum(combine_dict['est<8'][f'{condition_2}_experts_list']),
                               np.sum(combine_dict['est>8'][f'{condition_2}_experts_list'])]

        if in_title == 'All':
            data_1_p_enter_list = data_1_p_enter_list_all_rounds
            data_2_p_enter_list = data_2_p_enter_list_all_rounds

        ax1000.plot(['est<8', 'est>8'], data_1_p_enter_list, color=colors[condition_1],
                    label=condition_names_per_study[study][0], marker=markers[condition_1], linestyle='-')
        ax1000.plot(['est<8', 'est>8'], data_2_p_enter_list, color=colors[condition_2],
                    label=condition_names_per_study[study][1], marker=markers[condition_2], linestyle='-')

        plt.title(f"P(DM chose hotel) as a function of the experts\nnumerical estimate by Condition in "
                  f"{in_title} rounds for study {study}")
        plt.xlabel('Experts Numerical Estimate', fontsize=15)
        plt.ylabel('P(DM chose hotel)', fontsize=15)
        ax1000.legend(loc='upper left', shadow=True, fontsize=8)
        # plt.xticks(['est<8', 'est>8'])
        # Add a table at the bottom of the axes
        data_1_experts_list = [(data_1_experts_list[idx], data_1_p_enter_list[idx]) for idx in
                               range(len(data_1_p_enter_list))]
        data_2_experts_list = [(data_2_experts_list[idx], data_2_p_enter_list[idx]) for idx in
                               range(len(data_2_p_enter_list))]
        the_table = ax1000.table(
            cellText=[['est<8', 'est>8'], data_1_experts_list, data_2_experts_list],
            rowLabels=['Experts Numerical Estimate', condition_names_per_study[study][0],
                       condition_names_per_study[study][1]],
            bbox=(0, -0.5, 1, 0.3))
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        plt.show()
        fig1000.savefig(
            os.path.join(graph_directory, f'P_enter_as_a_function_of_the_experts_numerical_estimate_by_Condition_in_'
                                          f'{in_title}_rounds_est_8__for_study_{study}.png'), bbox_inches='tight')


def honesty_graph(study: int):
    """The Communication Type Effect on the Experts Cheating Level"""
    fig1, ax1 = plt.subplots()
    ax1.axis([4, 10, 4, 10])
    x = [4.17, 6.66, 7.44, 7.97, 8.11, 8.33, 8.94, 9.19, 9.54, 9.77]

    # ax1.plot(x, verbal_cl, color=colors[0], label='Verbal', marker=markers[0], linestyle='-')
    ax1.plot(x, honesty_dict[conditions_per_study[study][0]], color=colors[conditions_per_study[study][0]],
             label=condition_names_per_study[study][0], marker=markers[conditions_per_study[study][0]], linestyle='-')
    ax1.plot(x, honesty_dict[conditions_per_study[study][1]], color=colors[conditions_per_study[study][1]],
             label=condition_names_per_study[study][1], marker=markers[conditions_per_study[study][1]], linestyle='-')
    ax1.plot(x, x, color='darkviolet', marker='.', linestyle='-', label='Truth Telling')

    plt.title(f"The Selected Score as a Function of the Hotels' Average Score\nby Condition for study {study}")
    plt.xlabel('Decision Maker Expected Payoff', fontsize=15)
    plt.ylabel('Expert Average Signal', fontsize=15)
    ax1.legend(loc='upper left', shadow=True, fontsize=8)
    plt.xticks(range(4, 11))
    plt.yticks(range(4, 11))
    plt.show()
    fig1.savefig(os.path.join(graph_directory,
                              f'The_Communication_Type_Effect_on_the_Experts_Cheating_Level_study_{study}.png'),
                 bbox_inches='tight')


class SignificanceTests:
    def __init__(self):
        """Load data"""
        self.linear_scores = defaultdict(pd.DataFrame)
        self.all_data = defaultdict(pd.DataFrame)

        for condition in all_conditions:
            self.linear_scores[condition] = pd.read_csv(os.path.join(analysis_directory, condition, 'linear_scores.csv'))
            all_data_cond = pd.read_csv(os.path.join(analysis_directory, condition, 'results_payments_status.csv'))
            all_data_cond = clean_data(all_data_cond)
            self.all_data[condition] = all_data_cond

        return

    def expert_payoff_test(self, study: int):
        study_1_cond = self.all_data[conditions_per_study[study][0]]
        study_2_cond = self.all_data[conditions_per_study[study][1]]
        study_1_cond_mean_ex_payoff = study_1_cond.groupby(by='pair_id').group_sender_payoff.sum()
        study_2_cond_mean_ex_payoff = study_2_cond.groupby(by='pair_id').group_sender_payoff.sum()

        significant_tests(study_1_cond_mean_ex_payoff, study_2_cond_mean_ex_payoff,
                          f'Expert Payoff Study {study}: {conditions_per_study[study][0]}, '
                          f'{conditions_per_study[study][1]}')

        return

    def honesty_test(self, study: int):
        study_1_cond = self.all_data[conditions_per_study[study][0]]
        study_2_cond = self.all_data[conditions_per_study[study][1]]
        study_1_cond_mean_honesty = study_1_cond.groupby(by='pair_id').honesty.mean()
        study_2_cond_mean_honesty = study_2_cond.groupby(by='pair_id').honesty.mean()

        significant_tests(study_1_cond_mean_honesty, study_2_cond_mean_honesty,
                          f'Honesty Study {study}: {conditions_per_study[study][0]}, {conditions_per_study[study][1]}')

        honesty_graph(study=study)

        return

    def trust_test(self, study: int):
        study_1_cond = self.all_data[conditions_per_study[study][0]]
        study_2_cond = self.all_data[conditions_per_study[study][1]]
        data_1_p_enter_list_all_rounds = list()
        data_2_p_enter_list_all_rounds = list()
        for name, number in [['expert estimation below 8', 0], ['expert estimation above 8', 1]]:
            study_1_cond_name = study_1_cond.loc[study_1_cond.expert_answer_above_8 == number]
            study_2_cond_name = study_2_cond.loc[study_2_cond.expert_answer_above_8 == number]

            study_1_cond_mean = study_1_cond_name.groupby(by='pair_id').group_sender_payoff.mean()
            study_2_cond_mean = study_2_cond_name.groupby(by='pair_id').group_sender_payoff.mean()

            significant_tests(study_1_cond_mean, study_2_cond_mean,
                              f'Trust Study {study} for {name}: {conditions_per_study[study][0]}, '
                              f'{conditions_per_study[study][1]}')

            data_1_p_enter_list_all_rounds.append(round(study_1_cond_name.group_sender_payoff.mean(), 2))
            data_2_p_enter_list_all_rounds.append(round(study_2_cond_name.group_sender_payoff.mean(), 2))

        trust_graphs(study=study, data_1_p_enter_list_all_rounds=data_1_p_enter_list_all_rounds,
                     data_2_p_enter_list_all_rounds=data_2_p_enter_list_all_rounds)

        return


def main():
    tests_obj = SignificanceTests()
    for study in [1, 2]:
        print(f'Significant tests for study {study}')
        tests_obj.expert_payoff_test(study=study)
        tests_obj.honesty_test(study=study)
        tests_obj.trust_test(study=study)


if __name__ == '__main__':
    main()
