import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
from data_analysis import autolabel
import os
import scipy.stats


"""Onlu Num text experiment initial results analysis"""
# directory = '/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/data_analysis/analysis/' \
#             'text_exp_2_tests/deterministic_initial_analysis'
# data = pd.read_excel(os.path.join(directory, 'initial_analysis.xlsx'), sheet_name='data_to_plot_stochastic')
# # fig = plt.figure(figsize=(15, 15))
# # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# participants = data.participant_code.unique()
# colors = ['red', 'blue']
# for user_num, user in enumerate(participants):
#     user_data = data.loc[data.participant_code == user]
#     fig, ax = plt.subplots(figsize=(10, 5))
#     all_round_num = user_data.subsession_round_number.tolist()
#     all_expert_score = user_data.group_sender_answer_scores.tolist()
#     all_x_real_score = user_data.group_lottery_result.tolist()
#     all_x_index = user_data.group_sender_answer_index.tolist()
#     all_x_dm_decision = user_data.group_sender_payoff.tolist()
#     all_average_score = user_data.average_score.round(1).tolist()
#     all_median_score = user_data.median_score.round(1).tolist()
#     all_median_index = user_data.median_index.round(1).tolist()
#     all_index_median_diff = user_data.index_median_diff.round(1).tolist()
#     all_score_median_diff = user_data.score_median_diff.round(1).tolist()
#     all_score_average_diff = user_data.score_average_diff.round(1).tolist()
#     chose_points_x, chose_points_y = list(), list()
#     not_chose_points_x, not_chose_points_y = list(), list()
#     condition = user_data.condition.unique()[0]
#     for i, point in enumerate(all_round_num):
#         color = colors[0] if all_x_dm_decision[i] == 1 else colors[1]
#         if all_x_dm_decision[i] == 1:
#             color = colors[0]
#             chose_points_x.append(all_round_num[i])
#             chose_points_y.append(all_expert_score[i])
#         else:
#             color = colors[1]
#             not_chose_points_x.append(all_round_num[i])
#             not_chose_points_y.append(all_expert_score[i])
#
#         ax.annotate(f'({all_x_real_score[i]}, {all_expert_score[i]}, {all_x_index[i]},\n'
#                     f'{all_average_score[i]}, {all_median_score[i]}\n'
#                     f'{all_score_median_diff[i]}, {all_score_average_diff[i]}, {all_index_median_diff[i]})',
#                     (point - 0.4, all_expert_score[i] - 0.8), color=color, fontsize=8)
#     ax.scatter([chose_points_x], [chose_points_y], color=colors[0], marker=".", label='DM chose Hotel', s=0.5)
#     ax.scatter([not_chose_points_x], [not_chose_points_y], color=colors[1], marker=".", label='DM chose Stay Home', s=0.5)
#     average_rmse = round(math.sqrt(mean_squared_error(all_expert_score, all_average_score)), 2)
#     median_rmse = round(math.sqrt(mean_squared_error(all_expert_score, all_median_score)), 2)
#     avg_diff = round(sum(all_score_average_diff)/len(all_score_average_diff), 2)
#     median_diff = round(sum(all_score_median_diff)/len(all_score_median_diff), 2)
#     median_index_diff = round(sum(all_index_median_diff)/len(all_index_median_diff), 2)
#     print(f'pair number {user_num+1} with participant_code {user}')
#     plt.title(f'Pair number {user_num+1}, played {condition} condition results:\n'
#               f'(lottery score, expert chosen score, expert chosen index,\nhotel average score, hotel median score,\n'
#               f'chosen-median, chosen-average, index_median_diff)\n'
#               f'average score RMSE: {average_rmse}, median score RMSE: {median_rmse},\n'
#               f'average(chosen score-average score): {avg_diff},\n'
#               f'average(chosen score-median score): {median_diff}, '
#               f'average(index_median_diff): {median_index_diff}')
#     plt.xlabel('Round Number')
#     plt.ylabel('Expert Chosen Score')
#     plt.xticks(range(1, 11))
#     plt.yticks(range(1, 11))
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.show()
#     fig.savefig(os.path.join(directory, condition, f'Pair number {user_num+1} results.png'), bbox_inches='tight')


# """New text experiment initial results analysis"""
# directory = '/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/data_analysis/analysis/' \
#             'text_exp_2_tests/deterministic_initial_analysis'
# data = pd.read_excel(os.path.join(directory, 'initial_analysis.xlsx'), sheet_name='data_to_plot')
# # fig = plt.figure(figsize=(15, 15))
# # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# participants = data.participant_code.unique()
# colors = ['red', 'blue']
# for user_num, user in enumerate(participants):
#     user_data = data.loc[data.participant_code == user]
#     fig, ax = plt.subplots(figsize=(10, 5))
#     all_round_num = user_data.subsession_round_number.tolist()
#     all_expert_score = user_data.group_sender_answer_scores.tolist()
#     all_x_real_score = user_data.group_lottery_result.tolist()
#     all_x_index = user_data.group_sender_answer_index.tolist()
#     all_x_dm_decision = user_data.group_sender_payoff.tolist()
#     all_index_above = user_data.above.tolist()
#     all_index_below = user_data.below.tolist()
#     all_index_diff = user_data.index_diff.tolist()
#     all_chosen_index = user_data.chosen_index.tolist()
#     all_score_diff = user_data.score_diff.tolist()
#     chose_points_x, chose_points_y = list(), list()
#     not_chose_points_x, not_chose_points_y = list(), list()
#     condition = user_data.condition.unique()[0]
#     for i, point in enumerate(all_round_num):
#         color = colors[0] if all_x_dm_decision[i] == 1 else colors[1]
#         if all_x_dm_decision[i] == 1:
#             color = colors[0]
#             chose_points_x.append(all_round_num[i])
#             chose_points_y.append(all_expert_score[i])
#         else:
#             color = colors[1]
#             not_chose_points_x.append(all_round_num[i])
#             not_chose_points_y.append(all_expert_score[i])
#
#         ax.annotate(f'({all_x_real_score[i]},{all_expert_score[i]},\n'
#                     f'{all_index_above[i]}, {all_index_below[i]}, {all_index_diff[i]})',
#                     (point - 0.4, all_expert_score[i] - 0.6), color=color, fontsize=10)
#     ax.scatter([chose_points_x], [chose_points_y], color=colors[0], marker=".", label='DM chose Hotel')
#     ax.scatter([not_chose_points_x], [not_chose_points_y], color=colors[1], marker=".", label='DM chose Stay Home')
#     index_rmse = round(math.sqrt(mean_squared_error(all_x_index, all_chosen_index)), 2)
#     score_rmse = round(math.sqrt(mean_squared_error(all_expert_score, all_x_real_score)), 2)
#     index_avg_diff = round(sum(all_index_diff)/len(all_index_diff), 2)
#     score_avg_diff = round(sum(all_score_diff)/len(all_score_diff), 2)
#     print(f'pair number {user_num+1} with participant_code {user}')
#     plt.title(f'Pair number {user_num+1}, played {condition} condition results:\n'
#               f'(true score, expert chosen score, #numbers above, #numbers below, '
#               f'expert choice with respect to chosen index)\n'
#               f'index RMSE: {index_rmse}, score RMSE: {score_rmse}, score avg diff: {score_avg_diff},'
#               f'index avg diff: {index_avg_diff}')
#     plt.xlabel('Round Number')
#     plt.ylabel('Expert Chosen Score')
#     plt.xticks(range(1, 11))
#     plt.yticks(range(1, 11))
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.show()
#     fig.savefig(os.path.join(directory, condition, f'Pair number {user_num+1} results.png'), bbox_inches='tight')


"""P(Enter) as a function of the experts numerical estimate"""
x = [2.5, 3.3, 3.8, 4.2, 5, 5.4, 5.8, 6.3, 7.1, 7.5, 7.9, 8.3, 8.8, 9.2, 9.6, 10]
rounds_dict = {
    'all rounds': {
        'num_p_enter': [0.27, 0.11, 0.11, 0.19, 0.11, 0.6, 0.32, 0.36, 0.53, 0.59, 0.75, 0.72, 0.76, 0.83, 0.84, 0.89],
        'both_p_enter': [0, 0.2, 0, 0, 0, 0, 0.21, 0.23, 0.5, 0.25, 0.5, 0.66, 0.83, 0.85, 0.8, 0.89],
        'num_experts': [22, 52, 17, 61, 9, 15, 166, 158, 66, 77, 37, 274, 177, 431, 658, 1669],
        'both_experts': [3, 5, 2, 12, 1, 0, 33, 21, 14, 12, 2, 30, 31, 75, 102, 249],
        'verbal_p_enter': [0.06, 0.08, 0.19, 0.09, 0.09, 0.15, 0.22, 0.19, 0.54, 0.69, 0.34, 0.65, 0.71, 0.8, 0.76, 0.85],
        'verbal_experts': [17, 37, 16, 55, 11, 26, 187, 189, 67, 55, 29, 237, 156, 430, 717, 1835]},
    'round 1': {
        'num_p_enter': [0.33, 0.0, 0.0, 0.25, 0.5, 0.8, 0.42, 0.62, 1.0, 0.94, 0.62, 0.94, 1.0, 0.91, 0.93, 0.95],
        'both_p_enter': [None, 1.0, None, None, None, None, None, 0.67, 0.5, 0.33, None, 0.78, 1.0, 0.78, 0.75, 0.9],
        'num_experts': [3, 2, 2, 4, 2, 5, 12, 13, 4, 18, 8, 32, 18, 47, 72, 148],
        'both_experts': [0, 1, 0, 0, 0, 0, 0, 3, 2, 3, 0, 9, 4, 9, 8, 20],
        'verbal_p_enter': [0.0, 0.0, 0.4, 0.18, None, 0.0, 0.45, 0.15, 0.33, 1.0, 0.67, 0.7, 0.76, 0.92, 0.86, 0.93],
        'verbal_experts': [2, 3, 5, 11, 0, 1, 20, 26, 3, 6, 3, 20, 17, 49, 85, 153]},
    'round 2': {
        'num_p_enter': [0.0, 0.4, 0.0, 0.12, None, 0.5, 0.56, 0.26, 0.67, 1.0, 1.0, 0.78, 0.86, 0.84, 0.89, 0.92],
        'both_p_enter': [None, None, 0.0, 0.0, None, None, 0.0, 0.5, None, 0.33, 1.0, None, 1.0, 0.75, 0.62, 1.0],
        'num_experts': [1, 5, 5, 8, 0, 2, 18, 19, 9, 3, 3, 32, 21, 45, 65, 155],
        'both_experts': [0, 0, 1, 4, 0, 0, 3, 2, 0, 3, 1, 0, 4, 8, 8, 26],
        'verbal_p_enter': [0.0, 0.25, 0.0, 0.11, None, None, 0.22, 0.27, 0.5, 0.57, 1.0, 0.74, 0.65, 0.74, 0.81, 0.81],
        'verbal_experts': [2, 4, 3, 9, 0, 0, 32, 33, 12, 7, 1, 34, 20, 31, 78, 143]},
    'round 3': {
        'num_p_enter': [0.0, 0.5, 0.0, 0.15, None, 0.0, 0.2, 0.18, 0.14, 0.75, 1.0, 0.83, 0.54, 0.87, 0.88, 0.89],
        'both_p_enter': [None, None, None, None, 0.0, None, 0.14, 0.5, None, 0.0, None, 0.57, 0.75, 0.75, 0.62, 0.88],
        'num_experts': [3, 4, 1, 13, 0, 1, 15, 11, 7, 4, 3, 18, 13, 45, 57, 196],
        'both_experts': [0, 0, 0, 0, 1, 0, 7, 2, 0, 1, 0, 7, 4, 12, 8, 17],
        'verbal_p_enter': [0.0, 0.0, None, 0.0, None, 0.2, 0.1, 0.21, 0.67, 0.43, None, 0.86, 0.78, 0.8, 0.79, 0.88],
        'verbal_experts': [6, 1, 0, 5, 0, 5, 10, 14, 6, 7, 0, 22, 18, 41, 62, 207]},
    'round 4': {
        'num_p_enter': [0.0, 0.0, 0.5, 0.25, 0.0, 1.0, 0.41, 0.4, 0.67, 0.67, 1.0, 0.62, 0.78, 0.81, 0.85, 0.9],
        'both_p_enter': [None, None, None, 0.0, None, None, 0.5, 0.0, 0.25, None, None, 0.0, 1.0, 0.78, 0.7, 0.96],
        'num_experts': [1, 6, 2, 4, 2, 1, 17, 10, 6, 6, 5, 26, 32, 47, 67, 158],
        'both_experts': [0, 0, 0, 3, 0, 0, 4, 2, 4, 0, 0, 2, 1, 9, 10, 24],
        'verbal_p_enter': [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.09, 0.27, 0.43, 1.0, 0.0, 0.6, 0.85, 0.78, 0.78, 0.85],
        'verbal_experts': [2, 5, 2, 5, 2, 1, 22, 15, 7, 2, 3, 15, 13, 32, 79, 202]},
    'round 5': {
        'num_p_enter': [0.25, 0.17, None, 0.0, 0.0, None, 0.25, 0.27, 0.5, 0.5, 1.0, 0.76, 0.7, 0.86, 0.84, 0.9],
        'both_p_enter': [0.0, 0.0, 0.0, 0.0, None, None, 0.0, None, 0.0, 0.0, None, 1.0, 0.67, 0.8, 0.67, 0.81],
        'num_experts': [4, 6, 0, 5, 3, 0, 20, 15, 6, 10, 1, 34, 23, 35, 76, 152],
        'both_experts': [1, 2, 1, 2, 0, 0, 1, 0, 2, 1, 0, 4, 3, 5, 12, 26],
        'verbal_p_enter': [0.0, 0.11, 0.0, 0.4, None, 0.0, 0.31, 0.21, 0.75, 0.83, 0.0, 0.62, 0.89, 0.85, 0.76, 0.86],
        'verbal_experts': [1, 9, 1, 5, 0, 2, 16, 19, 4, 6, 4, 24, 9, 53, 71, 180]},
    'round 6': {
        'num_p_enter': [0.5, 0.14, 0.0, 0.25, None, None, 0.35, 0.45, 0.71, 0.5, 0.67, 0.7, 0.6, 0.74, 0.85, 0.86],
        'both_p_enter': [0.0, None, None, None, None, None, 0.33, 0.33, 0.67, 1.0, None, 0.0, 1.0, 0.5, 0.85, 0.9],
        'num_experts': [4, 7, 1, 4, 0, 0, 31, 22, 7, 4, 3, 27, 15, 38, 71, 156],
        'both_experts': [1, 0, 0, 0, 0, 0, 3, 3, 3, 1, 0, 1, 4, 2, 13, 29],
        'verbal_p_enter': [0.0, 0.2, None, 0.0, 0.0, 0.0, 0.6, 0.1, 0.62, 0.86, 0.8, 0.45, 0.62, 0.79, 0.67, 0.81],
        'verbal_experts': [1, 5, 0, 4, 2, 2, 10, 10, 8, 7, 5, 33, 13, 34, 81, 193]},
    'round 7': {
        'num_p_enter': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.27, 0.36, 0.33, 0.0, 1.0, 0.65, 0.94, 0.84, 0.83, 0.92],
        'both_p_enter': [None, None, None, 0.0, None, None, 0.5, 0.0, 1.0, 0.0, None, 1.0, 0.5, 1.0, 1.0, 0.93],
        'num_experts': [2, 6, 1, 2, 1, 1, 11, 11, 6, 6, 1, 31, 18, 38, 75, 179],
        'both_experts': [0, 0, 0, 1, 0, 0, 4, 1, 1, 1, 0, 1, 2, 5, 14, 28],
        'verbal_p_enter': [None, 0.0, 0.0, 0.0, 0.0, 0.33, 0.33, 0.18, 0.67, 0.6, 0.0, 0.52, 0.64, 0.91, 0.75, 0.86],
        'verbal_experts': [0, 4, 1, 4, 2, 3, 9, 17, 3, 5, 3, 21, 14, 47, 73, 200]},
    'round 8': {
        'num_p_enter': [None, 0.0, 0.0, 0.29, None, 0.5, 0.25, 0.25, 0.5, 0.69, 0.8, 0.6, 0.58, 0.82, 0.66, 0.91],
        'both_p_enter': [0.0, 0.0, None, 0.0, None, None, 0.0, 0.0, None, 0.0, None, 1.0, 1.0, 1.0, 0.8, 0.88],
        'num_experts': [0, 7, 2, 7, 0, 2, 16, 16, 4, 13, 5, 15, 12, 40, 56, 192],
        'both_experts': [1, 1, 0, 2, 0, 0, 6, 1, 0, 1, 0, 1, 4, 12, 5, 25],
        'verbal_p_enter': [1.0, 0.0, 0.0, 0.0, None, 0.14, 0.15, 0.15, 0.56, 0.8, 0.33, 0.64, 0.62, 0.67, 0.76, 0.82],
        'verbal_experts': [1, 1, 1, 7, 0, 7, 26, 13, 9, 5, 3, 22, 16, 46, 58, 191]},
    'round 9': {
        'num_p_enter': [1.0, 0.0, 0.0, 0.17, None, 1.0, 0.21, 0.38, 0.5, 0.17, 0.67, 0.69, 0.82, 0.82, 0.9, 0.88],
        'both_p_enter': [None, None, None, None, None, None, 0.0, 0.0, 1.0, 0.0, 0.0, None, 0.5, 1.0, 1.0, 0.86],
        'num_experts': [1, 2, 1, 6, 0, 1, 14, 13, 8, 6, 3, 26, 17, 40, 52, 196],
        'both_experts': [0, 0, 0, 0, 0, 0, 1, 3, 2, 1, 1, 0, 4, 7, 11, 28],
        'verbal_p_enter': [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.08, 0.0, 0.33, 0.67, 0.2, 0.71, 0.65, 0.69, 0.77, 0.84],
        'verbal_experts': [1, 2, 2, 3, 2, 2, 24, 13, 3, 3, 5, 24, 23, 54, 62, 184]},
    'round 10': {
        'num_p_enter': [0.33, 0.0, 0.5, 0.38, 0.0, 0.0, 0.25, 0.43, 0.44, 0.29, 0.4, 0.67, 0.62, 0.84, 0.79, 0.89],
        'both_p_enter': [None, 0.0, None, None, None, None, 0.25, 0.0, None, None, None, 0.6, 1.0, 1.0, 0.85, 0.85],
        'num_experts': [3, 7, 2, 8, 1, 2, 12, 28, 9, 7, 5, 33, 8, 56, 67, 137],
        'both_experts': [0, 1, 0, 0, 0, 0, 4, 4, 0, 0, 0, 5, 1, 6, 13, 26],
        'verbal_p_enter': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.17, 0.21, 0.5, 0.43, 0.5, 0.73, 0.77, 0.86, 0.65, 0.84],
        'verbal_experts': [1, 3, 1, 2, 3, 3, 18, 29, 12, 7, 2, 22, 13, 43, 68, 182]},
}
colors = ['pink', 'forestgreen', 'crimson', 'darkblue']
markers = ["s", ".", "v", "1"]

# first 5 and last 5:
combine_est_8 = True
for rng, in_title in [[range(1, 6), 'First'], [range(6, 11), 'Last']]:
    fig100, ax100 = plt.subplots()
    num_p_enter_list = list()
    both_p_enter_list = list()
    num_experts_list = list()
    both_experts_list = list()
    verbal_p_enter_list = list()
    verbal_experts_list = list()

    for idx in range(len(x)):
        values = [rounds_dict[f'round {j}']['num_p_enter'][idx] for j in rng]
        values = [x for x in values if x is not None]
        num_p_enter_list.append(round(np.average(values), 2))
        num_experts_list.append(sum([rounds_dict[f'round {j}']['num_experts'][idx] for j in rng]))

        values = [rounds_dict[f'round {z}']['both_p_enter'][idx] for z in rng]
        values = [x for x in values if x is not None]
        both_p_enter_list.append(round(np.average(values), 2))
        both_experts_list.append(sum([rounds_dict[f'round {j}']['both_experts'][idx] for j in rng]))

        values = [rounds_dict[f'round {j}']['verbal_p_enter'][idx] for j in rng]
        values = [x for x in values if x is not None]
        verbal_p_enter_list.append(round(np.average(values), 2))
        verbal_experts_list.append(sum([rounds_dict[f'round {j}']['verbal_experts'][idx] for j in rng]))

    ax100.plot(x, num_p_enter_list, color=colors[1], label='Expert-both-DM-Number', marker=markers[1], linestyle='-')
    ax100.plot(x, both_p_enter_list, color=colors[3], label='Expert-both-DM-both', marker=markers[2], linestyle='-')
    ax100.plot(x, verbal_p_enter_list, color=colors[0], label='Expert-both-DM-Verbal', marker=markers[0], linestyle='-')

    plt.title(f"P(DM chose hotel) as a function of the experts\nnumerical estimate by Condition in {in_title} 5 rounds")
    plt.xlabel('Experts Numerical Estimate', fontsize=15)
    plt.ylabel('P(DM chose hotel)', fontsize=15)
    ax100.legend(loc='upper left', shadow=True, fontsize=8)
    plt.xticks(x)
    # Add a table at the bottom of the axes
    the_table = ax100.table(cellText=[x, num_experts_list, both_experts_list, verbal_experts_list],
                            rowLabels=['Experts Numerical Estimate', 'Expert-both-DM-Number', 'Expert-both-DM-both',
                                       'Expert-both-DM-Verbal'],
                            bbox=(0, -0.5, 1, 0.3))
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    # plt.yticks(range(4, 11))
    plt.show()
    fig100.savefig(f'P_enter_as_a_function_of_the_experts_numerical_estimate_by_Condition_in_{in_title}_5_rounds.png',
                   bbox_inches='tight')

    if combine_est_8:
        fig1000, ax1000 = plt.subplots()
        combine_dict = {
            'est<8': {'num_p_enter_list': [], 'both_p_enter_list': [], 'num_experts_list': [], 'both_experts_list': [],
                      'verbal_p_enter_list': [], 'verbal_experts_list': []},
            'est>8': {'num_p_enter_list': [], 'both_p_enter_list': [], 'num_experts_list': [], 'both_experts_list': [],
                      'verbal_p_enter_list': [], 'verbal_experts_list': []}
        }
        for idx, est in enumerate(x):
            if est < 8:
                if not np.isnan(num_p_enter_list[idx]):
                    combine_dict['est<8']['num_p_enter_list'].append(num_p_enter_list[idx])
                if not np.isnan(both_p_enter_list[idx]):
                    combine_dict['est<8']['both_p_enter_list'].append(both_p_enter_list[idx])
                if not np.isnan(verbal_p_enter_list[idx]):
                    combine_dict['est<8']['verbal_p_enter_list'].append(verbal_p_enter_list[idx])
                combine_dict['est<8']['num_experts_list'].append(num_experts_list[idx])
                combine_dict['est<8']['both_experts_list'].append(both_experts_list[idx])
                combine_dict['est<8']['verbal_experts_list'].append(verbal_experts_list[idx])
            else:
                if not np.isnan(num_p_enter_list[idx]):
                    combine_dict['est>8']['num_p_enter_list'].append(num_p_enter_list[idx])
                if not np.isnan(both_p_enter_list[idx]):
                    combine_dict['est>8']['both_p_enter_list'].append(both_p_enter_list[idx])
                if not np.isnan(verbal_p_enter_list[idx]):
                    combine_dict['est>8']['verbal_p_enter_list'].append(verbal_p_enter_list[idx])
                combine_dict['est>8']['num_experts_list'].append(num_experts_list[idx])
                combine_dict['est>8']['both_experts_list'].append(both_experts_list[idx])
                combine_dict['est>8']['verbal_experts_list'].append(verbal_experts_list[idx])

        num_p_enter_list = [round(np.average(combine_dict['est<8']['num_p_enter_list']), 2),
                            round(np.average(combine_dict['est>8']['num_p_enter_list']), 2)]
        both_p_enter_list = [round(np.average(combine_dict['est<8']['both_p_enter_list']), 2),
                             round(np.average(combine_dict['est>8']['both_p_enter_list']), 2)]
        verbal_p_enter_list = [round(np.average(combine_dict['est<8']['verbal_p_enter_list']), 2),
                               round(np.average(combine_dict['est>8']['verbal_p_enter_list']), 2)]
        num_experts_list = [np.sum(combine_dict['est<8']['num_experts_list']),
                            np.sum(combine_dict['est>8']['num_experts_list'])]
        both_experts_list = [np.sum(combine_dict['est<8']['both_experts_list']),
                             np.sum(combine_dict['est>8']['both_experts_list'])]
        verbal_experts_list = [np.sum(combine_dict['est<8']['verbal_experts_list']),
                               np.sum(combine_dict['est>8']['verbal_experts_list'])]
        ax1000.plot(['est<8', 'est>8'], num_p_enter_list, color=colors[1], label='Expert-both-DM-Number',
                    marker=markers[1], linestyle='-')
        ax1000.plot(['est<8', 'est>8'], both_p_enter_list, color=colors[3], label='Expert-both-DM-both',
                    marker=markers[2], linestyle='-')
        ax1000.plot(['est<8', 'est>8'], verbal_p_enter_list, color=colors[0], label='Expert-both-DM-Verbal',
                    marker=markers[0], linestyle='-')

        plt.title(
            f"P(DM chose hotel) as a function of the experts\nnumerical estimate by Condition in {in_title} 5 rounds")
        plt.xlabel('Experts Numerical Estimate', fontsize=15)
        plt.ylabel('P(DM chose hotel)', fontsize=15)
        ax1000.legend(loc='upper left', shadow=True, fontsize=8)
        # plt.xticks(['est<8', 'est>8'])
        # Add a table at the bottom of the axes
        num_experts_list = [(num_experts_list[idx], num_p_enter_list[idx]) for idx in
                            range(len(num_p_enter_list))]
        both_experts_list = [(both_experts_list[idx], both_p_enter_list[idx]) for idx in
                             range(len(both_p_enter_list))]
        verbal_experts_list = [(verbal_experts_list[idx], verbal_p_enter_list[idx]) for idx in
                               range(len(verbal_p_enter_list))]
        the_table = ax1000.table(
            cellText=[['est<8', 'est>8'], num_experts_list, both_experts_list, verbal_experts_list],
            rowLabels=['Experts Numerical Estimate', 'Expert-both-DM-Number', 'Expert-both-DM-both',
                       'Expert-both-DM-Verbal'], bbox=(0, -0.5, 1, 0.3))
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        plt.show()
        fig1000.savefig(
            f'P_enter_as_a_function_of_the_experts_numerical_estimate_by_Condition_in_{in_title}_5_rounds_est_8.png',
            bbox_inches='tight')


for rounds in rounds_dict.keys():
    fig100, ax100 = plt.subplots()
    num_p_enter = rounds_dict[rounds]['num_p_enter']
    both_p_enter = rounds_dict[rounds]['both_p_enter']
    ax100.plot(x, num_p_enter, color=colors[1], label='Expert-both-DM-Number', marker=markers[0], linestyle='-')
    ax100.plot(x, both_p_enter, color=colors[3], label='Expert-both-DM-both', marker=markers[2], linestyle='-')

    plt.title(f"P(DM chose hotel) as a function of the experts\nnumerical estimate by Condition in {rounds}")
    plt.xlabel('Experts Numerical Estimate', fontsize=15)
    plt.ylabel('P(DM chose hotel)', fontsize=15)
    ax100.legend(loc='upper left', shadow=True, fontsize=8)
    plt.xticks(x)
    # plt.yticks(range(4, 11))
    plt.show()
    fig100.savefig(f'P_enter_as_a_function_of_the_experts_numerical_estimate_by_Condition_in_{rounds}.png',
                   bbox_inches='tight')


"""score evaluation task"""
score_eval_data = pd.read_excel('/Users/reutapel/Documents/Documents/Technion/Msc/thesis/experiment/decision_prediction'
                                '/data_analysis/results/text_exp_2_tests/score evaluation task.xlsx',
                                sheet_name='data_to_plot')
reviews = score_eval_data.review_id.unique()
colors = (list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.keys()))
colors.remove('tab:olive')
colors.remove('y')
colors.remove('w')
colors = colors*5
# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
fig, ax = plt.subplots()
all_round_num = score_eval_data.average_answer.round(1).tolist()
all_expert_score = score_eval_data.review_real_score.round(1).tolist()
all_x_min = score_eval_data.min_answer.tolist()
all_x_max = score_eval_data.max_answer.tolist()
all_x_avg = list()
all_y = list()
points_annotate = list()
move = False
for i, point in enumerate(all_round_num):
    y = all_expert_score[i]
    y_loc = y
    x_loc = point
    if point == 6.5 and y == 10 and [6.5, 10] not in points_annotate:
        y_loc = 10.2
        points_annotate.append([6.5, 10])
    elif point == 6.6 and y == 10 and [6.6, 10] not in points_annotate:
        y_loc = 9.8
        points_annotate.append([6.6, 10])
    # elif average_score == 8.1 and y == 10 and [8.1, 10] not in points_annotate:
    #     x_loc = 7.9
    #     points_annotate.append([8.1, 10])
    elif point == 8.3 and y == 10 and [8.3, 10] not in points_annotate:
        y_loc = 9.8
        points_annotate.append([8.3, 10])
    elif point == 8.5 and y == 10 and [8.5, 10] not in points_annotate:
        y_loc = 10.2
        points_annotate.append([8.5, 10])
    elif point == 8.9 and y == 10 and [8.9, 10] not in points_annotate:
        y_loc = 9.75
        points_annotate.append([8.9, 10])
    elif point == 8.7 and y == 10 and [8.7, 10] not in points_annotate:
        y_loc = 9.75
        x_loc = 8.65
        points_annotate.append([8.7, 10])
    elif point == 8.3 and y == 9.6 and [8.3, 9.6] not in points_annotate:
        x_loc = 8.1
        points_annotate.append([8.3, 9.6])
    elif point == 8.4 and y == 9.6 and [8.4, 9.6] not in points_annotate:
        y_loc = 9.35
        points_annotate.append([8.4, 9.6])
    elif point == 7.3 and y == 9.6 and [7.3, 9.6] not in points_annotate:
        x_loc = 7.1
        points_annotate.append([7.3, 9.6])
    elif point == 7.4 and y == 9.6 and [7.4, 9.6] not in points_annotate:
        y_loc = 9.35
        points_annotate.append([7.4, 9.6])
    elif point == 7.3 and y == 9.6 and [7.31, 9.6] not in points_annotate:
        y_loc = 9.35
        x_loc = 7.1
        points_annotate.append([7.31, 9.6])
    elif point == 6.1 and y == 9.6 and [6.1, 9.6] not in points_annotate:
        y_loc = 9.35
        x_loc = 5.9
        points_annotate.append([6.1, 9.6])
    elif point == 6.1 and y == 9.6 and [6.15, 9.6] not in points_annotate:
        x_loc = 5.9
        points_annotate.append([6.15, 9.6])
    elif point == 7.1 and y == 9.2 and [7.1, 9.2] not in points_annotate:
        y_loc = 9
        points_annotate.append([7.1, 9.2])
    elif point == 7.7 and y == 9.2 and [7.7, 9.2] not in points_annotate:
        x_loc = 7.5
        points_annotate.append([7.7, 9.2])
    elif point == 8.5 and y == 9.2 and [8.5, 9.2] not in points_annotate:
        y_loc = 8.95
        points_annotate.append([8.5, 9.2])
    elif point == 6.4 and y == 8.8 and [6.4, 8.8] not in points_annotate:
        y_loc = 8.55
        x_loc = 6.2
        points_annotate.append([6.4, 8.8])
    elif point == 7.3 and y == 8.3 and [7.3, 8.3] not in points_annotate:
        y_loc = 8.05
        x_loc = 7.1
        points_annotate.append([7.3, 8.3])
    elif point == 3.5 and y == 3.3 and [3.5, 3.3] not in points_annotate:
        x_loc = 3.3
        points_annotate.append([3.5, 3.3])
    elif point == 4.3 and y == 3.3 and [4.3, 3.3] not in points_annotate:
        x_loc = 4.0
        points_annotate.append([4.3, 3.3])
    else:
        points_annotate.append([x_loc, y])
    ax.annotate(f'({all_x_min[i]},{all_x_max[i]})', (x_loc, y_loc), color=colors[i], fontsize=6)
    # ax.annotate(f'({min(x)},{y[0]})', (min(x), y[0]), color=colors[i])
    # ax.annotate(f'({max(x)},{y[0]})', (max(x), y[0]), color=colors[i])

ax.scatter([all_round_num], [all_expert_score], color='black', marker=".")
# Add a table at the right size of the axes
# cell_text = list()
# cell_text.append(all_y)
# cell_text.append(all_x_avg)
# cell_text.append(all_x_min)
# cell_text.append(all_x_max)
# the_table = plt.table(cellText=cell_text,
#                       rowLabels=['Original Score', 'Average Evaluation', 'Min Evaluation', 'Max Evaluation'],
#                       loc='bottom')
# the_table.auto_set_font_size(False)
# the_table.set_fontsize(8)


plt.title('How Well do Humans Understand Text?')
plt.xlabel('Participants Average Evaluation')
plt.ylabel('Reviews Original Score')
plt.xticks(range(3, 11))
# plt.tick_params(
#     axis='x',  # changes apply to the x-axis
#     which='both',  # both major and minor ticks are affected
#     bottom=False,  # ticks along the bottom edge are off
#     top=True,  # ticks along the top edge are on
#     labelbottom=False,  # labels along the bottom edge are off
#     labeltop=True)  # labels along the top edge are on
plt.yticks(range(3, 11))
plt.show()
fig.savefig('score evaluation task.png', bbox_inches='tight')

"""Linear Regression"""
linear = LinearRegression()
all_round_num = np.array(all_round_num)
linear.fit(all_round_num.reshape(-1, 1), all_expert_score)
# Make predictions using the testing set
pred = linear.predict(all_round_num.reshape(-1, 1))
# The coefficients
print('Coefficients: \n', linear.coef_)
# The mean squared error
print('Root Mean squared error: %.2f'
      % math.sqrt(mean_squared_error(all_expert_score, pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(all_expert_score, pred))

# Plot outputs
plt.scatter(all_round_num, all_expert_score, color='black')
plt.plot(all_round_num, pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

"""The Communication Type Effect on the Experts Cheating Level"""
colors = ['pink', 'forestgreen', 'crimson', 'darkblue']
markers = [".", "x", "v", "1"]
fig1, ax1 = plt.subplots()
ax1.axis([4, 10, 4, 10])
x = [4.17, 6.66, 7.44, 7.97, 8.11, 8.33, 8.94, 9.19, 9.54, 9.77]
num_cl = [5.21, 8.45, 8.87, 9.13, 9.33, 9.59, 9.58, 9.59, 9.8, 9.86]
verbal_cl = [5.45, 8.51, 9.0, 9.21, 9.37, 9.56, 9.57, 9.64, 9.82, 9.87]
num_only_cl = [5.02, 8.16, 8.45, 8.89, 8.82, 8.95, 9.59, 9.38, 9.65, 9.75]
both_cl = [5.24, 8.58, 8.89, 9.24, 9.44, 9.48, 9.29, 9.68, 9.79, 9.85]

# ax1.plot(x, verbal_cl, color=colors[0], label='Verbal', marker=markers[0], linestyle='-')
ax1.plot(x, num_cl, color=colors[1], label='Expert-both-DM-Number', marker=markers[0], linestyle='-')
ax1.plot(x, num_only_cl, color=colors[2], label='Expert-Number-DM-Number', marker=markers[1], linestyle='-')
ax1.plot(x, both_cl, color=colors[3], label='Expert-both-DM-both', marker=markers[2], linestyle='-')

ax1.plot(x, x, color='darkviolet', marker=markers[0], linestyle='-', label='Truth Telling')

# for hotel in range(1, 11):
#     hotel_list = score_eval_data.loc[score_eval_data.hotel_id == hotel].answer_real.round(1).tolist()
#     if hotel != 4:
#         y_gap = 0
#     else:
#         y_gap = -0.1
#     ax1.text(x[hotel-1]+0.1, x[hotel-1]+y_gap, f'avg:{round(sum(hotel_list)/len(hotel_list), 1)}:{hotel_list}',
#              {'fontsize': 8})

# plt.title("The Selected Score as a Function of the Hotels' Average Score by Condition")
plt.xlabel('Decision Maker Expected Payoff', fontsize=15)
plt.ylabel('Expert Average Signal', fontsize=15)
ax1.legend(loc='upper left', shadow=True, fontsize=8)
plt.xticks(range(4, 11))
plt.yticks(range(4, 11))
plt.show()
fig1.savefig('The_Communication_Type_Effect_on_the_Experts_Cheating_Level.png', bbox_inches='tight')

"""Decision maker average payoff"""
verbal_dmap = [0.29, 0.14, 0.44, 0.52, 0.21, 0.43, 0.31, 0.27, 0.19, 0.16]
numerical_dmap = [0.41, 0.17, 0.48, 0.1, 0.19, 0.17, 0.28, 0.41, 0.38, -0.09]
num_only_dmap = [0.06, 0.3, 0.28, 0.08, 0.34, 0.53, 0.12, 0.64, -0.03, 0.32]
both_dmap = [0.01, 0.36, 0.14, 0.36, 0.46, 0.46, 0.31, -0.21, 0.24, 0.55]

index = list(range(1, 11))
decision_data = pd.DataFrame(
    {f'Verbal: average payoff: {round(sum(verbal_dmap)/ len(verbal_dmap), 2)}': verbal_dmap,
     f'Numerical: average payoff: {round(sum(numerical_dmap)/ len(numerical_dmap), 2)}': numerical_dmap,
     f'Only Numeric: average payoff: {round(sum(num_only_dmap)/ len(num_only_dmap), 2)}': num_only_dmap,
     f'Numeric + Verbal: {round(sum(both_dmap)/ len(both_dmap), 2)}': both_dmap}, index=index)
plt.figure(figsize=(10, 5))
ax2 = decision_data.plot(kind="bar", stacked=False, rot=0, figsize=(10, 5),
                         color=['forestgreen', 'darkblue', 'crimson', 'pink'])
plt.title("The Decision Makers' Average Payoff Throughout the Experiment")
plt.xlabel('Round Number')
plt.ylabel("Decision Makers' Average Payoff")
rects = ax2.patches
autolabel(rects, ax2, rotation='horizontal', max_height=0.52, convert_to_int=False)
ax2.legend(loc='lower center', shadow=True)
plt.show()
fig_to_save = ax2.get_figure()
fig_to_save.savefig('Decision maker average payoff.png', bbox_inches='tight')

"""Decision maker average expected payoff"""
verbal_dmaep = [0.31, 0.11, 0.38, 0.47, 0.2, 0.3, 0.32, 0.28, 0.21, 0.12]
numerical_dmaep = [0.22, 0.26, 0.53, 0.21, 0.2, 0.18, 0.32, 0.35, 0.36, -0.04]
num_only_dmaep = [0.28, 0.37, 0.15, 0.11, 0.23, 0.42, 0.29, 0.59, 0.15, 0.2]
both_dmaep = [0.26, 0.36, 0.03, 0.07, 0.36, 0.48, 0.5, 0.01, 0.42, 0.47]

index = list(range(1, 11))
decision_data = pd.DataFrame({
    f'Verbal: average expected payoff: {round(sum(verbal_dmaep)/ len(verbal_dmaep), 2)}': verbal_dmaep,
    f'Numerical: average expected payoff: {round(sum(numerical_dmaep)/ len(numerical_dmaep), 2)}': numerical_dmaep,
    f'Only Numeric: average expected payoff: {round(sum(num_only_dmaep)/ len(num_only_dmaep), 2)}': num_only_dmaep,
    f'Numeric + Verbal: average expected payoff: {round(sum(both_dmaep)/ len(both_dmaep), 2)}': both_dmaep},
    index=index)
plt.figure(figsize=(10, 5))
ax2 = decision_data.plot(kind="bar", stacked=False, rot=0, figsize=(10, 5),
                         color=['forestgreen', 'darkblue', 'crimson', 'pink'])
plt.title("The Decision Makers' Average Expected Payoff Throughout the Experiment")
plt.xlabel('Round Number')
plt.ylabel("Decision Makers' Average Expected Payoff")
rects = ax2.patches
autolabel(rects, ax2, rotation='horizontal', max_height=0.52, convert_to_int=False)
ax2.legend(loc='lower center', shadow=True)
plt.show()
fig_to_save = ax2.get_figure()
fig_to_save.savefig('Decision maker average expected payoff.png', bbox_inches='tight')


# plot
fig1, ax10 = plt.subplots()
# ax10.axis([4, 10, 4, 10])
ax10.plot(index, numerical_dmaep, color=colors[1], label='Expert-both-DM-Number', marker=markers[0], linestyle='-')
ax10.plot(index, num_only_dmaep, color=colors[2], label='Expert-Number-DM-Number', marker=markers[1], linestyle='-')
ax10.plot(index, both_dmaep, color=colors[3], label='Expert-both-DM-both', marker=markers[2], linestyle='-')

# plt.title("The Decision Makers' Average Expected Payoff Throughout the Experiment")
plt.xlabel('Round Number', fontsize=10)
plt.ylabel("Decision Makers' Average Expected Payoff", fontsize=10)
ax10.legend(loc='upper right', shadow=True, fontsize=8)
# plt.xticks(range(4, 11))
plt.xticks(index)
plt.show()
fig1.savefig('Decision_maker_average_expected_payoff_graph.png', bbox_inches='tight')

"""Expert average payoff"""
verbal_eap = [77, 66, 77, 71, 74, 70, 75, 67, 68, 68]
numerical_eap = [88, 79, 76, 78, 75, 72, 78, 73, 78, 71]
num_only_eap = [97, 90, 70, 71, 75, 70, 87, 78, 57, 85]
both_eap = [79, 73, 64, 69, 65, 78, 86, 72, 79, 73]
index = list(range(1, 11))
decision_data = pd.DataFrame({f'Verbal: average percentage: {sum(verbal_eap)/ len(verbal_eap)}': verbal_eap,
                              f'Numerical: average percentage: {sum(numerical_eap)/ len(numerical_eap)}': numerical_eap,
                              f'Only Numeric: average percentage: {sum(num_only_eap)/ len(num_only_eap)}': num_only_eap,
                              f'Numeric + Verbal: average percentage: {sum(both_eap)/ len(both_eap)}': both_eap},
                             index=index)
plt.figure(figsize=(10, 5))
ax2 = decision_data.plot(kind="bar", stacked=False, rot=0, figsize=(10, 5),
                         color=['forestgreen', 'darkblue', 'crimson', 'pink'])
plt.title("Percentage of Decision Makers that Chose the Hotel Option Throughout the Experiment", fontsize=15)
plt.xlabel('Trial Number', fontsize=15)
plt.ylabel("% Decision Makers", fontsize=15)
rects = ax2.patches
ax2.legend(loc='lower center', shadow=True)
autolabel(rects, ax2, rotation='horizontal', max_height=89, convert_to_int=False, fontsize=10)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
plt.show()
fig_to_save = ax2.get_figure()
fig_to_save.savefig('Expert average payoff.png', bbox_inches='tight')

# plot
fig1, ax20 = plt.subplots()
numerical_eap = [value/100 for value in numerical_eap]
num_only_eap = [value/100 for value in num_only_eap]
both_eap = [value/100 for value in both_eap]
ax20.plot(index, numerical_eap, color=colors[1], label='Expert-both-DM-Number', marker=markers[0], linestyle='-')
ax20.plot(index, num_only_eap, color=colors[2], label='Expert-Number-DM-Number', marker=markers[1], linestyle='-')
ax20.plot(index, both_eap, color=colors[3], label='Expert-both-DM-both', marker=markers[2], linestyle='-')

# plt.title("The Experts Average Payoff Throughout the Experiment")
plt.xlabel('Round Number', fontsize=10)
plt.ylabel("Experts Average Payoff", fontsize=10)
ax20.legend(loc='upper right', shadow=True, fontsize=8)
# plt.xticks(range(4, 11))
plt.xticks(index)
plt.show()
fig1.savefig('Expert_average_payoff_graph.png', bbox_inches='tight')


"""Linear Regression num-num_only"""
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()

Coefficients_dmaep = list()
rsqrt_dmeap = list()
Coefficients_sum = list()
rsqrt_sum = list()
for name, dmaep, eap, color in [['Numerical', numerical_dmaep, numerical_eap, 'darkblue'],
                                ['Numeric + Verbal', both_dmaep, both_eap, 'pink']]:
    cond_sum = [i + j for i, j in zip(dmaep, eap)]
    linear_dmaep = LinearRegression()
    index_np = np.array(index)
    index_np = index_np.reshape(-1, 1)
    linear_dmaep.fit(index_np, dmaep)
    # Make predictions using the testing set
    pred = linear_dmaep.predict(index_np)
    # The coefficients
    Coefficients_dmaep.append(round(linear_dmaep.coef_[0], 2))
    # The mean squared error
    rsqrt_dmeap.append(round(math.sqrt(mean_squared_error(dmaep, pred)), 2))
    # The coefficient of determination: 1 is perfect prediction
    # r2_score = r2_score(dmaep, pred)

    linear_sum = LinearRegression()
    linear_sum.fit(index_np, cond_sum)
    # Make predictions using the testing set
    pred_sum = linear_sum.predict(index_np)
    # The coefficients
    Coefficients_sum.append(round(linear_sum.coef_[0], 2))
    # The mean squared error
    rsqrt_sum.append(round(math.sqrt(mean_squared_error(cond_sum, pred_sum)), 2))

    # Plot outputs
    ax4.scatter(index, dmaep, color=color)
    ax4.plot(index, pred, color=color, linewidth=3, label=name)

    ax5.scatter(index, cond_sum, color=color)
    ax5.plot(index, pred_sum, color=color, linewidth=3, label=name)

sum_text_list = list()
dmaep_text_list = list()

for i, name in enumerate(['Numerical', 'Numeric + Verbal']):
    dmaep_text_list.append(f'{name}: Coefficients: {Coefficients_dmaep[i]}, RMSE: {rsqrt_dmeap[i]}')
    sum_text_list.append(f'{name}: Coefficients: {Coefficients_sum[i]}, RMSE: {rsqrt_sum[i]}')

sum_my_text = '\n'.join(sum_text_list)
dmaep_my_text = '\n'.join(dmaep_text_list)

ax4.text(0.01, 0.95, dmaep_my_text, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
         color='black', fontsize=10)
ax5.text(0.95, 0.01, sum_my_text, verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,
         color='black', fontsize=10)
ax4.legend()
ax5.legend()
ax4.set_xticks(index)
ax5.set_xticks(index)

ax4.set_yticks(list(np.arange(0, 0.8, 0.1)))
ax5.set_yticks(list(range(50, 101, 10)))

ax4.set_title('DM EV Linear Trend')
ax5.set_title('Sum of profit (Expert+DM EV) EV Linear Trend')

plt.show()
fig_to_save = ax4.get_figure()
fig_to_save.savefig('DM EV Linear Trend.png', bbox_inches='tight')
fig_to_save = ax5.get_figure()
fig_to_save.savefig('Sum of profit (Expert+DM EV) EV Linear Trend.png', bbox_inches='tight')


"""Performance graph"""
accuracy = [80.69, 79.6, 79.12, 78.95, 71.32, 71.84]
f_score_positive = [87.13, 86.27, 86.29, 86.45, 83.15, 83.61]
f_score_negative = [71.36, 60.31, 56.16, 52.89, 61.6, 0]
index = ['LC_DF_CH',
         'LC_B_CH',
         'LC_B_C',
         'NN_DF_CH',
         'LC_DF_H',
         'Most Frequent\nclass Baseline']

decision_data = pd.DataFrame({'Accuracy': accuracy,
                              'F1-Score (Positive Class)': f_score_positive,
                              'F1-Score (Negative Class)': f_score_negative}, index=index)
ax3 = decision_data.plot(kind="bar", stacked=False, rot=0, color=['forestgreen', 'darkblue', 'crimson'], width=0.8,
                         figsize=(10, 5))
plt.title("Performance of Decision Maker's Decisions Predictions", fontsize=15)
plt.xlabel('Model-Features Set Combination', fontsize=15)
rects = ax3.patches
autolabel(rects, ax3, rotation='horizontal', max_height=88, convert_to_int=False, add_0_label=True, fontsize=10)
# ax3.set_xticklabels(labels=index, rotation=70, rotation_mode="anchor", ha="right")
ax3.legend(loc='lower center', shadow=True)
ax3.set_yticks([0, 20, 40, 60, 80, 100])
plt.show()
fig_to_save = ax3.get_figure()
fig_to_save.savefig('predictions performances.png', bbox_inches='tight')


"""Check significant"""
numeric_data = pd.read_csv('/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/data_analysis/'
                           'analysis/text_exp_2_tests/numeric/linear_scores.csv')
both_data = pd.read_csv('/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/data_analysis/'
                        'analysis/text_exp_2_tests/both/linear_scores.csv')

linear_score_dm_ev_statistic = scipy.stats.f_oneway(numeric_data.linear_score_dm_ev, both_data.linear_score_dm_ev)
print('linear_score_dm_ev')
print(np.var(numeric_data.linear_score_dm_ev), np.var(both_data.linear_score_dm_ev))
linear_score_dm_ev_kruskal = scipy.stats.kruskal(numeric_data.linear_score_dm_ev, both_data.linear_score_dm_ev)
print(f'ANOVA test: {linear_score_dm_ev_statistic},\nKruskal test: {linear_score_dm_ev_kruskal}')

linear_expert_payoff_statistic = scipy.stats.f_oneway(numeric_data.linear_expert_payoff, both_data.linear_expert_payoff)
print('linear_expert_payoff')
print(np.var(numeric_data.linear_expert_payoff), np.var(both_data.linear_expert_payoff))
linear_expert_payoff_kruskal = scipy.stats.kruskal(numeric_data.linear_expert_payoff, both_data.linear_expert_payoff)
print(f'ANOVA test: {linear_expert_payoff_statistic},\nKruskal test: {linear_expert_payoff_kruskal}')

linear_sum_statistic = scipy.stats.f_oneway(numeric_data.linear_sum, both_data.linear_sum)
print('linear_sum')
print(np.var(numeric_data.linear_sum), np.var(both_data.linear_sum))
linear_sum_kruskal = scipy.stats.kruskal(numeric_data.linear_sum, both_data.linear_sum)
print(f'ANOVA test: {linear_sum_statistic},\nKruskal test: {linear_sum_kruskal}')
