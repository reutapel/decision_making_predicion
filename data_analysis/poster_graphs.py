import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
from data_analysis import autolabel
import os


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


"""score evaluation task"""
score_eval_data = pd.read_excel('/Users/reutapel/Documents/Technion/Msc/thesis/experiment/decision_prediction/'
                                'data_analysis/results/text_exp_2_tests/score evaluation task.xlsx',
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
colors = ['forestgreen', 'darkblue', 'crimson', 'pink']
markers = [".", "x", "+", "1"]
fig1, ax1 = plt.subplots()
ax1.axis([4, 10, 4, 10])
x = [4.17, 6.66, 7.44, 7.97, 8.11, 8.33, 8.94, 9.19, 9.54, 9.77]
num = [5.21, 8.45, 8.87, 9.13, 9.33, 9.59, 9.58, 9.59, 9.8, 9.86]
verbal = [5.45, 8.51, 9.0, 9.21, 9.37, 9.56, 9.57, 9.64, 9.82, 9.87]
num_only = [5.02, 8.16, 8.45, 8.89, 8.82, 8.95, 9.59, 9.38, 9.65, 9.75]
both = [5.12, 8.48, 8.94, 9.35, 9.13, 9.64, 9.01, 9.6, 9.83, 9.83]

ax1.plot(x, verbal, color=colors[0], label='Verbal', marker=markers[0], linestyle='-')
ax1.plot(x, num, color=colors[1], label='Numerical', marker=markers[0], linestyle='-')
ax1.plot(x, num_only, color=colors[2], label='Only Numeric', marker=markers[0], linestyle='-')
ax1.plot(x, both, color=colors[3], label='Numeric + Verbal', marker=markers[0], linestyle='-')

ax1.plot(x, x, color='darkviolet', marker=markers[0], linestyle='-', label='Truth Telling')

# for hotel in range(1, 11):
#     hotel_list = score_eval_data.loc[score_eval_data.hotel_id == hotel].answer_real.round(1).tolist()
#     if hotel != 4:
#         y_gap = 0
#     else:
#         y_gap = -0.1
#     ax1.text(x[hotel-1]+0.1, x[hotel-1]+y_gap, f'avg:{round(sum(hotel_list)/len(hotel_list), 1)}:{hotel_list}',
#              {'fontsize': 8})

plt.title("The Experts' Cheating Level in All Experiment Conditions")
plt.xlabel('Decision Maker Expected Payoff', fontsize=15)
plt.ylabel('Expert Average Signal', fontsize=15)
ax1.legend(loc='upper left', shadow=True, fontsize=8)
plt.xticks(range(4, 11))
plt.yticks(range(4, 11))
plt.show()
fig1.savefig('The Communication Type Effect on the Experts Cheating Level.png', bbox_inches='tight')

"""Decision maker average payoff"""
verbal = [0.29, 0.14, 0.44, 0.52, 0.21, 0.43, 0.31, 0.27, 0.19, 0.16]
numerical = [0.41, 0.17, 0.48, 0.1, 0.19, 0.17, 0.28, 0.41, 0.38, -0.09]
num_only = [0.06, 0.3, 0.28, 0.08, 0.34, 0.53, 0.12, 0.64, -0.03, 0.32]
both = [0.62, 0.56, 0.25, -0.06, 0.67, 0.69, 0.33, -1.19, 0.14, 0.43]

index = list(range(1, 11))
decision_data = pd.DataFrame({f'Verbal: average payoff: {round(sum(verbal)/ len(verbal), 2)}': verbal,
                              f'Numerical: average payoff: {round(sum(numerical)/ len(numerical), 2)}': numerical,
                              f'Only Numeric: average payoff: {round(sum(num_only)/ len(num_only), 2)}': num_only,
                              f'Numeric + Verbal: {round(sum(both)/ len(both), 2)}': both},
                             index=index)
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
verbal = [0.31, 0.11, 0.38, 0.47, 0.2, 0.3, 0.32, 0.28, 0.21, 0.12]
numerical = [0.22, 0.26, 0.53, 0.21, 0.2, 0.18, 0.32, 0.35, 0.36, -0.04]
num_only = [0.28, 0.37, 0.15, 0.11, 0.23, 0.42, 0.29, 0.59, 0.15, 0.2]
both = [0.72, 0.51, 0.03, -0.23, 0.97, 0.76, 0.16, -0.47, 0.32, 0.25]

index = list(range(1, 11))
decision_data = pd.DataFrame({
    f'Verbal: average expected payoff: {round(sum(verbal)/ len(verbal), 2)}': verbal,
    f'Numerical: average expected payoff: {round(sum(numerical)/ len(numerical), 2)}': numerical,
    f'Only Numeric: average expected payoff: {round(sum(num_only)/ len(num_only), 2)}': num_only,
    f'Numeric + Verbal: average expected payoff: {round(sum(both)/ len(both), 2)}': both}, index=index)
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

"""Expert average payoff"""
verbal = [77, 66, 77, 71, 74, 70, 75, 67, 68, 68]
numerical = [88, 79, 76, 78, 75, 72, 78, 73, 78, 71]
num_only = [97, 90, 70, 71, 75, 70, 87, 78, 57, 85]
both = [85, 71, 50, 42, 78, 78, 84, 78, 57, 71]
index = list(range(1, 11))
decision_data = pd.DataFrame({f'Verbal: average percentage: {sum(verbal)/ len(verbal)}': verbal,
                              f'Numerical: average percentage: {sum(numerical)/ len(numerical)}': numerical,
                              f'Only Numeric: average percentage: {sum(num_only)/ len(num_only)}': num_only,
                              f'Numeric + Verbal: average percentage: {sum(both)/ len(both)}': both}, index=index)
plt.figure(figsize=(10, 5))
ax2 = decision_data.plot(kind="bar", stacked=False, rot=0, figsize=(10, 5),
                         color=['forestgreen', 'darkblue', 'crimson', 'pink'])
plt.title("Percentage of Decision Makers that Chose the Hotel Option Throughout the Experiment", fontsize=15)
plt.xlabel('Round Number', fontsize=15)
plt.ylabel("% Decision Makers", fontsize=15)
rects = ax2.patches
ax2.legend(loc='lower center', shadow=True)
autolabel(rects, ax2, rotation='horizontal', max_height=89, convert_to_int=False, fontsize=10)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
plt.show()
fig_to_save = ax2.get_figure()
fig_to_save.savefig('Expert average payoff.png', bbox_inches='tight')


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
