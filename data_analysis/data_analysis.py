import pandas as pd
import os
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from matplotlib.font_manager import FontProperties
import math


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'results')
orig_data_analysis_directory = os.path.join(base_directory, 'analysis')
date_directory = 'all'
log_file_name = os.path.join(orig_data_analysis_directory, date_directory,
                             datetime.now().strftime('LogFile_data_analysis_%d_%m_%Y_%H_%M_%S.log'))

split_gender = False

data_analysis_directory = os.path.join(orig_data_analysis_directory, date_directory)
if split_gender:
    data_analysis_directory = os.path.join(data_analysis_directory, 'per_gender')
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


def create_bar_from_df(data: pd.DataFrame, title: str='', xlabel: str='', ylabel: str='',
                       curr_date_directory: str=date_directory, add_table: bool=True, rot: bool=True,
                       add_point: float=None, add_text_label: bool=False, max_height=100, convert_to_int=True,
                       axes_bounds: list=None, label_rotation='vertical', stacked=True, y_ticks_list: list=None,
                       figsize: tuple=(20, 5), autolabel_fontsize=6, autolabel_text: list=None):
    """
    This function create bar plot from data frame and adding table with values ad the bottom
    :param pd.DataFrame data: the data to plot
    :param str title: the plot title
    :param str xlabel: the label of x axe
    :param str ylabel: the label of y axe
    :param str curr_date_directory: the name of the date_directory if call this function from another file
    :param bool add_table: if to add table with the data values, if add table- the columns need to be ints
    :param bool rot: it to rotate xticks
    :param add_point: float: add line in a specific point
    :param add_text_label: if we want to add text label to the graph
    :param max_height: the max height to use when adding text
    :param convert_to_int: convert height to int it height>1 when adding text
    :param axes_bounds: if we have bounds for the axes
    :param label_rotation: the rotation of the label to print
    :param stacked: if more than 1 column and want the columns to be near each other and not on each other - put False
    :param y_ticks_list: list of ticks for the y axis
    :param figsize: the figure size
    :param autolabel_fontsize: the fontsize of the auto labels
    :param autolabel_text: if we want to pass specific text to autolabel
    :return:
    """
    print('Create plot from DF for', title)
    plt.figure(figsize=figsize)

    # Plot bars and create text labels for the table
    if add_table:
        ax = data.plot(kind="bar", stacked=True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        rows = data.columns
        n_rows = len(rows)
        cell_text = []
        for row in range(n_rows, 0, -1):
            cell_text.append([x for x in data[row]])

        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=cell_text,
                              rowLabels=list(range(n_rows, 0, -1)),
                              bbox=(0, -0.85, 1, 0.5))
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)

    else:
        # colors = ['orange', 'seagreen', 'navy', 'orchid', 'dimgray', 'purple', 'skyblue', 'yellow', 'black',
        #           'orangered']
        # plot_colors = [colors[i] for i in data.index]
        if rot:
            ax = data.plot(kind="bar", stacked=stacked, rot=1, figsize=figsize)  # , colormap=plot_colors)
        else:
            ax = data.plot(kind="bar", stacked=stacked, figsize=figsize)  # , colormap=plot_colors)

    if add_point is not None:
        ax.axvline(x=add_point, linewidth=1, color='black')
        ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        ax.tick_params(axis="x", labelsize=8)

    if add_text_label:
        rects = ax.patches
        autolabel(rects, ax, rotation=label_rotation, max_height=max_height, convert_to_int=convert_to_int,
                  fontsize=autolabel_fontsize, text=autolabel_text)

    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    if y_ticks_list is not None:
        plt.yticks(y_ticks_list)
    if axes_bounds is not None:
        ax.axis(axes_bounds, fontsize=20)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize('x-small')
    fig_to_save = ax.get_figure()
    # if add_table:
    fig_to_save.savefig(os.path.join(curr_date_directory, title + '.png'), bbox_inches='tight')
    # else:
    #     fig_to_save.savefig(os.path.join(curr_date_directory, title + '.png'))

    return


def create_data_for_plot_all_experts(data_to_plot, xlabel, ylabel, legend, title, batch_average=None, line_numbers=None,
                                     batch_with_line=True):
    """
    This function create the data for plot points. Split the data based on legend and create a plot for each of them.
    Plot for all points- took and didn't take (the DM decision)
    :param data_to_plot: data frame: the data to plot. must have at least 3 columns: xlabel, ylabel, prob_status
    :param str xlabel: the column name to use as the x axis
    :param str ylabel: he column name to use as the y axis
    :param dict line_numbers: number of points below, above and on the line
    :param list legend: list of the categories names.
    :param str title: the plot title
    :param dict batch_average: Keys are str(low) to str(high) of the batch.
    Values are: the average sender answer in this batch
    :param bool batch_with_line: if the batch numbers are regarding the line or not
    :return:
    """
    # both took and didn't take. All experts on one plot.
    x_points_took = list()
    y_points_took = list()

    other = data_to_plot.loc[data_to_plot.prob_status == legend[0]]
    if not other.empty:
        x_points_took.append(list(other[xlabel]))
        y_points_took.append(list(other[ylabel]))

    zero_one_accurate = data_to_plot.loc[data_to_plot.prob_status == legend[1]]
    if not zero_one_accurate.empty:
        x_points_took.append(list(zero_one_accurate[xlabel]))
        y_points_took.append(list(zero_one_accurate[ylabel]))

    zero_one_not_accurate = data_to_plot.loc[data_to_plot.prob_status == legend[2]]
    if not zero_one_not_accurate.empty:
        x_points_took.append(list(zero_one_not_accurate[xlabel]))
        y_points_took.append(list(zero_one_not_accurate[ylabel]))

    accurate = data_to_plot.loc[data_to_plot.prob_status == legend[3]]
    if not accurate.empty:
        x_points_took.append(list(accurate[xlabel]))
        y_points_took.append(list(accurate[ylabel]))

    create_point_plot(x_points_took, y_points_took, legend=legend, xlabel=xlabel, ylabel=ylabel, title=title,
                      line_numbers=line_numbers, batch_with_line=batch_with_line, batch_average=batch_average)

    return


def create_data_for_plot_2_took_didnt_take(data_to_plot, xlabel, ylabel, legend, title, batch_average=None,
                                           line_numbers=None, batch_with_line=False):
    """
    This function create the data for plot points. Split the data based on legend and create a plot for each of them.
    Split the data to DM took or didn't take the lottery.
    :param data_to_plot: data frame: the data to plot
    :param str xlabel: the column name to use as the x axis
    :param str ylabel: he column name to use as the y axis
    :param list[dict] line_numbers: number of points below, above and on the line
    :param list legend: list of the categories names.
    :param str title: the plot title
    :param list[dict] batch_average: Keys are str(low) to str(high) of the batch.
    Values are: the average sender answer in this batch
    :param bool batch_with_line: if the batch numbers are regarding the line or not
    :return:
    """

    x_points_took = list()
    x_points_not_took = list()
    y_points_took = list()
    y_points_not_took = list()

    for took, x_points_list, y_points_list in \
            [[0, x_points_took, y_points_took], [1, x_points_not_took, y_points_not_took]]:
        for group in legend:
            group_data = data_to_plot.loc[(data_to_plot.prob_status == group) &
                                          (data_to_plot.group_receiver_choice == took)]
            x_points_list.append(list(group_data[xlabel]))
            y_points_list.append(list(group_data[ylabel]))

    for i in range(len(legend)):
        inner_batch_average = batch_average if batch_average is None else batch_average[i]
        inner_line_numbers = line_numbers if line_numbers is None else line_numbers[i]

        create_point_plot([x_points_took[i]], [y_points_took[i]], legend=[legend[i]], xlabel=xlabel, ylabel=ylabel,
                          title=f'{title} \nfor the group: {legend[i]}', points_x_other_color=[x_points_not_took[i]],
                          points_y_other_color=[y_points_not_took[i]], line_numbers=inner_line_numbers,
                          batch_average=inner_batch_average, batch_with_line=batch_with_line)

    return


def batch_analysis(data, batch_size_func=None, group=None):
    """
    This function count the average expert answer in each batch of 5 of the expected value.
    i.e for each batch of 5 of the expected value (-20 to -15, -15 to -10 etc) the average of the group_sender_answer
    :param data: data frame with at least 2 columns: group_sender_answer, given_expected_value
    :param int batch_size_func: size of batch if not as defined above
    :param str group: the name of the group we work on
    :return: dict. Keys are str(low) to str(high) of the batch. Values are: the average sender answer in this batch.
    """

    batch_size_func = batch_size_func if batch_size_func is not None else batch_size

    batch_numbers_dict = dict()
    for low in range(-20, 21, batch_size_func):
        if low == 20 and batch_size_func == batch_size:
            continue
        high = low + batch_size_func
        batch_data = data.loc[(data.given_expected_value >= low) & (data.given_expected_value < high)]
        batch_sender_answer_average = round(batch_data.group_sender_answer.mean(), 2)
        batch_numbers_dict[str(low) + ' to ' + str(high)] = batch_sender_answer_average

        if batch_data.shape[0] == 0:  # no data in this section
            print('no data in batch from', low, 'to', high, 'for group', group)

    return batch_numbers_dict


def diff_prob_line_analysis(data, diff_prob=True):
    """
    This function count the the number of points below and above the line in each batch of 5 of the expected value.
    i.e for each batch of 5 of the expected value (-20 to -15, -15 to -10 etc) the number of points
    :param data: data frame with at least 1 column: diff_prob, given_expected_value, expert_estimate_expected_value
    :param bool diff_prob: do we calculate for diff_prob or EV line
    :return:
    """

    group_by_expert_estimate_expected_value = data.groupby(
        by=['given_expected_value', 'expert_estimate_expected_value'])['line'].count()

    if diff_prob:
        group_by = data.groupby(by=['given_expected_value', 'diff_prob'])['line'].count()

    else:
        group_by = data.groupby(by=['given_expected_value', 'line'])['diff_prob'].count()

    batch_numbers_dict = dict()
    for low in range(-20, 20, batch_size):
        high = low + batch_size
        batch_data = group_by.iloc[(group_by.index.get_level_values('given_expected_value') >= low) &
                                   (group_by.index.get_level_values('given_expected_value') < high)]
        batch_data_expert_estimate_expected_value =\
            group_by_expert_estimate_expected_value.iloc[
                (group_by_expert_estimate_expected_value.index.get_level_values('given_expected_value') >= low) &
                (group_by_expert_estimate_expected_value.index.get_level_values('given_expected_value') < high)]

        if diff_prob:
            batch_numbers_dict[str(low) + ' to ' + str(high) + 'on'] =\
                batch_data.iloc[batch_data.index.get_level_values('diff_prob') == 0].sum()
            batch_numbers_dict[str(low) + ' to ' + str(high) + 'below'] =\
                batch_data.iloc[batch_data.index.get_level_values('diff_prob') < 0].sum()
            batch_numbers_dict[str(low) + ' to ' + str(high) + 'above'] =\
                batch_data.iloc[batch_data.index.get_level_values('diff_prob') > 0].sum()

            batch_numbers_dict[str(low) + ' to ' + str(high) + 'positive EV'] = \
                batch_data_expert_estimate_expected_value.iloc[
                    batch_data_expert_estimate_expected_value.index.get_level_values('expert_estimate_expected_value')
                    >= 0].sum()
            batch_numbers_dict[str(low) + ' to ' + str(high) + 'negative EV'] = \
                batch_data_expert_estimate_expected_value.iloc[
                    batch_data_expert_estimate_expected_value.index.get_level_values('expert_estimate_expected_value')
                    < 0].sum()

        else:
            for line in ['on', 'below', 'above']:
                batch_numbers_dict[str(low) + ' to ' + str(high) + line] = \
                    batch_data.iloc[batch_data.index.get_level_values('line') == line].sum()

    return batch_numbers_dict


def line_analysis(data):
    """
    This function return the number of points above, below and on the line for each data points,
    split to negative and positive EV.
    On the line are on the line, or abs(expert_answer - given_p) <= 0.1
    :param data: data frame with at least 3 columns: given_expected_value, line, group_receiver_choice
    :return: dict: 12 numbers: positive_above, positive_below, positive_on, negative_above, negative_below, negative_on
    """

    positive_above_not_take = data.loc[(data.given_expected_value >= 0) & (data.line == 'above') &
                                       (data.group_receiver_choice == 1)].shape[0]
    positive_below_not_take = data.loc[(data.given_expected_value >= 0) & (data.line == 'below') &
                                       (data.group_receiver_choice == 1)].shape[0]
    positive_on_not_take = data.loc[(data.given_expected_value >= 0) & (data.line == 'on') &
                                    (data.group_receiver_choice == 1)].shape[0]

    positive_above_take = data.loc[(data.given_expected_value >= 0) & (data.line == 'above') &
                                   (data.group_receiver_choice == 0)].shape[0]
    positive_below_take = data.loc[(data.given_expected_value >= 0) & (data.line == 'below') &
                                   (data.group_receiver_choice == 0)].shape[0]
    positive_on_take = data.loc[(data.given_expected_value >= 0) & (data.line == 'on') &
                                (data.group_receiver_choice == 0)].shape[0]

    negative_above_not_took = data.loc[(data.given_expected_value < 0) & (data.line == 'above') &
                                       (data.group_receiver_choice == 1)].shape[0]
    negative_below_not_took = data.loc[(data.given_expected_value < 0) & (data.line == 'below') &
                                       (data.group_receiver_choice == 1)].shape[0]
    negative_on_not_took = data.loc[(data.given_expected_value < 0) & (data.line == 'on') &
                                    (data.group_receiver_choice == 1)].shape[0]

    negative_above_took = data.loc[(data.given_expected_value < 0) & (data.line == 'above') &
                                   (data.group_receiver_choice == 0)].shape[0]
    negative_below_took = data.loc[(data.given_expected_value < 0) & (data.line == 'below') &
                                   (data.group_receiver_choice == 0)].shape[0]
    negative_on_took = data.loc[(data.given_expected_value < 0) & (data.line == 'on') &
                                (data.group_receiver_choice == 0)].shape[0]

    return {"positive_above didn't take": positive_above_not_take,
            "positive_below didn't take": positive_below_not_take,
            "positive_on didn't take": positive_on_not_take,
            'positive_above took': positive_above_take, 'positive_below took': positive_below_take,
            'positive_on took': positive_on_take,
            "negative_above didn't take": negative_above_not_took,
            "negative_below didn't take": negative_below_not_took,
            "negative_on didn't take": negative_on_not_took,
            'negative_above took': negative_above_took, 'negative_below took': negative_below_took,
            'negative_on took': negative_on_took}


def create_statistics(data):
    """
    This function calculate some statistics for data
    :param data: series with the data need to calculate the statistics
    :return: list: list of min, max, mean, median of the data
    """

    min = data.min()
    max = data.max()
    mean = data.mean()
    median = data.median()
    std = data.std()
    count = data.count()

    return[min, max, mean, median, std, count]


def autolabel(rects, ax, rotation, max_height, percentage_graph=False, convert_to_int=True, add_0_label=False,
              fontsize=7, text=None):
    """
    Attach a text label above each bar displaying its height
    :param convert_to_int: if height>1 convert to int
    :param text: if we want to pass specific text
    """
    for rect_num, rect in enumerate(rects):
        if text is None:
            height = rect.get_height()
            if (height > 0.0 or height < 0.0) or (height == 0.0 and add_0_label):  # don't add 0 label
                if convert_to_int and height > 1:
                    height = int(height)
                else:
                    height = height.round(2)
                if height < 0:
                    y_pos = height - max_height * 0.035
                else:
                    y_pos = height + max_height * 0.01
            else:
                continue
        else:
            height = text[rect_num]
            y_pos = rect.get_height() + max_height * 0.001
        ax.text(rect.get_x() + rect.get_width() / 2, y_pos, height, ha='center', va='bottom', rotation=rotation,
                fontsize=fontsize)


def create_point_plot(points_x, points_y, legend, title, xlabel, ylabel, points_x_other_color=None,
                      points_y_other_color=None, line_numbers=None, batch_average=None, batch_with_line=False,
                      add_text=[' DM took', " DM didn't take"], add_line_points=False, prob_value=0.0, inner_dir='',
                      curr_date_directory=date_directory, add_line_between_points: bool=False,
                      add_truth_telling=True, print_max_min_values: dict=None):
    """
    This function create a plot using lists of points
    :param list(list) points_x: list of lists. each list is a list of the x value of the points to add to the plot
    :param list(list) points_y: list of lists. each list is a list of the y value of the points to add to the plot
    :param list legend: list of legends of the lists in points
    :param str title: the name of the plot
    :param str xlabel: the label of the x axis
    :param str ylabel: the label of the y axis
    :param list(list) points_x_other_color: x points to plot with different color
    :param list(list) points_y_other_color: y points to plot with different color
    :param dict line_numbers: number of points regard the line and the EV
    :param dict batch_average: the average sender answer in this batch
    :param bool batch_with_line: if the batch numbers are regarding the line or not
    :param list add_text: list of the text to add to the legend
    :param bool add_line_points: whether to add line between the points or not
    :param float prob_value: if table_prob_ev creation: the prob we create the plot for
    :param str inner_dir: if we want to save the plots in an inner directory
    :param str curr_date_directory: the name of the date_directory if call this function from another file
    :param bool add_line_between_points: if we want to add a line between the points in the plot
    :param bool add_truth_telling: if to add the truth_telling line- for expert answer analysis
    :param print_max_min_values: add the max and min values for each point
    :return:
    """

    print('Create point plot for', title)

    fig, ax = plt.subplots()
    x_max, y_max = 0, 0
    x_min, y_min = 1000, 10000
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
    # print('Create plot for', title)

    markers = [".", "x", "+", "1"]
    colors = ['orange', 'seagreen', 'navy', 'crimson']
    didnt_take_color = 'limegreen'

    for i, (x, y) in enumerate(zip(points_x, points_y)):
        if points_x_other_color is None:
            label = legend[i]
        else:
            label = legend[i] + add_text[0]

        if add_line_between_points:
            ax.plot(x, y, color=colors[i], label=label, marker=markers[i], linestyle='-')
            if add_truth_telling:
                ax.plot(x, x, color='navy', marker=markers[i], linestyle='-', label='Truth Telling')
            previous_moved = False  # if the previous j was moved
            for j, txt in enumerate(zip(x, y)):
                if j < len(y)-1 and y[j+1] - y[j] < 0.1 and x[j+1] - x[j] < 1:
                    if previous_moved:
                        x_pos = x[j] - 0.7
                    else:
                        previous_moved = True
                        x_pos = x[j] - 1
                else:
                    x_pos = x[j]
                ax.annotate(f'({x[j]},{round(y[j], 2)})', (x_pos, y[j]))
            if print_max_min_values:
                # Add table with the max, min, average and expeted values
                cell_text = list()
                cell_text.append([x[index] for index in range(len(x))])
                cell_text.append([round(y[index], 2) for index in range(len(x))])
                cell_text.append([print_max_min_values[x[index]][0] for index in range(len(x))])
                cell_text.append([print_max_min_values[x[index]][1] for index in range(len(x))])
                # Add a table at the bottom of the axes
                the_table = plt.table(cellText=cell_text,
                                      rowLabels=['Average Score', 'Expert Average Answer', 'Min Score', 'Max Score'],
                                      loc='bottom')
                the_table.auto_set_font_size(False)
                the_table.set_fontsize(10)
                plt.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=True,  # ticks along the top edge are on
                    labelbottom=False,  # labels along the bottom edge are off
                    labeltop=True)  # labels along the top edge are on
        else:
            ax.scatter(x, y, color=colors[i], label=label, marker=markers[i])
        if add_line_points:
            ax.plot(x, y, color=colors[i])
            if i == 1:
                ax.hlines(y=prob_value, xmin=x_min, xmax=x_max)

        if points_x_other_color is not None:
            ax.scatter(points_x_other_color[i], points_y_other_color[i], color=didnt_take_color,
                       label=legend[i] + add_text[1], marker=markers[i])
            if add_line_points:
                ax.plot(points_x_other_color[i], points_y_other_color[i], color=didnt_take_color)

    # add some text for labels, title and axes ticks
    ax.legend(loc=9, bbox_to_anchor=(0.5, -0.3))
    plt.title(title)
    plt.xlabel(xlabel, labelpad=55)
    plt.ylabel(ylabel)

    if batch_average is not None:
        step_size = 1 / 7.6
        start = 0.03
        if not batch_with_line:  # for batch not regarding the line
            if len(legend) == 1 and 'zero_one' in legend[0]:
                average_text = 'Probability of: expert answer = 1 (per section)'
            else:
                average_text = 'Average experts probability estimation (per section)'
            plt.text(1.05, -0.05, average_text, horizontalalignment='left', verticalalignment='bottom',
                     transform=ax.transAxes, color='navy')

            for index, low in enumerate(range(-20, 20, batch_size), 0):
                high = low + batch_size
                plt.text(start + (step_size * index), -0.05,
                         str(batch_average[str(low) + ' to ' + str(high)]),
                         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
                         color='navy')
        else:  # for batch regarding the line
            average_text = 'Number of points (per section)'
            extra = 0.05
            if str(-20) + ' to ' + str(-20 + batch_size) + 'positive EV' in batch_average.keys():
                # add the text for the number of points with positive and negative estimated EV per section
                # (for diff_prob graphs)
                plt.text(1.05, -0.05 - 0.2, average_text + ' with negative estimated EV',
                         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
                         color='navy')
                plt.text(1.05, -0.05 - 0.15, average_text + ' with positive estimated EV',
                         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
                         color='navy')

            for line_index, line in enumerate(['above', 'on', 'below'], 0):
                plt.text(1.05, -0.05 - (line_index * extra), average_text + ' ' + line + ' the line',
                         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color='navy')
                for index, low in enumerate(range(-20, 20, batch_size), 0):
                    high = low + batch_size
                    plt.text(start + (step_size * index), -0.05 - (line_index * extra),
                             str(batch_average[str(low) + ' to ' + str(high) + line]),
                             horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
                             color='navy')

                    # add the number of points with positive and negative estimated EV per section
                    # (for diff_prob graphs)
                    if str(low) + ' to ' + str(high) + 'positive EV' in batch_average.keys() and line_index == 0:
                        plt.text(start + (step_size * index), -0.05 - 0.15,
                                 str(batch_average[str(low) + ' to ' + str(high) + 'positive EV']),
                                 horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
                                 color='navy')
                        plt.text(start + (step_size * index), -0.05 - 0.2,
                                 str(batch_average[str(low) + ' to ' + str(high) + 'negative EV']),
                                 horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
                                 color='navy')

    if line_numbers is not None:
        left = -0.1
        bottom = 0
        right = 1.05
        top = 1

        took_list = [['took', 0], ["didn't take", 0.035]]
        colors_line = {took_list[0][0]: colors[0], took_list[1][0]: didnt_take_color}
        # add text: number of points below, above and on the line
        for took, extra in took_list:
            plt.text(right, top + 0.02 - extra,
                     'Above line, positive EV, DM ' + took + ':' + str(line_numbers['positive_above ' + took]),
                     horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
                     color=colors_line[took])

            plt.text(right, top - 0.03 - extra,
                     'On line, positive EV, DM ' + took + ':' + str(line_numbers['positive_on ' + took]),
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                     color=colors_line[took])

            plt.text(right, top - 0.13 - extra,
                     'Below line, positive EV, DM ' + took + ':' + str(line_numbers['positive_below ' + took]),
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                     color=colors_line[took])

            plt.text(left, bottom + 0.02 - extra,
                     'Above line, negative EV, DM ' + took + ':' + str(line_numbers['negative_above ' + took]),
                     horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                     color=colors_line[took])

            plt.text(left, bottom - 0.03 - extra,
                     'On line, negative EV, DM ' + took + ':' + str(line_numbers['negative_on ' + took]),
                     horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                     color=colors_line[took])

            plt.text(left, bottom - 0.13 - extra,
                     'Below line, negative EV, DM ' + took + ':' + str(line_numbers['negative_below ' + took]),
                     horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                     color=colors_line[took])

    # plt.show()

    if curr_date_directory != date_directory:
        directory = os.path.join(orig_data_analysis_directory, curr_date_directory)
    else:
        directory = data_analysis_directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    fig_to_save = fig
    fig_to_save.savefig(os.path.join(directory, inner_dir, title + '.png'), bbox_inches='tight')

    return


def create_chart_bars(title, x, y, xlabel=None, ylabel=None, legend=None, add_point=None,
                      curr_date_directory=date_directory, curr_colors=None, percentage_graph=False, step=5,
                      x_is_float=False, add_lst_xticks: bool=True, use_all_xticks: bool=False):
    """
    This function create a chart bar with number of samples in each bin
    :param str title: title of the bar
    :param list(numpy array) x: the values in the x axis
    :param list(list) y: the values in the y axis
    :param str xlabel: the label of the x axis
    :param str ylabel: the label of the y axis
    :param list legend: list of x legend and x2,x3 legend
    :param int add_point: one point we want to add - its x value
    :param str curr_date_directory: the name of the date_directory if call this function from another file
    :param list curr_colors: if we want to pass specific colors
    :param bool percentage_graph: if this is a graph for %
    :param int step: the step size in X axis
    :param bool x_is_float: if the x axis is probabilities
    :param add_lst_xticks: add xticks
    :param use_all_xticks: if to add all x as xticks
    :return:
    """

    fig, ax = plt.subplots()
    if x_is_float:
        width = 0.1
    else:
        width = 0.3
    print('Create bar for', title)

    number_of_unique_points = len(set([item for x_list in x for item in x_list]))
    if isinstance(x[0], list):
        all_max = max([max(x_list) for x_list in x if len(x_list) > 0])
        all_min = min([min(x_list) for x_list in x if len(x_list) > 0])
    else:
        all_max = max([x_list.max() for x_list in x if x_list.shape[0] > 0])
        all_min = min([x_list.min() for x_list in x if x_list.shape[0] > 0])

    y_max = max([max(y_list) for y_list in y if len(y_list) > 0])

    if percentage_graph:
        plt.ylim(0, 110)

    font = FontProperties()
    font.set_size('large')
    font.set_family('fantasy')
    font.set_style('normal')

    if type(all_max) in [int, float]:
        if all_max < 10:
            rotation = 'horizontal'
        else:
            rotation = 'vertical'
    else:
        rotation = 'vertical'

    if add_point is not None:
        ax.axvline(x=add_point, linewidth=3, color='red', label='average_payoff_accurate_player')
        ax.text(x=add_point, y=y_max, s=add_point, ha='center', va='bottom', rotation='horizontal',
                fontproperties=font)

    if curr_colors is None:
        colors = ['orange', 'seagreen', 'navy', 'orchid', 'dimgray', 'purple', 'skyblue', 'yellow', 'black', 'orangered']
    else:
        colors = curr_colors
    if len(x) > 1:  # add all points to the same plot
        for i in range(len(x)):
            rects = ax.bar(x[i] + width*i, y[i], width, color=colors[i], label=legend[i])
            autolabel(rects, ax, rotation, max_height=y_max, percentage_graph=percentage_graph)

    else:
        if x_is_float:
            plt.bar(x[0], y[0], width=width)
        else:
            plt.bar(x[0], y[0])
        rects = ax.patches
        autolabel(rects, ax, rotation, max_height=y_max, percentage_graph=percentage_graph)

    # add some text for labels, title and axes ticks
    ax.legend(loc=9, bbox_to_anchor=(0.5, -0.15))
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if add_lst_xticks:
        if number_of_unique_points <= 18 or use_all_xticks:
            if len(x) > 1:
                lst_xticks = np.append(x[0], x[1])
            else:
                lst_xticks = x[0]
        elif all_max % 5 == 0:
            lst_xticks = np.arange(all_min, all_max + 1, step=step)
        else:
            lst_xticks = np.arange(all_min, all_max, step=step)
            lst_xticks = np.append(lst_xticks, all_max)

        if not x_is_float and not isinstance(lst_xticks, list):
            lst_xticks = lst_xticks.astype(int)
        elif x_is_float and not isinstance(lst_xticks, list):
            lst_xticks = [round(elem, 2) for elem in lst_xticks]

        # if add_point:
        #     lst_xticks = np.append(lst_xticks, add_point)

        plt.xticks(lst_xticks)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize('x-small')
        # plt.tick_params(str='x', labelsize='small')

    else:
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

    # plt.show()

    if curr_date_directory != date_directory:
        directory = os.path.join(orig_data_analysis_directory, curr_date_directory)
    else:
        directory = data_analysis_directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    fig_to_save = fig
    fig_to_save.savefig(os.path.join(directory, title + '.png'), bbox_inches='tight')

    return


def create_histogram(title, x, xlabel, ylabel, add_labels=False, curr_date_directory=date_directory, step=5):
    """
    This function create a chart bar with number of samples in each bin
    :param str title: title of the bar
    :param numpy array x: the values in the x axis
    :param str xlabel: the label of the x axis
    :param str ylabel: the label of the y axis
    :param bool add_labels: whether to add label for each bar or not
    :param str curr_date_directory: the name of the date_directory if call this function from another file
    :param int step: the step size in X axis
    :return:
    """

    print(time.asctime(time.localtime(time.time())), ': Create histogram for', title)

    fig_per_hour = plt.figure()
    per_hour = fig_per_hour.add_subplot(111)
    counts, bins, patches = per_hour.hist(x, bins=75, normed=False, color='dodgerblue', linewidth=0)
    if add_labels:
        gap = 1
    else:
        gap = 0
    plt.gca().set_xlim(x.min() - gap, x.max() + gap)

    if add_labels:
        max_height = np.bincount([int(i) for i in x]).max()
        rects = per_hour.patches
        autolabel(rects, per_hour, 'vertical', max_height)

    # add some text for labels, title and axes ticks
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if add_labels:
        if x.shape[0] <= 20:
            lst_xticks = np.arange(x.min(), x.max(), step=1)
            lst_xticks = np.append(lst_xticks, x.max())
            plt.xticks(lst_xticks)
        elif x.max() % 5 == 0:
            plt.xticks(np.arange(x.min(), x.max() + gap, step=step))
        else:
            lst_xticks = np.arange(x.min(), x.max(), step=step)
            lst_xticks = np.append(lst_xticks, x.max())
            plt.xticks(lst_xticks)
    else:
        plt.xticks(np.arange(x.min(), x.max() + 0.01, step=0.25))
    # plt.show()

    if curr_date_directory != date_directory:
        directory = os.path.join(orig_data_analysis_directory, curr_date_directory)
    else:
        directory = data_analysis_directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    fig_to_save = fig_per_hour
    fig_to_save.savefig(os.path.join(directory, title + '.png'), bbox_inches='tight')

    return


def high_low_probability_diff(data, data_type='all'):
    """
    This function analyze the probability diff for high or low given probability
    :param data: data frame with the data to analyze
    :param str data_type: which type of data: all or other (like more_than_3_values)
    :return:
    """

    for threshold in [['high', 0.7], ['low', 0.3]]:
        if threshold[0] == 'high':
            condition = data.group_p_lottery > threshold[1]
        else:
            condition = data.group_p_lottery < threshold[1]

        data_for_bar = data.loc[condition].copy()
        group_by = data_for_bar.groupby(by='diff_prob').pair_id.count()
        x = np.array(group_by.index)
        title = 'diff_prob_' + threshold[0] + '_than_' + str(threshold[1]) + '_' + data_type
        create_histogram(title=title, x=x, xlabel='probability_difference', ylabel='number_of_rounds')

    return


def calculate_diff_prob(data, text=''):
    """
    This function get data and calculate the mean and median prob diff and create the char bar
    :param data: data frame to use, it has to have diff_prob column and pair_id and abs_diff_prob
    :param str text: test to add to the print
    :return:
    """
    print(time.asctime(time.localtime(time.time())), ': abs average difference in prob ' + text + ':',
          data['abs_diff_prob'].mean(), ', abs median difference in prob ' + text + ':',
          data['abs_diff_prob'].median())
    logging.info('{}: abs average difference in prob {}: {} abs median difference in prob {}: {}'.
                 format(time.asctime(time.localtime(time.time())), text, data['abs_diff_prob'].mean(),
                        text, data['abs_diff_prob'].median()))

    # bar chart for prob_diff
    group_by = data.groupby(by='diff_prob').pair_id.count()
    x = np.array(group_by.index)

    create_histogram(title='diff_prob_' + text, x=x, xlabel='probability_difference', ylabel='number_of_rounds')

    return


class DataAnalysis:
    def __init__(self, class_gender='all genders'):
        self.gender = class_gender
        print('Start running data analysis on data folder', date_directory)
        logging.info('Start running data analysis on data folder'.format(date_directory))
        columns_to_use = ['participant_code', 'participant__current_page_name', 'participant_visited',
                          'participant_mturk_worker_id', 'participant_mturk_assignment_id', 'participant_payoff',
                          'player_id_in_group', 'player_name', 'player_age', 'player_gender', 'player_is_student',
                          'player_occupation', 'player_residence', 'player_payoff', 'group_id_in_subsession',
                          'group_sender_answer', 'group_receiver_choice', 'group_lottery_result',
                          'group_sender_timeout',
                          'group_receiver_timeout', 'group_sender_payoff', 'group_x_lottery', 'group_y_lottery',
                          'group_p_lottery', 'subsession_round_number', 'session_code']
        self.results = pd.read_csv(os.path.join(data_directory, date_directory, 'first_results_data.csv'),)
                                   # usecols=columns_to_use)
        self.results.columns = self.results.columns.str.replace(r".", "_")

        if split_gender:
            gender_participants = self.results.loc[self.results['player_gender'] == self.gender].participant_code
            self.results = self.results.loc[self.results.participant_code.isin(gender_participants)]

        # keep only data to use
        self.results = self.results.loc[self.results.participant_visited == 1]
        self.time_spent = pd.read_csv(os.path.join(data_directory, date_directory, 'TimeSpent_first_results.csv'))
        self.payments = pd.read_csv(os.path.join(data_directory, date_directory, 'Approved assignments.csv'))
        self.results_payments =\
            self.results.merge(self.payments, left_on=['participant_mturk_worker_id', 'participant_mturk_assignment_id',
                                                       'participant_code'],
                               right_on=['worker_id', 'assignment_id', 'participant_code'], how='left')
        self.results_payments = self.results_payments.assign(status='')
        self.results_payments = self.results_payments.assign(prob_status='')
        self.results_payments = self.results_payments.assign(player_timeout=0)
        self.results_payments = self.results_payments.assign(player_answer=0)
        self.results_payments = self.results_payments.assign(max_num_rounds_timeout=0)
        self.results_payments = self.results_payments.assign(zero_one_answer='')

        # adding pair_id
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
        self.participants_started = self.results_payments_play.participant_code.unique()

    def arrange_player_data(self):
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
        self.results_payments['player_timeout'] = np.where(self.results_payments.group_p_lottery.isnull(), 1,
                                                           self.results_payments.player_timeout)
        self.results_payments['partner_timeout'] = np.where(self.results_payments.group_p_lottery.isnull(), 1,
                                                            self.results_payments.partner_timeout)
        self.results_payments['player_answer'] = np.where(self.results_payments.player_id_in_group == 1,
                                                          self.results_payments.group_sender_answer,
                                                          self.results_payments.group_receiver_choice)
        self.results_payments['partner_answer'] = np.where(self.results_payments.player_id_in_group == 2,
                                                           self.results_payments.group_sender_answer,
                                                           self.results_payments.group_receiver_choice)
        return

    def time_spent_analysis(self):
        """
        This function analyze the time spent in SenderPage, ReceiverPage, Introduction and PersonalInformation
        :return:
        """

        # arrange time spent
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
                time_spent_to_merge = time_spent_to_merge.append(
                    role_data[['participant_code', 'subsession_round_number', 'seconds_on_page']])
                role_data = role_data.loc[role_data.seconds_on_page <= 7]
                x = np.array(role_data['seconds_on_page'])
                create_histogram(title='time_spent_in_page' + role[1], x=x, xlabel='seconds_in_page',
                                 ylabel='total_number_of_rounds', add_labels=True)

            average_time_spent = role_data.seconds_on_page.mean()
            median_time_spent = role_data.seconds_on_page.median()
            min_time_spent = role_data.seconds_on_page.min()
            max_time_spent = role_data.seconds_on_page.max()

            print(time.asctime(time.localtime(time.time())), ': average time spent on page', role[1], 'is:',
                  average_time_spent, ', median time spent on page', role[1], 'is:', median_time_spent,
                  'max time:', max_time_spent, ', min_time:', min_time_spent)
            logging.info('{}: average time spent on page: {} is {}, median time spent on page {} is: {},'
                         'max time is: {}, min time is: {}'.
                         format(time.asctime(time.localtime(time.time())), role[1], average_time_spent, role[1],
                                median_time_spent, max_time_spent, min_time_spent))

        self.results_payments = self.results_payments.merge(time_spent_to_merge,
                                                            on=['participant_code', 'subsession_round_number'])

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
        # wait: if we pay less than $2.5
        self.results_payments.loc[self.results_payments.total_pay < 2.5, 'status'] = 'wait'
        # partner_left_before_start: they inserted personal info and didn't play - probably the partner didn't start
        self.results_payments.loc[self.results_payments['participant__current_page_name'] == 'AfterInstructions',
                                  'status'] = 'partner_left_before_start'
        # left_before_start: didn't start to play, didn't insert personal info and have not got paid
        self.results_payments.loc[(~self.results_payments['participant_code'].isin(self.participants_started)) &
                                  (self.results_payments['participant__current_page_name'] != 'AfterInstructions') &
                                  (self.results_payments.total_pay.isnull()) &
                                  (self.results_payments.participant_payoff == 0), 'status'] = 'left_before_start'
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
                                  (self.results_payments.participant_payoff < 0), 'status'] = 'wait_but_not_got_paid'
        # left: played and left in the middle. They started, but not got money
        self.results_payments.loc[(self.results_payments['participant_code'].isin(self.participants_started)) &
                                  (self.results_payments.total_pay.isnull()), 'status'] = 'left'
        # if the player left, but in the last 2 rounds he played - no timeout --> so he played
        played_not_paid = self.results_payments.loc[(self.results_payments.status == 'left') &
                                                    (self.results_payments.subsession_round_number.isin([50, 49])) &
                                                    (self.results_payments.player_timeout == 0) &
                                                    (self.results_payments.group_p_lottery.notnull())].\
            participant_code.unique()
        self.results_payments.loc[self.results_payments.participant_code.isin(played_not_paid), 'status'] = 'play'
        # get the participants that left info
        pair_id_left = self.results_payments.loc[self.results_payments.status == 'left']['pair_id'].unique()
        left_participant_code = self.results_payments.loc[self.results_payments.status == 'left'][
            'participant_code'].unique()
        number_of_left_in_pair = self.results_payments.loc[self.results_payments.status == 'left'].\
            groupby(by='pair_id').participant_code.count()
        both_left_pair_id = number_of_left_in_pair.loc[number_of_left_in_pair == 100].index
        one_left_pair_id = number_of_left_in_pair.loc[number_of_left_in_pair == 50].index
        # partner_left: the pair_id is in the list of left participants and the participant_code is not in the list
        self.results_payments.loc[(~self.results_payments.participant_code.isin(left_participant_code)) &
                                  (self.results_payments.pair_id.isin(one_left_pair_id) &
                                   (self.results_payments['participant_code'].isin(self.participants_started))),
                                  'status'] = 'partner_left'
        # both_left: the pair_id and the participant_code are in the list of left participants
        self.results_payments.loc[(self.results_payments.participant_code.isin(left_participant_code)) &
                                  (self.results_payments.pair_id.isin(both_left_pair_id) &
                                   (self.results_payments['participant_code'].isin(self.participants_started))),
                                  'status'] = 'both_left'
        # play: the pair_id and the participant_code are not in the list of left participants, and they got paid
        self.results_payments.loc[(~self.results_payments.participant_code.isin(left_participant_code)) &
                                  (~self.results_payments.pair_id.isin(pair_id_left) &
                                   (self.results_payments.bonus >= 0) &
                                   (self.results_payments['participant_code'].isin(self.participants_started))),
                                  'status'] = 'play'
        # play_but_not_paid: the pair_id and the participant_code are not in the list of left participants,
        # but they didn't get money
        # self.results_payments.loc[(~self.results_payments.participant_code.isin(left_participant_code)) &
        #                           (~self.results_payments.pair_id.isin(pair_id_left) &
        #                            (self.results_payments.total_pay.isnull()) &
        #                            (self.results_payments['participant_code'].isin(self.participants_started))),
        #                           'status'] = 'play_but_not_paid'

        # unknown - I didn't assign anything
        self.results_payments.loc[self.results_payments.status == '', 'status'] = 'unknown'

        self.results_payments.to_csv(os.path.join(data_analysis_directory, 'results_payments_status.csv'))

        return

    def print_pairs(self):
        """
        This function calculate the number of pairs of each type : both finished, one left, both left
        :return:
        """
        both_finished = self.results_payments.loc[self.results_payments.status == 'play'].pair_id.unique()
        one_left = self.results_payments.loc[self.results_payments.status == 'left'].pair_id.unique()
        both_left = self.results_payments.loc[self.results_payments.status == 'both_left'].pair_id.unique()

        print(time.asctime(time.localtime(time.time())), ': number of participants started:',
              self.participants_started.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of pairs finished:', both_finished.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of pairs one left:', one_left.shape[0])
        print(time.asctime(time.localtime(time.time())), ': number of pairs both left:', both_left.shape[0])
        logging.info('{}: number of participants started: {}'.format(time.asctime(time.localtime(time.time())),
                                                                     self.participants_started.shape[0]))
        logging.info('{}: number of pairs finished: {}'.format(time.asctime(time.localtime(time.time())),
                                                               both_finished.shape[0]))
        logging.info('{}: number of pairs one left: {}'.format(time.asctime(time.localtime(time.time())),
                                                               one_left.shape[0]))
        logging.info('{}: number of pairs both left: {}'.format(time.asctime(time.localtime(time.time())),
                                                                both_left.shape[0]))

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

    def probability_analysis(self):
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
        # calculate diff probability
        data_to_analyze['diff_prob'] = data_to_analyze['player_answer'] - data_to_analyze['group_p_lottery']
        data_to_analyze['abs_diff_prob'] = abs(data_to_analyze['player_answer'] - data_to_analyze['group_p_lottery'])
        self.results_payments['diff_prob'] = self.results_payments['player_answer'] -\
                                             self.results_payments['group_p_lottery']
        self.results_payments['abs_diff_prob'] = abs(self.results_payments['player_answer'] -
                                                     self.results_payments['group_p_lottery'])
        calculate_diff_prob(data_to_analyze)

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
        unique_list_zero_one = np.unique(np.array([list(lst) for lst in short.values]))
        unique_numbers = list(set(list(itertools.chain.from_iterable(unique_list_zero_one))))
        if 0.0 in unique_numbers and 1.0 in unique_numbers:
            self.results_payments.loc[self.results_payments.participant_code.isin(short.index), 'prob_status'] =\
                '3_or_less_values'
            data_to_analyze.loc[data_to_analyze.participant_code.isin(short.index), 'prob_status'] = '3_or_less_values'
            play_short = data_to_analyze.loc[(data_to_analyze.prob_status == '3_or_less_values') &
                                             (data_to_analyze.status == 'play')]
            print(time.asctime(time.localtime(time.time())),
                  ': number of pairs play but less than 3 values of prob:', play_short.pair_id.unique().shape[0])
            logging.info('{}: number of pairs play but less than 3 values of prob: {}'.
                         format(time.asctime(time.localtime(time.time())), play_short.pair_id.unique().shape[0]))
            partner_left_short = data_to_analyze.loc[(data_to_analyze.prob_status == '3_or_less_values') &
                                                     (data_to_analyze.status == 'partner_left')]
            print(time.asctime(time.localtime(time.time())),
                  ': number of paris one left but less than 3 values of prob:',
                  partner_left_short.pair_id.unique().shape[0])
            logging.info('{}: number of paris one left but less than 3 values of prob: {}'.
                         format(time.asctime(time.localtime(time.time())),
                                partner_left_short.pair_id.unique().shape[0]))
            more_than_3_values = data_to_analyze.loc[data_to_analyze.prob_status != '3_or_less_values']
            calculate_diff_prob(more_than_3_values, 'more_than_3_values')

            # 1-3 values
            less_than_3_values = data_to_analyze.loc[data_to_analyze.prob_status == '3_or_less_values']
            calculate_diff_prob(less_than_3_values, 'less_than_3_values')

            # bar for low and high probability
            high_low_probability_diff(data=more_than_3_values, data_type='more_than_3_values')
            high_low_probability_diff(data=less_than_3_values, data_type='less_than_3_values')

        else:
            print('0 and 1 are not in unique numbers in short')

        # bar for low and high probability
        high_low_probability_diff(data=data_to_analyze)

        # define accurate participants
        data_to_analyze = data_to_analyze.loc[data_to_analyze.pct_timeout <= 50]
        data_to_analyze['accurate_answer'] = np.where(data_to_analyze.abs_diff_prob <= 0.2, 1, 0)
        accurate_to_analyze = data_to_analyze.groupby(by='participant_code').agg({'accurate_answer': ['count', 'sum']})
        accurate_to_analyze.columns = accurate_to_analyze.columns.droplevel()
        accurate_to_analyze['pct_accurate'] = accurate_to_analyze['sum'] / accurate_to_analyze['count']
        accurate = accurate_to_analyze.loc[accurate_to_analyze.pct_accurate >= 0.75]
        self.results_payments.loc[self.results_payments.participant_code.isin(accurate.index), 'prob_status'] =\
            'accurate'

        # print numbers
        play_accurate = self.results_payments.loc[(self.results_payments.prob_status == 'accurate') &
                                                  (self.results_payments.status == 'play')]
        print(time.asctime(time.localtime(time.time())),
              ': number of pairs play expert estimate accurate probability:', play_accurate.pair_id.unique().shape[0])
        logging.info('{}: number of pairs play expert estimate accurate probability: {}'.
                     format(time.asctime(time.localtime(time.time())), play_accurate.pair_id.unique().shape[0]))
        partner_left_accurate = self.results_payments.loc[(self.results_payments.prob_status == 'accurate') &
                                                          (self.results_payments.status == 'partner_left')]
        print(time.asctime(time.localtime(time.time())),
              ': number of paris one left expert estimate accurate probability:',
              partner_left_accurate.pair_id.unique().shape[0])
        logging.info('{}: number of paris one left expert estimate accurate probability: {}'.
                     format(time.asctime(time.localtime(time.time())),
                            partner_left_accurate.pair_id.unique().shape[0]))

        return

    def split_accurate(self):
        """
        This function split the accurate experts to 3 groups: really accurate, cheat a little bit up,
        cheat a little bit down
        :return:
        """

        accurate = self.results_payments.loc[(self.results_payments.prob_status == 'accurate') &
                                             (self.results_payments.player_timeout == 0) &
                                             (self.results_payments.player_id_in_group == 1)].copy()
        if accurate.empty:
            print(f'no data in accurate for gender {self.gender} in function split_accurate')
            return

        accurate = accurate.assign(cheat_up=0)

        accurate.loc[accurate.diff_prob > 0, 'cheat_up'] = 1
        accurate.loc[accurate.diff_prob < 0, 'cheat_up'] = -1

        cheat_up = accurate.loc[accurate.cheat_up == 1]
        cheat_up = pd.DataFrame(cheat_up.groupby(by='participant_code').cheat_up.count().rename('accurate_overdo'))
        cheat_down = accurate.loc[accurate.cheat_up == -1]
        cheat_down = pd.DataFrame(cheat_down.groupby(by='participant_code').cheat_up.count().rename('accurate_reduce'))
        accurate_group = accurate.loc[accurate.cheat_up == 0]
        accurate_group = pd.DataFrame(accurate_group.groupby(by='participant_code').cheat_up.count().rename('accurate'))
        final = cheat_up.merge(cheat_down, left_index=True, right_index=True)
        final = final.merge(accurate_group, left_index=True, right_index=True)

        final['highest'] = final.idxmax(axis=1)
        # change the prob status for the accurate participants
        for participant in final.index:
            self.results_payments.loc[self.results_payments.participant_code == participant, 'prob_status'] =\
                final.loc[participant, 'highest']

        return

    def split_3_values(self):
        """
        This function split the experts that gave 3 or less values into 2 groups: signal and cheat
        :return:
        """

        zero_one = self.results_payments.loc[(self.results_payments.prob_status == '3_or_less_values') &
                                             (self.results_payments.player_timeout == 0) &
                                             (self.results_payments.player_id_in_group == 1)].copy()

        if zero_one.empty:
            print(f'no data in zero_one for gender {self.gender} in function split_2_values')
            return

        # define for each lottery zero_one_ev_status and zero_one_prob_status and zero_one_average_status
        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.given_expected_value >= 0),
                     'zero_one_ev_status_round'] = 1
        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.given_expected_value < 0),
                     'zero_one_ev_status_round'] = 0
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.given_expected_value >= 0),
                     'zero_one_ev_status_round'] = 0
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.given_expected_value < 0),
                     'zero_one_ev_status_round'] = 1

        # zero-one average status per round
        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.lottery_average >= 0),
                     'zero_one_average_status_round'] = 1
        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.lottery_average < 0),
                     'zero_one_average_status_round'] = 0
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.lottery_average >= 0),
                     'zero_one_average_status_round'] = 0
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.lottery_average < 0),
                     'zero_one_average_status_round'] = 1

        # zero-one probability status per round
        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.group_p_lottery >= 0.5),
                     'zero_one_prob_status_round'] = 1
        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.group_p_lottery < 0.5),
                     'zero_one_prob_status_round'] = 0
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.group_p_lottery >= 0.5),
                     'zero_one_prob_status_round'] = 0
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.group_p_lottery < 0.5),
                     'zero_one_prob_status_round'] = 1

        # for values other than 0, 1
        zero_one.loc[(~zero_one.player_answer.isin([0, 1])) &
                     (zero_one.group_p_lottery - zero_one.player_answer <= 0.1), 'zero_one_prob_status_round'] = 1
        zero_one.loc[(~zero_one.player_answer.isin([0, 1])) &
                     (zero_one.group_p_lottery - zero_one.player_answer > 0.1), 'zero_one_prob_status_round'] = 0

        zero_one.loc[(~zero_one.player_answer.isin([0, 1])) & (zero_one.player_answer >= 0.5) &
                     (zero_one.given_expected_value >= 0), 'zero_one_ev_status_round'] = 1
        zero_one.loc[(~zero_one.player_answer.isin([0, 1])) & (zero_one.player_answer >= 0.5) &
                     (zero_one.given_expected_value < 0), 'zero_one_ev_status_round'] = 0

        zero_one.loc[(~zero_one.player_answer.isin([0, 1])) & (zero_one.player_answer < 0.5) &
                     (zero_one.given_expected_value >= 0), 'zero_one_ev_status_round'] = 0
        zero_one.loc[(~zero_one.player_answer.isin([0, 1])) & (zero_one.player_answer < 0.5) &
                     (zero_one.given_expected_value > 0), 'zero_one_ev_status_round'] = 1

        # define if the expert was accurate with its zero-one answers
        experts_sum = zero_one.groupby(by='participant_code').agg({'zero_one_ev_status_round': 'sum',
                                                                   'zero_one_prob_status_round': 'sum',
                                                                   'zero_one_average_status_round': 'sum',
                                                                   'subsession_round_number': 'count'})
        experts_sum = experts_sum.loc[experts_sum.subsession_round_number >= 30]
        experts_sum['pct_true_ev'] = experts_sum.zero_one_ev_status_round / experts_sum.subsession_round_number
        experts_sum['pct_true_prob'] = experts_sum.zero_one_prob_status_round / experts_sum.subsession_round_number
        experts_sum['pct_true_avg'] = experts_sum.zero_one_average_status_round / experts_sum.subsession_round_number

        experts_sum['zero_one_prob_status'] = np.where(experts_sum.pct_true_prob >= 0.6, 1, 0)
        experts_sum['zero_one_ev_status'] = np.where(experts_sum.pct_true_ev >= 0.6, 1, 0)
        experts_sum['zero_one_avg_status'] = np.where(experts_sum.pct_true_avg >= 0.6, 1, 0)

        # experts_sum['participant_code'] = experts_sum.index
        experts_sum = experts_sum[['zero_one_ev_status', 'zero_one_prob_status',
                                   'zero_one_avg_status']]
        self.results_payments = self.results_payments.merge(experts_sum, right_index=True, left_on='participant_code',
                                                            how='left')

        # define the prob status in results_payments:
        self.results_payments.loc[self.results_payments.zero_one_ev_status == 1, 'prob_status'] = 'zero_one_accurate_ev'
        self.results_payments.loc[self.results_payments.zero_one_prob_status == 1, 'prob_status'] =\
            'zero_one_accurate_prob'
        self.results_payments.loc[(self.results_payments.zero_one_ev_status == 1) &
                                  (self.results_payments.zero_one_prob_status == 1) &
                                  (self.results_payments.zero_one_avg_status == 1), 'prob_status'] =\
            'zero_one_accurate_all'
        self.results_payments.loc[(self.results_payments.zero_one_ev_status == 1) &
                                  (self.results_payments.zero_one_prob_status == 1), 'prob_status'] =\
            'zero_one_accurate_ev_prob'
        self.results_payments.loc[(self.results_payments.zero_one_ev_status == 1) &
                                  (self.results_payments.zero_one_avg_status == 1), 'prob_status'] =\
            'zero_one_accurate_ev_avg'
        self.results_payments.loc[self.results_payments.zero_one_avg_status == 1, 'prob_status'] =\
            'zero_one_accurate_avg'

        # not accurate for all
        self.results_payments.loc[(self.results_payments.zero_one_prob_status == 0) &
                                  (self.results_payments.zero_one_ev_status == 0) &
                                  (self.results_payments.zero_one_avg_status == 0), 'prob_status'] =\
            'zero_one_not_accurate'

        # self.results_payments.loc[(self.results_payments.zero_one_ev_status == 1) |
        #                           (self.results_payments.zero_one_prob_status == 1), 'prob_status'] =\
        #     'zero_one_accurate'
        # self.results_payments.loc[(self.results_payments.zero_one_ev_status == 0) &
        #                           (self.results_payments.zero_one_prob_status == 0), 'prob_status'] =\
        #     'zero_one_not_accurate'

        return

    def experts_always_overdo_reduce(self):
        """Check if there are experts that lie in all the rounds: overdo/reduce"""
        played = self.results_payments.loc[(self.results_payments.player_timeout == 0) &
                                           (self.results_payments.player_id_in_group == 1) &
                                           (self.results_payments.status == 'play')].copy()

        if played.empty:
            print(f'no data in played for gender {self.gender} in function experts_always_overdo_reduce')
            return

        played = played.assign(overdo_round=0)
        played = played.assign(reduce_round=0)

        played.loc[played.player_answer > played.group_p_lottery, 'overdo_round'] = 1
        played.loc[played.player_answer < played.group_p_lottery, 'reduce_round'] = 1

        bias = played.groupby(by='participant_code').agg({'overdo_round': 'sum', 'reduce_round': 'sum',
                                                          'subsession_round_number': 'count'})

        bias['pct_overdo'] = bias.overdo_round / bias.subsession_round_number
        bias['pct_reduce'] = bias.reduce_round / bias.subsession_round_number

        bias['overdo'] = np.where(bias.pct_overdo > 0.8, 1, 0)
        bias['reduce'] = np.where(bias.pct_reduce > 0.8, 1, 0)
        bias['more_overdo'] = np.where(bias.pct_overdo > bias.pct_reduce, 1, 0)
        bias['more_reduce'] = np.where(bias.pct_overdo < bias.pct_reduce, 1, 0)

        payoff = self.results_payments[['participant_code', 'total_payoff_no_partner_timeout', 'prob_status',
                                        'num_positive_given_ev', 'score_num_positive']]
        payoff = payoff.drop_duplicates('participant_code')
        zero_one_accurate_bias = bias.merge(payoff, how='left', right_on='participant_code', left_index=True)
        zero_one_accurate_bias.to_csv(os.path.join(data_analysis_directory, 'all_played_bias.csv'))

        for expert_type in ['overdo', 'reduce', 'more_overdo', 'more_reduce']:
            for overdo in [1, 0]:
                participants = bias.loc[bias[expert_type] == overdo].index
                participants_average_payoff =\
                    self.results_payments.loc[self.results_payments.participant_code.isin(participants)].\
                        total_payoff_no_partner_timeout.mean()
                participants_median_payoff =\
                    self.results_payments.loc[self.results_payments.participant_code.isin(participants)].\
                        total_payoff_no_partner_timeout.median()
                print(
                    f'Number of {expert_type} overdo = {overdo}: '
                    f'{bias.loc[bias[expert_type] == overdo].shape[0]} '
                    f'with average payoff: {participants_average_payoff} '
                    f'and median payoff: {participants_median_payoff}')

        return

    def anaylze_zero_one_accurate(self):
        """check if the accurate zero-one experts tend to overdo or reduce their estimation
        when they was not accurate"""

        zero_one_accurate_list = like_zero_one_list.copy()
        zero_one_accurate_list.remove('zero_one_not_accurate')

        zero_one = self.results_payments.loc[(self.results_payments.prob_status.isin(zero_one_accurate_list)) &
                                             (self.results_payments.player_timeout == 0) &
                                             (self.results_payments.player_id_in_group == 1) &
                                             (self.results_payments.status == 'play')].copy()

        if zero_one.empty:
            print(f'no data in zero_one for gender {self.gender} in function anaylze_zero_one_accurate')
            return

        zero_one = zero_one.assign(zero_one_ev_overdo_round=0)
        zero_one = zero_one.assign(zero_one_ev_reduce_round=0)
        zero_one = zero_one.assign(zero_one_prob_overdo_round=0)
        zero_one = zero_one.assign(zero_one_prob_reduce_round=0)

        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.given_expected_value < 0) &
                     (zero_one.prob_status.isin(['zero_one_accurate_ev', 'zero_one_accurate_all',
                                                 'zero_one_accurate_ev_prob', 'zero_one_accurate_ev_avg'])),
                     'zero_one_ev_overdo_round'] = 1
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.given_expected_value >= 0) &
                     (zero_one.prob_status.isin(['zero_one_accurate_ev', 'zero_one_accurate_all',
                                                 'zero_one_accurate_ev_prob', 'zero_one_accurate_ev_avg'])),
                     'zero_one_ev_reduce_round'] = 1

        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.lottery_average < 0) &
                     (zero_one.prob_status.isin(['zero_one_accurate_avg', 'zero_one_accurate_all',
                                                 'zero_one_accurate_ev_avg'])),
                     'zero_one_avg_overdo_round'] = 1
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.lottery_average >= 0) &
                     (zero_one.prob_status.isin(['zero_one_accurate_avg', 'zero_one_accurate_all',
                                                 'zero_one_accurate_ev_avg'])),
                     'zero_one_avg_reduce_round'] = 1

        zero_one.loc[(zero_one.player_answer == 1) & (zero_one.group_p_lottery < 0.5) &
                     (zero_one.prob_status.isin(['zero_one_accurate_prob', 'zero_one_accurate_all',
                                                 'zero_one_accurate_ev_prob'])),
                     'zero_one_prob_overdo_round'] = 1
        zero_one.loc[(zero_one.player_answer == 0) & (zero_one.group_p_lottery >= 0.5) &
                     (zero_one.prob_status.isin(['zero_one_accurate_prob', 'zero_one_accurate_all',
                                                 'zero_one_accurate_ev_prob'])),
                     'zero_one_prob_reduce_round'] = 1

        accurate_bias = zero_one.groupby(by='participant_code').agg({'zero_one_ev_overdo_round': 'sum',
                                                                     'zero_one_ev_reduce_round': 'sum',
                                                                     'zero_one_prob_overdo_round': 'sum',
                                                                     'zero_one_prob_reduce_round': 'sum',
                                                                     'zero_one_avg_overdo_round': 'sum',
                                                                     'zero_one_avg_reduce_round': 'sum',
                                                                     'subsession_round_number': 'count'})
        accurate_bias['pct_overdo_prob'] = accurate_bias.zero_one_prob_overdo_round /\
                                           accurate_bias.subsession_round_number
        accurate_bias['pct_reduce_prob'] = accurate_bias.zero_one_prob_reduce_round /\
                                           accurate_bias.subsession_round_number
        accurate_bias['pct_overdo_ev'] = accurate_bias.zero_one_ev_overdo_round /\
                                           accurate_bias.subsession_round_number
        accurate_bias['pct_reduce_ev'] = accurate_bias.zero_one_ev_reduce_round /\
                                           accurate_bias.subsession_round_number
        accurate_bias['pct_overdo_avg'] = accurate_bias.zero_one_avg_overdo_round /\
                                           accurate_bias.subsession_round_number
        accurate_bias['pct_reduce_avg'] = accurate_bias.zero_one_avg_reduce_round /\
                                           accurate_bias.subsession_round_number

        accurate_bias['zero_one_accurate_ev_overdo'] = np.where(accurate_bias.pct_overdo_ev >
                                                                accurate_bias.pct_reduce_ev, 1, 0)
        accurate_bias['zero_one_accurate_avg_overdo'] = np.where(accurate_bias.pct_overdo_avg >
                                                                 accurate_bias.pct_reduce_avg, 1, 0)
        accurate_bias['zero_one_accurate_prob_overdo'] = np.where(accurate_bias.pct_overdo_prob >
                                                                  accurate_bias.pct_reduce_prob, 1, 0)

        payoff = self.results_payments[['participant_code', 'total_payoff_no_partner_timeout', 'prob_status',
                                        'num_positive_given_ev', 'score_num_positive']]
        payoff = payoff.drop_duplicates('participant_code')
        zero_one_accurate_bias = accurate_bias.merge(payoff, how='left', right_on='participant_code', left_index=True)
        zero_one_accurate_bias.to_csv(os.path.join(data_analysis_directory, 'zero_one_accurate_bias.csv'))

        for expert_type in ['zero_one_accurate_ev_overdo', 'zero_one_accurate_avg_overdo',
                            'zero_one_accurate_prob_overdo']:
            for overdo in [1, 0]:
                participants = accurate_bias.loc[accurate_bias[expert_type] == overdo].index
                participants_average_payoff =\
                    self.results_payments.loc[self.results_payments.participant_code.isin(participants)].\
                        total_payoff_no_partner_timeout.mean()
                participants_median_payoff =\
                    self.results_payments.loc[self.results_payments.participant_code.isin(participants)].\
                        total_payoff_no_partner_timeout.median()
                print(
                    f'Number of {expert_type} overdo = {overdo}: '
                    f'{accurate_bias.loc[accurate_bias[expert_type] == overdo].shape[0]} '
                    f'with average payoff: {participants_average_payoff} '
                    f'and median payoff: {participants_median_payoff}')

        return

    def cutoff_analysis(self):
        """
        This function analysis the cutoff: from which given EV the expert become consistence with its 1-0 answer.
        First define the 1-0 answer for the non zero-one groups.
        Then find the cutoff for each expert.
        Then create a graph of the cutoff and the expert's payoff.
        :return:
        """

        # define the 1-0 for the non zero-one groups:
        curr_like_zero_one_list = like_zero_one_list + ['3_or_less_values']
        group_list = self.results_payments.prob_status.unique().tolist()
        for item in curr_like_zero_one_list + ['', 'one_value']:
            if item in group_list:
                group_list.remove(item)

        experts_to_define = self.results_payments.loc[(self.results_payments.player_id_in_group == 1) &
                                                      (self.results_payments.prob_status.isin(group_list))].\
            participant_code.unique()

        if experts_to_define.shape[0] == 0:
            print(f'no data in experts_to_define for gender {self.gender} in function cutoff_analysis')
            return

        self.results_payments.loc[(self.results_payments.participant_code.isin(experts_to_define)) &
                                  (self.results_payments.expert_estimate_expected_value >= 0), 'zero_one_answer'] = 1
        self.results_payments.loc[(self.results_payments.participant_code.isin(experts_to_define)) &
                                  (self.results_payments.expert_estimate_expected_value < 0), 'zero_one_answer'] = 0

        # define for zero-one experts:
        zero_one_experts = self.results_payments.loc[(self.results_payments.player_id_in_group == 1) &
                                                     (self.results_payments.prob_status.isin(curr_like_zero_one_list))].\
            participant_code.unique()

        self.results_payments.loc[(self.results_payments.participant_code.isin(zero_one_experts)) &
                                  (self.results_payments.player_answer.isin([0, 1])), 'zero_one_answer'] =\
            self.results_payments.player_answer
        self.results_payments.loc[(self.results_payments.participant_code.isin(zero_one_experts)) &
                                  (self.results_payments.player_answer < 0.5), 'zero_one_answer'] = 0
        self.results_payments.loc[(self.results_payments.participant_code.isin(zero_one_experts)) &
                                  (self.results_payments.player_answer >= 0.5), 'zero_one_answer'] = 1

        # find the cutoff
        for column in ['given_expected_value', 'group_p_lottery']:
            data_to_analyze = self.results_payments.loc[(self.results_payments.zero_one_answer.isin([1, 0])) &
                                                        (self.results_payments.status == 'play') &
                                                        (self.results_payments.player_timeout == 0)]

            if data_to_analyze.empty:
                print(f'no data in data_to_analyze for gender {self.gender} in function cutoff_analysis')
                return

            data_to_analyze = data_to_analyze[['participant_code', 'zero_one_answer', column]]
            data_to_analyze = data_to_analyze.sort_values(by=column, ascending=False)
            data_to_analyze.zero_one_answer = data_to_analyze.zero_one_answer.astype(np.int64)
            pivot = data_to_analyze.pivot_table(index='participant_code', columns=column)
            pivot.columns = pivot.columns.levels[1]
            pivot['cutoff_' + column] = ''
            pivot['pct_mistakes_' + column] = ''

            # for each participant, check for each EV what is its best cutoff, and the 0.9 cutoff
            all_participants_cutoff = pd.DataFrame()
            for participant in range(pivot.shape[0]):
                participant_number_of_miss = math.inf
                participant_data = pd.DataFrame(pivot.iloc[participant])
                participant_data = participant_data.drop(['pct_mistakes_' + column, 'cutoff_' + column]).dropna().\
                    sort_index(ascending=False)
                participant_data = participant_data.assign(average_above='')
                participant_data = participant_data.assign(average_below='')
                participant_data['participant_code'] = participant_data.columns[0]
                participant_code = participant_data.columns[0]
                participant_data.columns = [['zero_one_answer', 'average_above', 'average_below', 'participant_code']]

                for ev in range(participant_data.shape[0], 0, -1):
                    number_of_miss = participant_data.loc[((participant_data.index >= participant_data.index[ev-1]) &
                                                           (participant_data.zero_one_answer == 0)) |
                                                          ((participant_data.index < participant_data.index[ev-1]) &
                                                           (participant_data.zero_one_answer == 1))]
                    number_of_miss = number_of_miss.shape[0]
                    ev_cutoff =\
                        participant_data.iloc[:ev].sum()['zero_one_answer'] /\
                        participant_data.iloc[:ev].count()['zero_one_answer']
                    participant_data.loc[participant_data.index[ev-1], 'average_above'] = ev_cutoff
                    participant_data.loc[participant_data.index[ev-1], 'average_below'] =\
                        participant_data.iloc[ev:].sum()['zero_one_answer'] /\
                        participant_data.iloc[ev:].count()['zero_one_answer']
                    participant_data.loc[participant_data.index[ev-1], 'number_of_miss'] = number_of_miss
                    #
                    # if ev_cutoff > cutoff_value:
                    #     if not cutoff_09 and ev_cutoff >= 0.9:  # create cutoff of 0.9
                    #         pivot.loc[participant_code, 'cutoff_09_' + column] =\
                    #             participant_data.index[ev - 1]
                    #         pivot.loc[participant_code, 'pct_mistakes_09_' + column] = ev_cutoff
                    #         cutoff_09 = True
                    #     cutoff_value = ev_cutoff
                    #     pivot.loc[participant_code, 'cutoff_' + column] = participant_data.index[ev-1]
                    #     pivot.loc[participant_code, 'pct_mistakes_' + column] = ev_cutoff

                    # the cutoff will be the average between the current cutoff and the EV below it.
                    if ev < participant_data.index.shape[0]:
                        cutoff = (participant_data.index[ev - 1] + participant_data.index[ev]) / 2
                    else:
                        cutoff = participant_data.index[ev - 1]

                    if number_of_miss < participant_number_of_miss:
                        participant_number_of_miss = number_of_miss
                        pivot.loc[participant_code, 'cutoff_' + column] = cutoff
                        pivot.loc[participant_code, 'pct_mistakes_' + column] = number_of_miss /\
                                                                                participant_data.shape[0]
                    elif number_of_miss == participant_number_of_miss:
                        # if there are 2 cutoffs with the same number of miss, take the average
                        average_cutoff = (pivot.loc[participant_code, 'cutoff_' + column] + cutoff) / 2
                        pivot.loc[participant_code, 'cutoff_' + column] = average_cutoff
                        pivot.loc[participant_code, 'pct_mistakes_' + column] = number_of_miss /\
                                                                                participant_data.shape[0]

                all_participants_cutoff = all_participants_cutoff.append(participant_data)
            all_participants_cutoff.to_csv(os.path.join(data_analysis_directory, 'all_participants_cutoff.csv'))

            # merge with results_payments
            pivot_to_merge = pivot[['pct_mistakes_' + column, 'cutoff_' + column]]
            self.results_payments = self.results_payments.merge(pivot_to_merge, how='left', left_on='participant_code',
                                                                right_index=True)

            # create plot:
            # experts that played and have zero_one_answer
            data_to_plot = self.results_payments.loc[(self.results_payments.status == 'play') &
                                                     (self.results_payments.player_id_in_group == 1) &
                                                     (self.results_payments.zero_one_answer.isin([1, 0]))].\
                drop_duplicates(subset='participant_code')

            for threshold in [1, 0.3]:
                data_to_plot = data_to_plot.loc[data_to_plot['pct_mistakes_' + column] <= threshold]
                data_to_plot = data_to_plot[['pct_mistakes_' + column, 'cutoff_' + column, 'participant_code',
                                             'total_payoff_no_partner_timeout', 'prob_status', 'total_payoff']]
                legend = data_to_plot.prob_status.unique().tolist()
                create_data_for_plot_all_experts(data_to_plot, xlabel='cutoff_' + column,
                                                 ylabel='total_payoff_no_partner_timeout', legend=legend,
                                                 title='cutoff vs total payoff (no partner timeout) - best cutoff '
                                                       + column + ' less than ' + str(threshold*100) + '% mistakes',
                                                 batch_with_line=False)
                create_data_for_plot_all_experts(data_to_plot, xlabel='cutoff_' + column,
                                                 ylabel='total_payoff', legend=legend,
                                                 title='cutoff vs total payoff - best cutoff ' +
                                                       column + ' less than ' + str(threshold*100) + '% mistakes',
                                                 batch_with_line=False)

        data_to_save = self.results_payments.loc[(self.results_payments.status == 'play') &
                                                 (self.results_payments.player_id_in_group == 1) &
                                                 (self.results_payments.zero_one_answer.isin([1, 0]))][
            ['pct_mistakes_given_expected_value', 'cutoff_given_expected_value', 'pct_mistakes_group_p_lottery',
             'cutoff_group_p_lottery', 'participant_code', 'player_answer', 'zero_one_answer',
             'total_payoff_no_partner_timeout', 'prob_status', 'given_expected_value', 'group_p_lottery',
             'group_x_lottery', 'group_y_lottery', 'total_payoff']]
        data_to_save.to_csv(os.path.join(data_analysis_directory, f'cutoff_analysis_data {self.gender}.csv'))

        return

    def check_payoff_vs_average_ev(self):
        """
        This function create a plot of the total payoff of the expert vs the average EV.
        :return:
        """

        # create average EV column - take not partner timeout rows
        data_for_average = self.results_payments.loc[(self.results_payments.status == 'play') &
                                                     (self.results_payments.player_id_in_group == 1) &
                                                     (self.results_payments.partner_timeout == 0)]

        if data_for_average.empty:
            print(f'no data in data_for_average for gender {self.gender} in function check_payoff_vs_average_ev')
            return

        data_for_average['positive_given_ev'] =\
            data_for_average.apply(lambda row: 1 if row['given_expected_value'] >= 0 else 0, axis=1)
        data_for_average = pd.DataFrame(data_for_average.groupby(by='participant_code').positive_given_ev.mean())
        # data_for_average['participant_code'] = data_for_average.index
        data_for_average.columns = ['pct_positive_given_ev']
        self.results_payments = self.results_payments.merge(data_for_average, how='left', on='participant_code')

        # create plot:
        # experts that played and have zero_one_answer
        data_to_plot = self.results_payments.loc[(self.results_payments.status == 'play') &
                                                 (self.results_payments.player_id_in_group == 1)]. \
            drop_duplicates(subset='participant_code')
        print('min % of trials with positive given expected value is:', data_to_plot['pct_positive_given_ev'].min())
        print('max % of trials with positive given expected value is:', data_to_plot['pct_positive_given_ev'].max())

        data_to_plot = data_to_plot[['pct_positive_given_ev', 'total_payoff_no_partner_timeout', 'prob_status']]
        legend = data_to_plot.prob_status.unique().tolist()
        create_data_for_plot_all_experts(data_to_plot, xlabel='pct_positive_given_ev',
                                         ylabel='total_payoff_no_partner_timeout',
                                         legend=legend, batch_with_line=False,
                                         title=f'% of trials with positive given expected value'
                                               f'vs total payoff for gender {self.gender} (no partner timeout)')

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
            print(f'no data in data_for_average for gender {self.gender} in function effectiveness_measure')
            return

        data_for_average['num_positive_given_ev'] =\
            data_for_average.apply(lambda row: 1 if row['given_expected_value'] >= 0 else 0, axis=1)
        # count the number of trials with positive EV
        group_data_for_average = pd.DataFrame(data_for_average.groupby(by='participant_code').
                                              num_positive_given_ev.sum())
        group_data_for_average['participant_code'] = group_data_for_average.index
        group_data_for_average = group_data_for_average.reset_index(drop=True)

        # option 1: compute the score relative to the mean number of trial with positive EV
        average_num_positive_ev = group_data_for_average.num_positive_given_ev.mean()
        data_to_merge = data_for_average[['total_payoff_no_partner_timeout', 'participant_code']].drop_duplicates()
        group_data_for_average = group_data_for_average.merge(data_to_merge, how='left', on='participant_code')

        group_data_for_average['score_average_num_positive'] = group_data_for_average.total_payoff_no_partner_timeout -\
                                                               average_num_positive_ev
        group_data_for_average['score_num_positive'] = group_data_for_average.total_payoff_no_partner_timeout -\
                                                       group_data_for_average.num_positive_given_ev

        group_data_for_average = group_data_for_average[['num_positive_given_ev', 'participant_code',
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
            print(f'no data in data_for_average for gender {self.gender} in function round_analysis')
            return

        data_of_left = data_of_left[['participant_code', 'subsession_round_number', 'player_timeout']]
        pivot = data_of_left.pivot_table(index='participant_code', columns='subsession_round_number')
        pivot.columns = np.arange(1, 51)
        pivot = pivot.assign(left_round='')
        for i in range(50, 0, -1):
            pivot['sum_' + str(i)] = pivot.iloc[:, i:50].sum(axis=1)
            pivot['sum_' + str(i)] = 50 - pivot['sum_' + str(i)]
            pivot.loc[(pivot['sum_' + str(i)] == i) & (pivot[i] == 1), 'left_round'] = i

        pivot_to_merge = pivot[['left_round']]
        create_histogram(title='trial_of_left', x=np.array(pivot_to_merge), xlabel='trial_number',
                         ylabel='number_of_participants', add_labels=True)
        self.results_payments = self.results_payments.merge(pivot_to_merge, how='left', left_on='participant_code',
                                                            right_index=True)

        # get the max number of timeouts that is not left
        max_num_rounds_timeout = 0
        max_num_rounds_timeout_participant_code = ''
        # get all players that play
        data_max_timeout = self.results_payments.loc[self.results_payments.status.isin(['left', 'both_left', 'play',
                                                                                        'partner_left'])]

        if data_max_timeout.empty:
            print(f'no data in data_max_timeout for gender {self.gender} in function round_analysis')
            return

        data_max_timeout = data_max_timeout[['participant_code', 'subsession_round_number', 'player_timeout']]
        pivot = data_max_timeout.pivot_table(index='participant_code', columns='subsession_round_number')
        pivot.columns = np.arange(1, 51)
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
                max_num_round = 50
            self.results_payments.loc[self.results_payments.participant_code == participant_code,
                                      'max_num_rounds_timeout'] = max_num_round
            if (max_num_round > max_num_rounds_timeout) and (max_num_round < 50):  # set the max number of rounds
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
            print(f'no data in data_to_visual for gender {self.gender} in function round_analysis')
            return

        data_to_visual = data_to_visual.drop_duplicates(subset=['participant_code'])
        data_to_visual = data_to_visual.loc[data_to_visual.left_round != 1]
        data_to_visual = data_to_visual['max_num_rounds_timeout']
        create_histogram(title=f'max_number_of_timeout_trials_before_left for gender {self.gender}',
                         x=np.array(data_to_visual), xlabel='number_of_timeout_trials',
                         ylabel='number_of_participants', add_labels=True)

        # timeouts
        for status in [[['play', 'partner_left'], 'finish'], [['left', 'both_left', 'play', 'partner_left'], 'all']]:
            for role in [[[1], 'expert'], [[2], 'decision_maker'], [[1, 2], 'all']]:
                data_to_use = self.results_payments.loc[self.results_payments.status.isin(status[0])]

                if data_to_use.empty:
                    print(f'no data in data_to_use for gender {self.gender} in function round_analysis')
                    continue

                role_participants = data_to_use.loc[data_to_use.player_id_in_group.isin(role[0])]
                role_timeout = role_participants.groupby(by='participant_code')['player_timeout'].sum()
                role_timeout = (role_timeout / 50) * 100
                x = np.array(role_timeout.values)
                create_histogram(title=role[1] + '_timeout_' + status[1], x=x, xlabel='pct_timeout',
                                 ylabel='number_of_participants', add_labels=True)

        role_timeout = pd.DataFrame(role_timeout)
        # role_timeout['participant_code'] = role_timeout.index
        role_timeout.columns = ['pct_timeout']
        self.results_payments = self.results_payments.merge(role_timeout, right_index=True, left_on='participant_code',
                                                            how='left')
        self.results_payments.loc[(self.results_payments.pct_timeout >= 80) & (self.results_payments.status == 'play'),
                                  'status'] = 'more_than_80_pct_timeout'
        play_always_timeout = self.results_payments.loc[self.results_payments.status == 'more_than_80_pct_timeout']
        participants_always_timeout = play_always_timeout.participant_code.unique()
        print(time.asctime(time.localtime(time.time())), ': number of participants that played and always had timeout:',
              participants_always_timeout.shape[0])
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
        partner_left_bonus = partner_left.loc[partner_left.bonus > 1]
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
                print(f'no data in role_started for gender {self.gender} in function bonus_analysis')
                continue

            number_of_started = role_started.participant_code.unique()
            # participants with role[0] that got bonus
            role_data_bonus = role_started.loc[role_started.bonus >= 1]
            # participants with role[0] that played but didn't got bonus
            role_data_no_bonus = role_started.loc[(role_started.bonus >= 0) & (role_started.bonus < 1)]
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
            print(f'no data in play_participants for gender {self.gender} in function '
                  f'define_total_payoff_no_partner_timeout')
            return

        for total_payoff_type in type_list:
            if total_payoff_type == 'total_payoff_no_partner_timeout':
                play_participants = play_participants.loc[play_participants.partner_timeout == 0]
            total_payoff = pd.DataFrame(play_participants.groupby(by='participant_code').player_payoff.sum())
            # total_payoff['participant_code'] = total_payoff.index
            total_payoff.columns = [total_payoff_type]
            self.results_payments = self.results_payments.merge(total_payoff, how='left', left_on='participant_code',
                                                                right_index=True)

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
            accurate_participants = self.results_payments.loc[(self.results_payments.prob_status == 'accurate') &
                                                              (self.results_payments.player_id_in_group == 1) &
                                                              (self.results_payments.status == 'play') &
                                                              (self.results_payments.pct_timeout <= 50)]

            if accurate_participants.empty:
                print(f'no data in accurate_participants for gender {self.gender} in function expert_payoff_analysis')
                continue

            if total_payoff_type == 'total_payoff_no_partner_timeout':
                accurate_participants = accurate_participants.loc[accurate_participants.partner_timeout == 0]
            accurate_participants = accurate_participants.drop_duplicates('participant_code')
            accurate_average_payoff = round(accurate_participants[total_payoff_type].mean(), 2)

            play_participants_groups = \
                self.results_payments.loc[(self.results_payments.status == 'play') &
                                          (self.results_payments.player_id_in_group == 1) &
                                          (self.results_payments.pct_timeout <= 50)]

            if play_participants_groups.empty:
                print(f'no data in play_participants_groups for gender {self.gender} in function expert_payoff_analysis')
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
            create_chart_bars(title=f'expert {total_payoff_type} with average for gender {self.gender}',
                              x=x_points, y=y_points,
                              xlabel='expert_total_payoff', ylabel='number_of_participants', legend=legend,
                              add_point=accurate_average_payoff)

            create_chart_bars(title=f'expert {total_payoff_type} for gender {self.gender}', x=x_points, y=y_points,
                              xlabel='expert_total_payoff', ylabel='number_of_participants', legend=legend)

        writer = pd.ExcelWriter(os.path.join(data_analysis_directory, f'groups_payoff_statistics_{self.gender}.xlsx'))
        for total_payoff_type in type_list:
            groups_payoff_statistics_dict[total_payoff_type].to_excel(writer, sheet_name=total_payoff_type)
        writer.save()

        return

    def calculate_expected_value(self):
        """
        This function calculate the real and the expert estimation expected value
        :return:
        """

        xlabel = 'given_expected_value'
        ylabel = 'expert_estimate_expected_value'

        self.results_payments[xlabel] =\
            (self.results_payments.group_x_lottery * self.results_payments.group_p_lottery) + \
            (self.results_payments.group_y_lottery * (1 - self.results_payments.group_p_lottery))
        self.results_payments[ylabel] =\
            (self.results_payments.group_x_lottery * self.results_payments.group_sender_answer) + \
            (self.results_payments.group_y_lottery * (1 - self.results_payments.group_sender_answer))
        self.results_payments['lottery_average'] = 0.5 * (self.results_payments.group_x_lottery +
                                                          self.results_payments.group_y_lottery)

        # above, below or on the line
        self.results_payments.loc[self.results_payments[xlabel] < self.results_payments[ylabel], 'line'] = 'above'
        self.results_payments.loc[self.results_payments[xlabel] > self.results_payments[ylabel], 'line'] = 'below'
        self.results_payments.loc[(self.results_payments[xlabel] == self.results_payments[ylabel]) |
                                  (self.results_payments.abs_diff_prob <= 0.1), 'line'] = 'on'

        return

    def plot_expected_analysis(self):
        """
        Plot the given expected value vs the expert estimate expected value,
        and the given EV vs the expert estimate probability
        :return:
        """

        xlabel = 'given_expected_value'
        ylabel = 'expert_estimate_expected_value'

        line_numbers = list()
        batch_average = list()
        # all points of experts that have less than 50% timeout, they play and the DM answer (no timeout)
        data_to_plot = self.results_payments.loc[(self.results_payments.pct_timeout <= 50) &
                                                 (self.results_payments.player_id_in_group == 1) &
                                                 (self.results_payments.status.isin(['play', 'partner_left'])) &
                                                 (self.results_payments.partner_timeout == 0) &
                                                 (self.results_payments.player_timeout == 0)]

        if data_to_plot.empty:
            print(f'no data in data_to_plot for gender {self.gender} in function plot_expected_analysis')
            return

        # define participants to have prob status = other if they didn't get any other status
        legend = data_to_plot.prob_status.unique().tolist()
        data_to_plot.loc[data_to_plot.prob_status == '', 'prob_status'] = 'other'
        other_participant = data_to_plot.loc[data_to_plot.prob_status == 'other'].participant_code.unique()
        self.results_payments.loc[self.results_payments.participant_code.isin(other_participant), 'prob_status'] =\
            'other'

        for item in ['3_or_less_values', '', 'other']:
            if item in legend:
                legend.remove(item)

        for i in range(len(legend)):
            data = data_to_plot.loc[data_to_plot.prob_status == legend[i]]
            if data.empty:
                print(f'no data in data for gender {self.gender} in function plot_expected_analysis')
                continue
            line_numbers.append(line_analysis(data))
            batch_average.append(diff_prob_line_analysis(data, diff_prob=False))

        # create x and y points lists
        # expected value analysis: for each group the took and didn't take points
        create_data_for_plot_2_took_didnt_take(data_to_plot=data_to_plot, xlabel=xlabel, ylabel=ylabel,
                                               line_numbers=line_numbers, legend=legend,
                                               title=f'expected value analysis for {self.gender}',
                                               batch_average=batch_average, batch_with_line=True)

        batch_average = list()
        for i in range(len(legend)):
            data = data_to_plot.loc[data_to_plot.prob_status == legend[i]]
            if data.empty:
                print(f'no data in data for gender {self.gender} in function plot_expected_analysis')
                continue
            batch_average.append(batch_analysis(data))

        # create x and y points lists
        # expected value vs expert answer analysis: for each group the took and didn't take points
        create_data_for_plot_2_took_didnt_take(data_to_plot=data_to_plot, xlabel=xlabel, ylabel='group_sender_answer',
                                               legend=legend, batch_average=batch_average,
                                               title=f'expected value vs expert p estimation for {self.gender}')

        batch_average = list()
        for i in range(len(legend)):
            data = data_to_plot.loc[data_to_plot.prob_status == legend[i]]
            if data.empty:
                print(f'no data in data for gender {self.gender} in function plot_expected_analysis')
                continue
            batch_average.append(diff_prob_line_analysis(data))

        # create x and y points lists
        # expected value vs diff_prob analysis: for each group the took and didn't take points
        create_data_for_plot_2_took_didnt_take(data_to_plot=data_to_plot, xlabel=xlabel, ylabel='diff_prob',
                                               legend=legend, title=f'expected value vs difference between \n' 
                                                                    f' expert estimation for p and given p for '
                                                                    f'{self.gender}',
                                               batch_average=batch_average, batch_with_line=True)

        # both took and didn't take. All experts on one plot.
        line_numbers = line_analysis(data_to_plot)
        batch_average = diff_prob_line_analysis(data_to_plot, diff_prob=False)
        create_data_for_plot_all_experts(data_to_plot, xlabel, ylabel, legend, 'expected value analysis', batch_average,
                                         line_numbers)

        return

    def probability_for_one(self):
        """
        This function calculate for each EV the probability that expert reveal 1 --> only for zero_one experts.
        Then create a chart bar
        :return:
        """
        # all points of experts that have less than 50% timeout, they play and the DM answer (no timeout)
        data_to_plot = self.results_payments.loc[(self.results_payments.pct_timeout <= 50) &
                                                 (self.results_payments.player_id_in_group == 1) &
                                                 (self.results_payments.status.isin(['play', 'partner_left'])) &
                                                 (self.results_payments.partner_timeout == 0) &
                                                 (self.results_payments.player_timeout == 0)]

        if data_to_plot.empty:
            print(f'no data in data_to_plot for gender {self.gender} in function probability_for_one')
            return

        legend = like_zero_one_list

        for i in range(len(legend)):
            data = data_to_plot.loc[data_to_plot.prob_status == legend[i]]
            if data.empty:
                print(f'no data in data for gender {self.gender} in function probability_for_one')
                continue
            batch_average_dict = batch_analysis(data, batch_size_func=1, group=legend[i])
            y = list(batch_average_dict.values())
            x = np.array(range(-20, 21))
            create_chart_bars(title=f'Expected value vs the probability of expert answer = 1\n for the group: '
                                    f'{legend[i]} and {self.gender}', x=[x], y=[y], xlabel='expected value',
                              ylabel='probability of expert answer = 1')

        return

    def create_table_prob_ev(self):
        """
        This function built a table with the prob of each problem in the rows, the EV in the columns and values are
        the average estimation or the average bias from the real prob
        :return:
        """

        all_data_to_plot = self.results_payments.loc[(self.results_payments.pct_timeout <= 50) &
                                                     (self.results_payments.player_id_in_group == 1) &
                                                     (self.results_payments.status.isin(['play', 'partner_left'])) &
                                                     (self.results_payments.player_timeout == 0)]

        if all_data_to_plot.empty:
            print(f'no data in all_data_to_plot for gender {self.gender} in function create_table_prob_ev')
            return

        group_list = all_data_to_plot.prob_status.unique().tolist()
        group_list.append('all')

        for group in group_list:
            if group != 'all':
                data_to_plot = all_data_to_plot.loc[all_data_to_plot.prob_status == group]
                if data_to_plot.empty:
                    print(f'no data in data_to_plot for gender {self.gender} in function create_table_prob_ev')
                    continue
            else:
                data_to_plot = all_data_to_plot

            data_to_plot = data_to_plot[['given_expected_value', 'group_p_lottery', 'expert_estimate_expected_value',
                                         'player_answer', 'diff_prob']]
            data_to_plot = data_to_plot.round(2)

            for name_func in [['average', np.mean], ['median', np.median]]:
                average_est_pivot = pd.pivot_table(data_to_plot, values='player_answer', index='group_p_lottery',
                                                   columns='given_expected_value', aggfunc=name_func[1])
                average_est_pivot.to_csv(os.path.join(data_analysis_directory, f'average_est_pivot_{self.gender}.csv'))

                # average_bias_pivot = pd.pivot_table(data_to_plot, values='diff_prob', index='group_p_lottery',
                #                                     columns='given_expected_value', aggfunc=name_func[1])
                # average_bias_pivot.to_csv(os.path.join(data_analysis_directory, date_directory, 'table_prob',
                #                                        f'{name_func[0]}_bias_pivot.csv'))

                legend = ['Average estimation', 'Average bias from p']
                ylabel = 'Average value'
                xlabel = 'Given expected value'

                for prob in average_est_pivot.index.unique().tolist():
                    points_y = average_est_pivot.loc[average_est_pivot.index == prob].values.tolist()
                    points_x = [average_est_pivot.loc[average_est_pivot.index == prob].columns.tolist()]
                    # points_x.append(average_bias_pivot.loc[average_est_pivot.index == prob].columns.tolist())
                    # points_y.append(average_bias_pivot.loc[average_est_pivot.index == prob].values.tolist()[0])
                    title = f'{name_func[0]} estimation and average bias from given p - for ' \
                            f'p={prob} and group {group} and gender {self.gender}'

                    create_point_plot(points_x, points_y, legend, title, xlabel, ylabel, add_text=['', ''],
                                      add_line_points=True, prob_value=prob, inner_dir='table_prob')

        # calculate per participant
        for participant in all_data_to_plot.participant_code.unique():
            data_to_plot = all_data_to_plot.loc[all_data_to_plot.participant_code == participant]
            if data_to_plot.empty:
                print(f'no data in data_to_plot for gender {self.gender} in function create_table_prob_ev')
                continue
            data_to_plot = data_to_plot[['given_expected_value', 'group_p_lottery', 'expert_estimate_expected_value',
                                         'player_answer', 'diff_prob', 'prob_status']]
            data_to_plot = data_to_plot.round(2)
            data_to_plot = data_to_plot.sort_values(by=['given_expected_value', 'group_p_lottery'])

            legend = ['Expert estimation', 'Bias from p']
            ylabel = 'Expert estimation'
            xlabel = 'Given expected value'

            for prob in data_to_plot.group_p_lottery.unique().tolist():
                prob_status = data_to_plot.prob_status.unique()[0]
                points_y = [data_to_plot.loc[data_to_plot.group_p_lottery == prob].player_answer.tolist()]
                points_x = [data_to_plot.loc[data_to_plot.group_p_lottery == prob].given_expected_value.tolist()]
                # points_x.append(data_to_plot.loc[data_to_plot.group_p_lottery == prob].given_expected_value.tolist())
                # points_y.append(data_to_plot.loc[data_to_plot.group_p_lottery == prob].diff_prob.tolist())
                title = f'Expert estimation and average bias from given p - for p={prob} and participant {participant} ' \
                        f'with prob status {prob_status} and gender {self.gender}'

                create_point_plot(points_x, points_y, legend, title, xlabel, ylabel, add_text=['', ''],
                                  add_line_points=True, prob_value=prob, inner_dir='table_prob_per_user_no_bias')

        return

    def not_zero_one_changed_to_zero_one(self):

        # check how many always answer 0/1
        like_zero_one_list_to_check = like_zero_one_list + ['3_or_less_values']
        data_to_analyze =\
            self.results_payments.loc[(~self.results_payments.prob_status.isin(like_zero_one_list_to_check)) &
                                      (self.results_payments.subsession_round_number >= 25) &
                                      (self.results_payments.player_timeout == 0) &
                                      (self.results_payments.status.isin(['play', 'partner_left'])) &
                                      (self.results_payments.player_id_in_group == 1)]

        if data_to_analyze.empty:
            print(f'no data in data_to_plot for gender {self.gender} in function not_zero_one_changed_to_zero_one')
            return

        zero_one = data_to_analyze.groupby(by='participant_code').player_answer
        zero_one = zero_one.unique()
        zero_one_len = zero_one.str.len()

        # played with 2 or 3 values
        short = zero_one_len.loc[zero_one_len.isin([2, 3])].index
        short = zero_one.loc[zero_one.index.isin(short)]
        participants_changed = list()
        # go over all the participants with 2 or 3 values and check is 0/1 in their lists
        for participant in short.iteritems():
            if 0.0 in participant[1] and 1.0 in participant[1]:
                participants_changed.append(participant[0])

        print(f'Number of participants changed to zero one {len(participants_changed)}')

        return

    def define_prob_status_for_dm(self):
        """assign the prob status of the expert to its decision maker partner"""
        expert_tyeps = self.results_payments.loc[self.results_payments.player_id_in_group == 1][
            ['pair_id', 'prob_status']].drop_duplicates('pair_id')
        expert_tyeps.columns = ['pair_id', 'player_expert_type']
        self.results_payments = self.results_payments.merge(expert_tyeps, on='pair_id')

        return


def main(main_gender=None):
    data_analysis_obj = DataAnalysis(main_gender)
    data_analysis_obj.arrange_player_data()
    data_analysis_obj.define_player_status()
    data_analysis_obj.round_analysis()
    data_analysis_obj.how_much_pay()
    data_analysis_obj.print_pairs()
    data_analysis_obj.probability_analysis()
    data_analysis_obj.calculate_expected_value()
    data_analysis_obj.split_accurate()
    data_analysis_obj.split_3_values()
    data_analysis_obj.bonus_analysis()
    data_analysis_obj.plot_expected_analysis()
    data_analysis_obj.time_spent_analysis()
    data_analysis_obj.define_total_payoff_no_partner_timeout()
    data_analysis_obj.effectiveness_measure()
    data_analysis_obj.expert_payoff_analysis()
    data_analysis_obj.anaylze_zero_one_accurate()
    data_analysis_obj.not_zero_one_changed_to_zero_one()
    # data_analysis_obj.probability_for_one()
    # data_analysis_obj.cutoff_analysis()
    data_analysis_obj.check_payoff_vs_average_ev()
    data_analysis_obj.experts_always_overdo_reduce()
    data_analysis_obj.define_prob_status_for_dm()
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
            main(gender)

    else:
        main()
