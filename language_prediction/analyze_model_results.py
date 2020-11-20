import pandas as pd
import os
from collections import defaultdict
from os import listdir
from os.path import isfile, join, isdir
import joblib
from utils import *
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np
import sklearn.metrics as metrics

base_directory = os.path.abspath(os.curdir)

final_total_payoff_prediction_column = 'final_total_payoff_prediction'
total_payoff_label_column = 'total_payoff_label'
raisha_column_name = 'raisha'


def combine_models_results(folder_list: list, total_payoff_true_label: pd.DataFrame, baseline: bool=False):
    """Combine models results to one excel"""
    all_files_results = dict()
    all_models_results = pd.DataFrame()
    skip = False
    if not skip:
        for folder in folder_list:
            folder_path = os.path.join(base_directory, 'logs', folder)
            for inner_folder in listdir(folder_path):  # fold_number
                if inner_folder == '.DS_Store':
                    continue
                # if inner_folder not in ['fold_4', 'fold_5']:
                #     continue
                if isdir(join(folder_path, inner_folder, 'excel_models_results')):
                    files_path = join(folder_path, inner_folder, 'excel_models_results')
                elif join(folder_path, inner_folder):
                    files_path = join(folder_path, inner_folder)
                else:
                    print('No correct directory')
                    return
                folder_files = [f for f in listdir(files_path) if isfile(join(files_path, f))]
                for file in folder_files:
                    if file != '.DS_Store' and '_all_models' not in file:
                        print(f'load file {file}')
                        df = pd.read_excel(os.path.join(files_path, file), sheet_name='Model results', skiprows=[0],
                                           index_col=0)
                        df = df.assign(fold=inner_folder)
                        if baseline:
                            df.model_type = df.model_name
                        # in avg_turn models - we should have both Bin_Accuracy and Bin_Accuracy_seq
                        if 'avg_turn' in df.model_type.unique()[0] and 'Bin_Accuracy_seq' not in df.columns\
                                and 'Bin_Accuracy' in df.columns:
                            df['Bin_Accuracy_seq'] = df.Bin_Accuracy
                            model_num = df.model_num.unique()[0]
                            all_prediction = pd.read_excel(os.path.join(files_path, file),
                                                           sheet_name=f'Model_{model_num}_reg', skiprows=[0],
                                                           index_col=0, parse_dates=True)
                            all_prediction = all_prediction.loc[all_prediction.is_train == False]
                            final_total_payoff_prediction = all_prediction[final_total_payoff_prediction_column]
                            total_payoff_label = all_prediction[total_payoff_label_column]
                            if type(total_payoff_label.values[0]) == str:  # tensor(value, cuda=...) instead of float
                                total_payoff_label = all_prediction[['sample_id']].merge(
                                    total_payoff_true_label, on='sample_id')
                                total_payoff_label.index = total_payoff_label.sample_id
                                total_payoff_label = total_payoff_label[inner_folder]
                                all_prediction.index = all_prediction.sample_id
                            bin_predictions, bin_label = create_bin_columns(final_total_payoff_prediction,
                                                                            total_payoff_label, hotel_label_0=True)
                            accuracy = metrics.accuracy_score(bin_label, bin_predictions)
                            df.loc[(df.Raisha == 'All_raishas') & (df.Round == 'All_rounds'), f'Bin_Accuracy'] =\
                                round(accuracy * 100, 2)
                            raisha_options = all_prediction[raisha_column_name].unique()
                            bin_predictions = pd.DataFrame(bin_predictions).join(all_prediction[[raisha_column_name]],
                                                                                 how='left')
                            bin_label = pd.DataFrame(bin_label).join(all_prediction[[raisha_column_name]], how='left')
                            for raisha in raisha_options:
                                raisha_bin_label = bin_label.loc[bin_label[raisha_column_name] == raisha]
                                raisha_bin_predictions = bin_predictions.loc[
                                    bin_predictions[raisha_column_name] == raisha]
                                raisha_accuracy = metrics.accuracy_score(raisha_bin_label['bin_label'],
                                                                  raisha_bin_predictions['bin_predictions'])
                                df.loc[(df.Raisha == f'raisha_{raisha}') & (df.Round == 'All_rounds'), f'Bin_Accuracy']\
                                    = round(raisha_accuracy * 100, 2)

                        all_files_results[f'{folder}_{file}'] = df

        joblib.dump(all_files_results, os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.pkl'))
    else:
        all_files_results = joblib.load(os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.pkl'))

    for key, value in all_files_results.items():
        value.index = [key]*value.shape[0]
        all_models_results = pd.concat([all_models_results, value], sort='False')

    # all_files_results = pd.DataFrame.from_dict(all_files_results, orient='index')
    # all_files_results = all_files_results.drop_duplicates()
    new_results_path = os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.csv')
    all_models_results.to_csv(new_results_path)
    return f'all_server_results_{folder_list}.csv'


def combine_models_predictions(folder_list: list):
    """Combine models results to one excel"""
    all_files_results = dict()
    all_models_results = pd.DataFrame()
    skip = False
    folds_list = [f'fold_{i}' for i in range(6)]
    if not skip:
        for folder in folder_list:
            folder_path = os.path.join(base_directory, 'logs', folder)
            inner_folder_list = listdir(folder_path)
            if list(set(folds_list) & set(inner_folder_list)):
                for inner_folder in inner_folder_list:
                    if inner_folder == '.DS_Store':
                        continue
                    inner_folder_path =\
                        os.path.join(base_directory, 'logs', folder, inner_folder, 'excel_best_models_results')
                    inner_files_list = listdir(inner_folder_path)
                    all_results_file = 'Results_test_data_best_models.xlsx'
                    if all_results_file in inner_files_list:
                        file_path = os.path.join(inner_folder_path, all_results_file)
                        print(f'load file {file_path}')
                        sheet_name = 'All models results'
                        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=[0], index_col=0)
                        all_files_results[f'{folder}_{inner_folder}_{all_results_file}_all_models'] = df
                    else:
                        for file in inner_files_list:
                            if file == '.DS_Store':
                                continue
                            file_path = os.path.join(inner_folder_path, file)
                            print(f'load file {file_path}')
                            sheet_name = 'Model results'
                            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=[0], index_col=0)
                            all_files_results[f'{folder}_{file}_all_models'] = df

            else:
                folder_path = os.path.join(base_directory, 'logs', folder, 'excel_best_models_results')
                inner_files_list = listdir(folder_path)
                for file in inner_files_list:
                    if file == '.DS_Store':
                        continue
                    file_path = os.path.join(folder_path, file)
                    print(f'load file {file_path}')
                    if 'Results_test_data_best_models.xlsx' in file_path:
                        sheet_name = 'All models results'
                    else:
                        sheet_name = 'Model results'
                    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=[0], index_col=0)
                    all_files_results[f'{folder}_{file}_all_models'] = df

        joblib.dump(all_files_results, os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.pkl'))
    else:
        all_files_results = joblib.load(os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.pkl'))

    for key, value in all_files_results.items():
        value.index = [key]*value.shape[0]
        all_models_results = pd.concat([all_models_results, value], sort='False')

    # all_files_results = pd.DataFrame.from_dict(all_files_results, orient='index')
    # all_files_results = all_files_results.drop_duplicates()
    new_results_path = os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.csv')
    all_models_results.to_csv(new_results_path)
    return f'all_server_results_{folder_list}.csv'


def combine_models_all_results(folder_list: list, hyper_parameters: bool=True):
    """Combine models results to one excel"""
    all_files_results = dict()
    all_models_results = pd.DataFrame()
    skip = False
    if not skip:
        for folder in folder_list:
            folder_path = os.path.join(base_directory, 'logs', folder)
            if hyper_parameters:
                inner_folders_list = listdir(folder_path)
            else:
                inner_folders_list = ['']
            for inner_folder in inner_folders_list:
                if inner_folder == '.DS_Store':
                    continue
                files_path = join(folder_path, inner_folder, 'excel_models_results')
                if not os.path.isdir(files_path):
                    files_path = join(folder_path, inner_folder, 'excel_best_models_results')
                    if not os.path.isdir(files_path):
                        print(f'excel_models_results and excel_best_models_results are not folders in {inner_folder}')
                print(f'load file {files_path}')
                if 'excel_best_models_results' in files_path:
                    file_name = 'Results_test_data_best_models.xlsx'
                else:
                    file_name = f'Results_{inner_folder}_all_models.xlsx'
                df = pd.read_excel(os.path.join(files_path, file_name), sheet_name='All models results',
                                   skiprows=[0], index_col=0)
                if 'fold' not in df.columns:
                    print('create fold column')
                    df = df.assign(fold=inner_folder)
                all_files_results[f'{folder}_{inner_folder}_all_models'] = df

        joblib.dump(all_files_results, os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.pkl'))
    else:
        all_files_results = joblib.load(os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.pkl'))

    for key, value in all_files_results.items():
        value.index = [key]*value.shape[0]
        all_models_results = pd.concat([all_models_results, value], sort='False')

    # all_files_results = pd.DataFrame.from_dict(all_files_results, orient='index')
    # all_files_results = all_files_results.drop_duplicates()
    new_results_path = os.path.join(base_directory, 'logs', f'all_server_results_{folder_list}.csv')
    all_models_results.to_csv(new_results_path)
    return f'all_server_results_{folder_list}.csv'


def best_model_results_per_raisha_per_round(file_name: str, best_results_all_raisha_all_rounds: pd.DataFrame,
                                            best_model_to_plot_file_name: str, sheet_name: str= 'Sheet1',
                                            all_results: pd.DataFrame=None):
    if all_results is None:
        if 'csv' in file_name:
            all_results = pd.read_csv(file_name, index_col=0)
        else:
            all_results = pd.read_excel(file_name, sheet_name=sheet_name, index_col=0)

    if 'folder' not in all_results.columns:
        all_results['folder'] = all_results.index

    columns_to_merge_on = ['model_num', 'model_type', 'model_name', 'hyper_parameters_str']
    second_columns_to_merge_on = ['Round', 'Raisha', 'model_num', 'model_type', 'model_name']
    measures_columns = ['Per_round_Accuracy', 'RMSE', 'Bin_4_bins_Fbeta_score_micro', 'Bin_4_bins_Fbeta_score_macro']
    all_results['row_number'] = np.arange(len(all_results))
    all_results = all_results[columns_to_merge_on + measures_columns + ['Round', 'Raisha', 'fold']]

    columns_to_use = list()
    base_columns_name = [f'Best Model number for fold ', f'Best Model type for fold ', f'Best Model name for fold ',
                         f'Best Model Hyper-parameters for fold ']
    for column in base_columns_name:
        columns_to_use.extend([f'{column}{fold}' for fold in range(6)])

    best_results_all_raisha_all_rounds = best_results_all_raisha_all_rounds[columns_to_use]
    best_results_all_raisha_all_rounds.index.name = 'modelID'

    best_models_all_results = pd.DataFrame()
    if 'fold_0' in all_results.fold.unique():
        folds = [f'fold_{i}' for i in range(6)]
    elif 0 in all_results.fold.unique():
        folds = list(range(6))
    elif '0' in all_results.fold.unique():
        folds = [str(i) for i in range(6)]
    else:
        print('check fold column in all_results')
        return
    for fold in folds:
        data_to_merge =\
            best_results_all_raisha_all_rounds[[f'{column}{str(fold)[-1]}' for column in base_columns_name]].copy(deep=True)
        data_to_merge.columns = columns_to_merge_on
        fold_all_results = all_results.loc[all_results.fold == fold].copy(deep=True)
        fold_all_results = fold_all_results.drop('fold', axis=1)
        fold_merge = fold_all_results.merge(data_to_merge, on=columns_to_merge_on)
        fold_merge = fold_merge.drop('hyper_parameters_str', axis=1)
        if best_models_all_results.empty:
            best_models_all_results = fold_merge
        else:
            best_models_all_results = best_models_all_results.merge(fold_merge, on=second_columns_to_merge_on)
        for column in measures_columns:
            best_models_all_results.rename(columns={column: f'{column}_fold_{fold}'}, inplace=True)

    for column in measures_columns:
        best_models_all_results[f'average_{column}'] =\
            best_models_all_results[[f'{column}_fold_{fold}' for fold in folds]].mean(axis=1)

    average_columns = [f'average_{column}' for column in measures_columns]
    best_models_all_results_average = best_models_all_results[average_columns + second_columns_to_merge_on]
    best_models_all_results_average['model_id'] = best_models_all_results_average.model_num.map(str) + '_' +\
                                                  best_models_all_results_average.model_type + '_' + \
                                                  best_models_all_results_average.model_name

    # average per raisha --> take all rounds:
    avg_per_raisha_data =\
        best_models_all_results_average.loc[best_models_all_results_average.Round == 'All_rounds'].copy(deep=True)
    avg_per_raisha_data = avg_per_raisha_data.drop('Round', axis=1)
    avg_per_raisha_data = avg_per_raisha_data.loc[avg_per_raisha_data.Raisha != 'All_raishas']
    avg_per_raisha = avg_per_raisha_data.groupby(by=['Raisha', 'model_id']).mean()
    avg_per_raisha.reset_index(inplace=True)

    unique_model_id = avg_per_raisha.model_id.unique()
    final_dict = defaultdict(defaultdict)
    for column in measures_columns:
        final_dict[f'{column}'] = defaultdict(list)

    for model_id in unique_model_id:
        model_data = avg_per_raisha.loc[avg_per_raisha.model_id == model_id]
        for column in measures_columns:
            final_dict[f'{column}'][model_id] = model_data[f'average_{column}'].values.round(2).tolist()

    # average per round --> remove all raishas
    avg_per_round_data =\
        best_models_all_results_average.loc[best_models_all_results_average.Raisha == 'All_raishas'].copy(deep=True)
    avg_per_round_data = avg_per_round_data.drop('Raisha', axis=1)
    avg_per_round_data = avg_per_round_data.loc[avg_per_round_data.Round != 'All_rounds']
    avg_per_round = avg_per_round_data.groupby(by=['Round', 'model_id']).mean()
    avg_per_round.reset_index(inplace=True)

    unique_model_id = avg_per_round.model_id.unique()
    final_dict['per_trial_accuracy_per_round'] = defaultdict(list)

    for model_id in unique_model_id:
        model_data = avg_per_round.loc[avg_per_round.model_id == model_id]
        final_dict['per_trial_accuracy_per_round'][model_id] =\
            model_data[f'average_Per_round_Accuracy'].values.round(2).tolist()

    table_writer_to_plot = pd.ExcelWriter(os.path.join(base_directory, 'logs', 'analyze_results',
                                                       best_model_to_plot_file_name), engine='xlsxwriter')
    best_models_all_results_average['model_type'] = best_models_all_results_average.model_type + '_' + \
                                                         best_models_all_results_average.model_name
    to_merge = best_models_all_results_average[['model_type', 'model_id']]
    to_merge = to_merge.drop_duplicates()
    for key, value in final_dict.items():
        df = pd.DataFrame()
        df['model_id'] = value.keys()
        df = df.merge(to_merge, how='left')
        df[key] = value.values()
        if len(key) > 31:
            key = key[6:]
        write_to_excel(table_writer_to_plot, key, headers=[key], data=df)
    table_writer_to_plot.save()

    return final_dict


def select_best_model_per_type(file_name: str, rounds: str, raishas: str, measure_to_select_best: str,
                               table_writer: pd.ExcelWriter, measure_to_select_best_name: str,
                               sheet_name: str= 'Sheet1', all_measures: list=None, all_results: pd.DataFrame=None,
                               all_not_include_model_num: list=None, per_model_num: bool=False,
                               hyper_parameters: bool=True) -> \
        tuple([defaultdict, pd.Series, pd.Series, pd.Series, pd.Series]):
    if all_results is None:
        if 'csv' in file_name:
            all_results = pd.read_csv(file_name, index_col=0)
        else:
            all_results = pd.read_excel(file_name, sheet_name=sheet_name, index_col=0)

    models_to_compare = pd.read_excel(os.path.join(base_directory, 'models_to_hyper_parameters.xlsx'),
                                      sheet_name='table_to_load', skiprows=[0])
    if 'folder' not in all_results.columns:
        all_results['folder'] = all_results.index

    all_results['row_number'] = np.arange(len(all_results))
    all_rounds_all_raisha = all_results.loc[(all_results.Raisha == raishas) & (all_results.Round == rounds)]
    all_rounds_all_raisha.drop_duplicates(['folder', 'model_name', 'model_num', 'model_type', 'fold'], inplace=True)

    if all_rounds_all_raisha.empty:
        return None, None, None, None, None
    # all_rounds_all_raisha.model_num = all_rounds_all_raisha.model_num.astype(str)
    # new_column = all_rounds_all_raisha.model_num.str.split('_', expand=True)
    # all_rounds_all_raisha['model_num'] = new_column[0]
    # all_rounds_all_raisha['model_dropout'] = new_column[1]
    # all_rounds_all_raisha['text_features'] = \
    #     np.where(all_rounds_all_raisha.model_name.str.contains('bert'), 'BERT', 'Manual')
    # all_rounds_all_raisha =\
    #     all_rounds_all_raisha.merge(models_to_compare[['model_type', 'model_name', 'text_features']],
    #                                 on=['model_type', 'model_name'])

    all_rounds_all_raisha['modelID'] = all_rounds_all_raisha.model_type + '_' + all_rounds_all_raisha.model_name
    if 'fold_0' in all_rounds_all_raisha.fold.unique():
        prefix_fold = 'fold_'
    else:
        all_rounds_all_raisha.fold = all_rounds_all_raisha.fold.astype(str)
        prefix_fold = ''

    if all_not_include_model_num is not None:
        all_rounds_all_raisha = all_rounds_all_raisha.loc[
            ~all_rounds_all_raisha.model_num.isin(all_not_include_model_num)]
    all_rounds_all_raisha = all_rounds_all_raisha.fillna(0)
    all_model_names = all_rounds_all_raisha.modelID.unique()
    if per_model_num:
        all_model_names = all_rounds_all_raisha.model_num.unique()
    all_best_results = defaultdict(dict)
    for fold in range(6):
        best_results = defaultdict(dict)
        for model_name in all_model_names:
            if per_model_num:
                data = pd.DataFrame(all_rounds_all_raisha.loc[
                                        (all_rounds_all_raisha.model_num == model_name) &
                                        (all_rounds_all_raisha.fold == f'{prefix_fold}{fold}')])
            else:
                data = pd.DataFrame(all_rounds_all_raisha.loc[
                                        (all_rounds_all_raisha.modelID == model_name) &
                                        (all_rounds_all_raisha.fold == f'{prefix_fold}{fold}')])
            if data.empty:
                # print(f'data empty for model_type {model_type} and fold {fold}')
                continue
            # best models need to have only 1 row per model, fold, text_features
            if not hyper_parameters and data.shape[0] > 1:
                print('best models need to have only 1 row per model, fold, text_features')
                return
            use_seq = False
            if measure_to_select_best == 'RMSE':
                argmin_index = data[measure_to_select_best].idxmin()
                if f'{measure_to_select_best}_seq' in data.columns:
                    argmin_index_seq = data[f'{measure_to_select_best}_seq'].idxmin()
                    if data.loc[argmin_index][measure_to_select_best] >\
                            data.loc[argmin_index_seq][measure_to_select_best]:
                        argmin_index = argmin_index_seq
                        use_seq = True
            else:
                argmin_index = data[measure_to_select_best].idxmax()
                if f'{measure_to_select_best}_seq' in data.columns:
                    argmin_index_seq = data[f'{measure_to_select_best}_seq'].idxmax()
                    if data.loc[argmin_index][measure_to_select_best] <\
                            data.loc[argmin_index_seq][measure_to_select_best]:
                        argmin_index = argmin_index_seq
                        use_seq = True
            data = data.loc[argmin_index]
            best_results[f'{model_name}'] =\
                {f'Best {measure_to_select_best} for fold {fold}': data[measure_to_select_best],
                 f'Best Model number for fold {fold}': data.model_num,
                 f'Best Model type for fold {fold}': data.model_type,
                 f'Best Model file name for fold {fold}': data.data_file_name,
                 f'Best Model folder for fold {fold}': data.folder,
                 f'Best Model name for fold {fold}': data.model_name,
                 f'Best Model Hyper-parameters for fold {fold}': data.hyper_parameters_str,
                 f'Seq is better for {fold}': use_seq,
                 f'row number fold {fold}': data.row_number}
            if all_measures is not None:  # add all other measures of the best model
                for measure in all_measures:
                    if measure in data:
                        best_results[f'{model_name}'][f'Best {measure} for fold {fold}'] =\
                            data[measure]

        all_best_results = update_default_dict(all_best_results, best_results)

    all_best_results = pd.DataFrame.from_dict(all_best_results).T
    if per_model_num:
        # models_to_compare['text_features'] =\
        #     np.where(models_to_compare.model_name.str.contains('bert'), 'BERT', 'manual')
        model_types = models_to_compare[['model_num', 'model_type', 'text_features']]
        index_to_insert = model_types.shape[1]
        model_types.model_num = model_types.model_num.astype(str)
        all_best_results = model_types.merge(all_best_results, right_index=True, left_on='model_num')

    else:
        all_best_results.index.name = 'model_type'
        index_to_insert = 0

    measure_to_select_all_folds = [f'Best {measure_to_select_best} for fold {fold}' for fold in range(6)]
    all_best_results.insert(index_to_insert, f'Average_{measure_to_select_best}',
                            all_best_results[measure_to_select_all_folds].mean(axis=1))
    index_to_insert += 1
    all_best_results.insert(index_to_insert, f'STD_{measure_to_select_best}',
                            all_best_results[measure_to_select_all_folds].std(axis=1))
    index_to_insert += 1

    if all_measures is not None:  # add all other measures of the best model
        for measure in all_measures:
            if measure in data:
                measure_all_folds = [f'Best {measure} for fold {fold}' for fold in range(6)]
                all_best_results.insert(index_to_insert, f'Average_{measure}',
                                        all_best_results[measure_all_folds].mean(axis=1))
                index_to_insert += 1
                all_best_results.insert(index_to_insert, f'STD_{measure}', all_best_results[measure_all_folds].std(axis=1))
                index_to_insert += 1

    if measure_to_select_best == 'RMSE':
        sort_ascending = True
    else:
        sort_ascending = False
    all_best_results = all_best_results.sort_values(by=f'Average_{measure_to_select_best}', ascending=sort_ascending)
    if per_model_num:
        all_best_results.reset_index(inplace=True, drop=True)
    if rounds == 'All_rounds':
        write_to_excel(table_writer, f'{measure_to_select_best_name}_{raishas}',
                       headers=[f'best_models_based_on_{measure_to_select_best_name}_for_{raishas}'],
                       data=all_best_results)
    else:
        write_to_excel(table_writer, f'{measure_to_select_best_name}_{rounds}',
                       headers=[f'best_models_based_on_{measure_to_select_best_name}_for_{rounds}'],
                       data=all_best_results)
    return all_best_results, all_best_results[f'Average_{measure_to_select_best}'],\
           all_best_results['Average_Bin_4_bins_Fbeta_score_macro'],\
           all_best_results['Average_Bin_4_bins_Fbeta_score_micro'], all_best_results['Average_Per_round_Accuracy']


def concat_new_results(new_results_name: str, old_results_name: str, new_results_file_name_to_save: str):
    if 'xslx' in old_results_name:
        all_results = pd.read_excel(os.path.join(base_directory, 'logs', f'{old_results_name}'), sheet_name='Sheet1',
                                    index_col=[0])
    elif 'csv' in old_results_name:
        all_results = pd.read_csv(os.path.join(base_directory, 'logs', f'{old_results_name}'), index_col=[0])
    else:
        print('old_results_name is not csv or excel file')
        return

    # for model_num in [186, 187, 188, 63, 64, 65]:
    #     all_results = all_results.loc[~all_results.model_num.str.contains(str(model_num))]
    # all_results = all_results.loc[~all_results.model_num.isin([186, 187, 188, 63, 64, 65])]

    if new_results_name == old_results_name:  # no need to concat
        final = all_results
    else:
        new_results = pd.read_csv(os.path.join(base_directory, 'logs', new_results_name), index_col=[0])
        # new_results.rename(columns={'Unnamed: 0': 'folder'}, inplace=True)

        if 'hyper_parameters_dict' in new_results.columns:
            new_results.rename(columns={'hyper_parameters_dict': 'hyper_parameters_str'}, inplace=True)
        # all_results = all_results[new_results.columns]

        if new_results_name == "all_server_results_['compare_prediction_models_27_08_2020_11_30'].csv":
            new_results = new_results.loc[~new_results.model_num.str.contains('112')]
            new_results = new_results.loc[~new_results.model_num.str.contains('113')]
            new_results = new_results.loc[~new_results.model_num.str.contains('114')]

        final = pd.concat([all_results, new_results], axis=0)
    # all_columns = final.columns.tolist()
    # all_columns.remove('folder')
    final = final.drop_duplicates()
    final.index.name = 'folder'
    # joblib.dump(final, os.path.join(base_directory, 'logs', f'{new_results_file_name_to_save}.pkl'))
    print(f'Save new data to: {new_results_file_name_to_save} concat of {old_results_name} and {new_results_name}')
    final.to_csv(os.path.join(base_directory, 'logs', f'{new_results_file_name_to_save}'))
    # final.to_excel(os.path.join(base_directory, 'logs', f'{new_results_file_name_to_save}.xlsx'))


def correlation_analysis(data_path: str):
    """Analyze the correlation between rounds"""
    data = pd.read_csv(data_path)
    data = data.loc[data.raisha == 0]
    rounds_list = list(range(1, 11))
    df_list = list()
    for my_round in rounds_list:
        df = data.loc[data.round_number == my_round].labels
        df.reset_index(drop=True, inplace=True)
        df_list.append(df)

    labels = pd.concat(df_list, axis=1, ignore_index=True)
    labels.columns = list(range(1, 11))
    corr = labels.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # plt.figure(figsize=(5, 5))
    ax = sns.heatmap(corr, cmap='coolwarm', annot=True, mask=mask, fmt='.2f', annot_kws={"size": 10})
    # ax = sns.heatmap(corr, annot=True, fmt='.2f', annot_kws={"size": 8})
    ax.set(xlabel='Trial Number', ylabel='Trial Number')
    ax.set_title('Heat Map for Correlation Between Decisions in Different Trials')
    corr.to_csv(os.path.join(base_directory, 'logs', 'analyze_results', 'rounds_correlation_analysis.csv'))
    plt.savefig(os.path.join(base_directory, 'logs', 'analyze_results', 'correlation_heat_map.png'))
    plt.show()

    return


def find_all_best_models_of_directory(directory: str, best_models_file_name: str):
    xls = pd.ExcelFile(os.path.join(base_directory, 'logs', 'analyze_results', best_models_file_name))
    number_best_fold = defaultdict(list)
    for sheet in xls.sheet_names:
        if 'All_raishas' in sheet:
            df = pd.read_excel(xls, sheet, skiprows=[0])
            for fold in range(6):
                data = df[f'Best Model folder for fold {fold}']
                index_data = data.str.findall(directory)
                for my_index in index_data.index:
                    if type(index_data[my_index]) == list and len(index_data[my_index]) > 0:
                        if 'SVM' in df.iloc[my_index][f'Best Model type for fold {fold}']:
                            number_best_fold[fold].append(
                                f"mv {directory}/fold_{fold}/{df.iloc[my_index][f'Best Model number for fold {fold}']}_"
                                f"{df.iloc[my_index][f'Best Model type for fold {fold}']}_"
                                f"{df.iloc[my_index][f'Best Model name for fold {fold}']}"
                                f"_fold_{fold}.pkl {directory}_best/fold_{fold}")
                        else:
                            number_best_fold[fold].append(
                                f"mv {directory}/fold_{fold}/{df.iloc[my_index][f'Best Model number for fold {fold}']}_"
                                f"{df.iloc[my_index][f'Best Model type for fold {fold}']}_"
                                f"{df.iloc[my_index][f'Best Model name for fold {fold}']}"
                                f"_100_epochs_fold_num_{fold}/ {directory}_best/fold_{fold}")

    # print(f'\nBest models in {directory}')
    print(f'mkdir {directory}_best')
    for fold in range(6):
        best_models_list = sorted(set(number_best_fold[fold]))
        # print(f'\nNumber of best models for fold {fold} is {len(best_models_list)}')
        print(f'mkdir {directory}_best/fold_{fold}')
        for item in best_models_list:
            print(item)

    return


def main(new_results_file_name, old_results_name, best_model_file_name: str, best_model_to_plot_file_name: str,
         results_to_use: pd.DataFrame=None, hyper_parameters=True,):
    # for dir in ['compare_prediction_models_07_09_2020_11_12']:
    #     find_all_best_models_of_directory(dir, best_model_file_name)
    # return
    # total_payoff_true_label_per_fold = pd.read_excel(os.path.join(base_directory, 'logs', 'total_label_per_fold.xlsx'))
    # for i, folder in enumerate([['compare_prediction_models_16_11_2020_21_33']]):
    #     results_file_name = combine_models_results(folder, total_payoff_true_label=total_payoff_true_label_per_fold,
    #                                                baseline=False)
    #     if i == 0:  # first folder
    #         old_results_name = old_results_name
    #     else:
    #         old_results_name = new_results_file_name
    #     concat_new_results(new_results_name=results_file_name,
    #                        old_results_name=old_results_name,
    #                        new_results_file_name_to_save=new_results_file_name)
    # for i, folder in enumerate([['predict_best_models_20_10_2020_14_13'],
    #                             ['predict_best_models_16_11_2020_21_33'],
    #                             ['predict_best_models_22_09_2020_10_42'],
    #                             ['predict_best_models_06_11_2020_18_50']]):
    #     if i == 0:  # first folder
    #         old_results_name = old_results_name
    #     else:
    #         old_results_name = new_results_file_name
    #
    #     if isfile(os.path.join(base_directory, 'logs', f"all_server_results_['{folder}'].csv")):
    #         results_file_name = f"all_server_results_['{folder}'].csv"
    #     else:
    #         results_file_name = combine_models_predictions(folder)
    #     concat_new_results(new_results_name=results_file_name,
    #                        old_results_name=old_results_name,
    #                        new_results_file_name_to_save=new_results_file_name)
    # results_file_name = combine_models_predictions(['predict_best_models_18_11_2020_12_26',
    #                                                 'predict_best_models_16_11_2020_21_33'])
    # concat_new_results(new_results_name=results_file_name,
    #                    old_results_name=old_results_name,
    #                    new_results_file_name_to_save=new_results_file_name)
    # concat_new_results(new_results_name="all_server_results_['predict_best_models_10_09_2020_11_36', "
    #                                     "'predict_best_models_10_09_2020_15_20', "
    #                                     "'predict_best_models_07_09_2020_19_36'].csv",
    #                    old_results_name=new_results_file_name,
    #                    new_results_file_name_to_save=new_results_file_name)
    # results_file_name = "all_server_results_['compare_prediction_models_13_08_2020_11_48', " \
    #                "'compare_prediction_models_13_08_2020_15_33'].csv"

    # return
    # for results_file_name in ["all_server_results_['compare_prediction_models_27_08_2020_11_30'].csv",
    #                           "all_server_results_['compare_prediction_models_30_08_2020_11_25'].csv",
    #                           "all_server_results_['compare_prediction_models_30_08_2020_11_27'].csv",
    #                           "all_server_results_['compare_prediction_models_31_08_2020_10_23'].csv",
    #                           "all_server_results_['compare_prediction_models_31_08_2020_10_18'].csv"]:
    #     concat_new_results(results_file_name, old_results_name=old_results_name,
    #                        new_results_file_name_to_save=new_results_file_name)

    # for folder_results in ["all_server_results_['compare_prediction_models_16_11_2020_21_33'].csv"]:
    #     concat_new_results(folder_results,
    #                        old_results_name=new_results_file_name,
    #                        new_results_file_name_to_save=new_results_file_name)

    # old_file = pd.read_csv(os.path.join(base_directory, 'logs', old_results_name), index_col=[0])
    # old_file.to_csv(os.path.join(base_directory, 'logs',
    #                              'all_results_05_11_remove_50_participants_folders_25_08_27_08.csv'))
    # old_file = old_file.drop(['folder.1'], axis=1)
    # for folder in ['compare_prediction_models_27_08_2020_11_30', 'compare_prediction_models_27_08_2020_11_09',
    #                'compare_prediction_models_27_08_2020_11_29', 'compare_prediction_models_26_08_2020_10_17',
    #                'compare_prediction_models_26_08_2020_10_08', 'compare_prediction_models_26_08_2020_10_02',
    #                'compare_prediction_models_25_08_2020_21_34']:
    #     old_file = old_file[~old_file.index.str.contains(folder)]
    # old_file = old_file.loc[~old_file.model_name.isin([
    #     'LSTM_avg_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_bert_text_features',
    #     'LSTM_avg_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_manual_text_features',
    #     'LSTM_avg_last_hidden_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_bert_text_features',
    #     'LSTM_avg_last_hidden_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_manual_text_features',
    #     'LSTM_avg_turn_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_bert_text_features',
    #     'LSTM_avg_turn_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_manual_text_features',
    #     'LSTM_avg_turn_linear_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_bert_text_features',
    #     'LSTM_avg_turn_linear_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_manual_text_features',
    #     'LSTM_avg_turn_last_hidden_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_bert_text_features',
    #     'LSTM_avg_turn_last_hidden_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_manual_text_features',
    #     'LSTM_turn_linear_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_bert_text_features',
    #     'LSTM_turn_linear_avg_raisha_global_0.8_text_0.9_current_round_text_average_saifa_text_manual_text_features',
    #     'SVM_total_bert_text_features_average_raisha_text_0.9_global_0.8_all_saifa_average_text_first_round_saifa_text',
    #     'SVM_total_manual_text_features_average_raisha_text_0.9_global_0.8_all_saifa_average_text_first_round_saifa_text',])]
    # old_file.to_csv(os.path.join(base_directory, 'logs', new_results_file_name))
    # concat_new_results("all_server_results_['compare_prediction_models_08_09_2020_22_59'].csv",
    #                    old_results_name='all_results_20_09_new_models_before_concat_to_new_lstm_old_version.csv',
    #                    new_results_file_name_to_save=new_results_file_name)

    # return

    # correlation_analysis(
    #     os.path.join(base_directory, 'data', 'verbal', 'cv_framework',
    #                  'all_data_single_round_label_crf_raisha_non_nn_turn_model_prev_round_label_all_history_features_'
    #                  'all_history_text_manual_binary_features_predict_first_round_verbal_data.csv'))
    #
    # return

    main_file_name = (os.path.join(base_directory, 'logs', new_results_file_name))
    if 'csv' in main_file_name:
        all_results = pd.read_csv(main_file_name, index_col=0)
    else:
        all_results = pd.read_excel(main_file_name, sheet_name='Sheet1', index_col=0)

    all_measures = ['Bin_Accuracy', 'Bin_Fbeta_score_1/3 < total future payoff < 2/3',
                    'Bin_Fbeta_score_total future payoff < 1/3',
                    'Bin_Fbeta_score_total future payoff > 2/3', 'Bin_precision_1/3 < total future payoff < 2/3',
                    'Bin_precision_total future payoff < 1/3', 'Bin_precision_total future payoff > 2/3',
                    'Bin_recall_1/3 < total future payoff < 2/3', 'Bin_recall_total future payoff < 1/3',
                    'Bin_recall_total future payoff > 2/3', 'MAE', 'MSE', 'Per_round_Accuracy',
                    'Per_round_Fbeta_score_DM chose hotel', 'Per_round_Fbeta_score_DM chose stay home',
                    'Per_round_precision_DM chose hotel', 'Per_round_precision_DM chose stay home',
                    'Per_round_recall_DM chose hotel', 'Per_round_recall_DM chose stay home', 'RMSE']
    raishas = ['All_raishas'] + [f'raisha_{raisha}' for raisha in range(10)]
    if hyper_parameters:
        best_model_file_name = best_model_file_name
        best_model_to_plot_file_name = best_model_to_plot_file_name
    else:
        best_model_file_name = best_model_file_name
        best_model_to_plot_file_name = best_model_to_plot_file_name
    table_writer = pd.ExcelWriter(os.path.join(base_directory, 'logs', 'analyze_results', best_model_file_name),
                                  engine='xlsxwriter')
    # not_include_model_num = [16, 24, 32, 43, 47, 58, 69, 80, 94, 105, 119, 130, 145, 156, 170, 181, 187, 391, 402, 413,
    #                          424, 435, 446, 730, 744]
    # all_not_include_model_num = copy.deepcopy(not_include_model_num)
    # for num in not_include_model_num:
    #     all_not_include_model_num.append(str(num))
    #     for dropout in [None, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
    #         all_not_include_model_num.append(f'{str(num)}_{dropout}')

    best_results_to_plot_rmse = list()
    best_results_to_plot_micro = list()
    best_results_to_plot_macro = list()
    best_results_to_plot_per_trial_accuracy = list()

    # measures_list = [['Bin_Fbeta_score_total future payoff < 1/3', 'Bin_Fbeta_1_3'],
    #                  ['Bin_Fbeta_score_total future payoff > 2/3', 'Bin_Fbeta_2_3'],
    #                  ['Bin_Fbeta_score_1/3 < total future payoff < 2/3', 'Bin_Fbeta_1_3_2_3'],
    #                  ['RMSE', 'RMSE'], ['Bin_Accuracy', 'Bin_Accuracy']]
    all_measures = ['Bin_3_bins_Accuracy', 'Bin_3_bins_Fbeta_score_macro', 'Bin_3_bins_Fbeta_score_micro',
                    'Bin_3_bins_precision_macro', 'Bin_3_bins_precision_micro', 'Bin_3_bins_recall_macro',
                    'Bin_3_bins_recall_micro', 'Bin_4_bins_Accuracy', 'Bin_4_bins_Fbeta_score_macro',
                    'Bin_4_bins_Fbeta_score_micro', 'Bin_4_bins_precision_macro', 'Bin_4_bins_precision_micro',
                    'Bin_4_bins_recall_macro', 'Bin_4_bins_recall_micro',
                    'Bin_Fbeta_score_1/2 < total future payoff < 3/4',
                    'Bin_Fbeta_score_1/3 < total future payoff < 2/3',
                    'Bin_Fbeta_score_1/4 < total future payoff < 1/2', 'Bin_Fbeta_score_total future payoff < 1/3',
                    'Bin_Fbeta_score_total future payoff < 1/4', 'Bin_Fbeta_score_total future payoff > 2/3',
                    'Bin_Fbeta_score_total future payoff > 3/4', 'Bin_precision_1/2 < total future payoff < 3/4',
                    'Bin_precision_1/3 < total future payoff < 2/3',
                    'Bin_precision_1/4 < total future payoff < 1/2', 'Bin_precision_total future payoff < 1/3',
                    'Bin_precision_total future payoff < 1/4', 'Bin_precision_total future payoff > 2/3',
                    'Bin_precision_total future payoff > 3/4', 'Bin_recall_1/2 < total future payoff < 3/4',
                    'Bin_recall_1/3 < total future payoff < 2/3', 'Bin_recall_1/4 < total future payoff < 1/2',
                    'Bin_recall_total future payoff < 1/3', 'Bin_recall_total future payoff < 1/4',
                    'Bin_recall_total future payoff > 2/3', 'Bin_recall_total future payoff > 3/4', 'MAE', 'MSE',
                    'Per_round_Accuracy', 'Per_round_Fbeta_score_DM chose hotel',
                    'Per_round_Fbeta_score_DM chose stay home', 'Per_round_precision_DM chose hotel',
                    'Per_round_precision_DM chose stay home', 'Per_round_recall_DM chose hotel',
                    'Per_round_recall_DM chose stay home', 'RMSE']
    measures_list = [['RMSE', 'RMSE']]
    for measure_to_analyze in measures_list:
        for raisha in raishas:
            print(f'Analyze results based on {measure_to_analyze} for {raisha}')
            other_measures = copy.deepcopy(all_measures)
            other_measures.remove(measure_to_analyze[0])
            all_best_results, best_rmse, best_macro, best_micro, best_trial_accuracy =\
                select_best_model_per_type(main_file_name, raishas=raisha, rounds='All_rounds',
                                           table_writer=table_writer, measure_to_select_best=measure_to_analyze[0],
                                           measure_to_select_best_name=measure_to_analyze[1],
                                           all_measures=other_measures, all_results=all_results,
                                           # all_not_include_model_num=all_not_include_model_num,
                                           per_model_num=False, hyper_parameters=hyper_parameters)
            if raisha == 'All_raishas':
                per_raisha_per_round_dict = best_model_results_per_raisha_per_round(
                    main_file_name, best_results_all_raisha_all_rounds=all_best_results, all_results=all_results,
                    best_model_to_plot_file_name=best_model_to_plot_file_name)
            # if measure_to_analyze[0] in ['RMSE', 'Bin_Accuracy'] and all_best_results is not None:
            #     best_rmse.name = f'Average_RMSE_for_raisha_{raisha}'
            #     best_results_to_plot_rmse.append(best_rmse.round(2))
            #     best_macro.name = f'Average_Bin_4_bins_Fbeta_score_macro_for_raisha_{raisha}'
            #     best_results_to_plot_macro.append(best_macro.round(2))
            #     best_micro.name = f'Average_Bin_4_bins_Fbeta_score_micro_for_raisha_{raisha}'
            #     best_results_to_plot_micro.append(best_micro.round(2))
            #     best_trial_accuracy.name = f'Average_Per_round_Accuracy_for_raisha_{raisha}'
            #     best_results_to_plot_per_trial_accuracy.append(best_trial_accuracy.round(2))

    table_writer.save()

    # best_results_to_plot_rmse = pd.concat(best_results_to_plot_rmse, sort=False, axis=1)
    # best_results_to_plot_macro = pd.concat(best_results_to_plot_macro, sort=False, axis=1)
    # best_results_to_plot_micro = pd.concat(best_results_to_plot_micro, sort=False, axis=1)
    # best_results_to_plot_per_trial_accuracy = pd.concat(best_results_to_plot_per_trial_accuracy, sort=False, axis=1)
    # table_writer_to_plot = pd.ExcelWriter(os.path.join(base_directory, 'logs', 'analyze_results',
    #                                                    best_model_to_plot_file_name), engine='xlsxwriter')
    # write_to_excel(table_writer_to_plot, 'best_RMSE_results', headers=['best_RMSE_results'],
    #                data=best_results_to_plot_rmse)
    # write_to_excel(table_writer_to_plot, 'best_4_bins_Fbeta_score_macro',
    #                headers=['best_4_bins_Fbeta_score_macro'], data=best_results_to_plot_macro)
    # write_to_excel(table_writer_to_plot, 'best_4_bins_Fbeta_score_micro',
    #                headers=['best_4_bins_Fbeta_score_micro'], data=best_results_to_plot_micro)
    # write_to_excel(table_writer_to_plot, 'best_Per_round_Accuracy', headers=['best_Per_round_Accuracy'],
    #                data=best_results_to_plot_per_trial_accuracy)
    # table_writer_to_plot.save()

    # for dir in ['compare_prediction_models_16_11_2020_21_11', 'compare_prediction_models_16_11_2020_21_33']:
    #     find_all_best_models_of_directory(dir, best_model_file_name)
    return


if __name__ == '__main__':
    main(
        # compare folders
        # new_results_file_name='all_results_19_11.csv',
        # old_results_name='all_results_18_11.csv', hyper_parameters=True,
        # best_model_file_name='Best models analysis best models tuning new test data.xlsx',
        # best_model_to_plot_file_name='Best models analysis best models tuning new test data for plot.xlsx'
        # predict folder
        new_results_file_name='all_server_results_predict_best_models_16_11.csv',
        old_results_name='all_server_results_predict_best_models_16_11.csv', hyper_parameters=True,
        best_model_file_name='Best models prediction new test data.xlsx',
        best_model_to_plot_file_name='Best models prediction new test data for plot.xlsx'
    )
