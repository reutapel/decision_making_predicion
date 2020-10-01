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


def select_best_model_per_type(file_name: str, rounds: str, raishas: str, measure_to_select_best: str,
                               table_writer: pd.ExcelWriter, measure_to_select_best_name: str,
                               sheet_name: str= 'Sheet1', all_measures: list=None, all_results: pd.DataFrame=None,
                               all_not_include_model_num: list=None, per_model_num: bool=False,
                               hyper_parameters: bool=True):
    if all_results is None:
        if 'csv' in file_name:
            all_results = pd.read_csv(file_name, index_col=0)
        else:
            all_results = pd.read_excel(file_name, sheet_name=sheet_name, index_col=0)

    models_to_compare = pd.read_excel(os.path.join(base_directory, 'models_to_hyper_parameters.xlsx'),
                                      sheet_name='table_to_load', skiprows=[0])
    if 'folder' not in all_results.columns:
        all_results['folder'] = all_results.index

    all_rounds_all_raisha = all_results.loc[(all_results.Raisha == raishas) & (all_results.Round == rounds)]
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
                 f'Seq is better for {fold}': use_seq}
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
    write_to_excel(table_writer, f'{measure_to_select_best_name}_{raishas}',
                   headers=[f'best_models_based_on_{measure_to_select_best_name}_for_{raishas}'], data=all_best_results)
    return all_best_results, all_best_results[f'Average_{measure_to_select_best}']


def concat_new_results(new_results_name: str, old_results_name: str, new_results_file_name_to_save: str):
    if 'xslx' in old_results_name:
        all_results = pd.read_excel(os.path.join(base_directory, 'logs', f'{old_results_name}'), sheet_name='Sheet1',
                                    index_col=[0])
    elif 'csv' in old_results_name:
        all_results = pd.read_csv(os.path.join(base_directory, 'logs', f'{old_results_name}'), index_col=[0])
    else:
        print('old_results_name is not csv or excel file')
        return

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
    total_payoff_true_label_per_fold = pd.read_excel(os.path.join(base_directory, 'logs', 'total_label_per_fold.xlsx'))
    for i, folder in enumerate([['compare_prediction_models_24_09_2020_19_21'],
                                ['compare_prediction_models_21_09_2020_11_52',
                                 'compare_prediction_models_29_09_2020_10_10']]):
        results_file_name = combine_models_results(folder, total_payoff_true_label=total_payoff_true_label_per_fold,
                                                   baseline=False)
        if i == 0:  # first folder
            old_results_name = old_results_name
        else:
            old_results_name = new_results_file_name
        concat_new_results(new_results_name=results_file_name,
                           old_results_name=old_results_name,
                           new_results_file_name_to_save=new_results_file_name)
    # results_file_name = combine_models_all_results(['predict_best_models_10_09_2020_11_36',
    #                                                 'predict_best_models_10_09_2020_15_20',
    #                                                 'predict_best_models_07_09_2020_19_36',
    #                                                 'predict_best_models_22_09_2020_10_42',
    #                                                 'predict_best_models_22_09_2020_11_09'],
    #                                                hyper_parameters=hyper_parameters)
    # new_results_file_name = results_file_name
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

    # concat_new_results("all_server_results_['compare_prediction_models_23_06_2020_10_41_hyper', "
    #                    "'compare_prediction_models_23_06_2020_16_44_hyper'].csv",
    #                    old_results_name=new_results_file_name,
    #                    new_results_file_name_to_save=new_results_file_name)
    #
    # old_file = pd.read_csv(os.path.join(base_directory, 'logs', old_results_name))
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
    # old_file.index = old_file.folder
    # old_file.to_csv(os.path.join(base_directory, 'logs',
    #                              'all_results_20_09_new_models_before_concat_to_new_lstm_old_version.csv'))
    # concat_new_results("all_server_results_['compare_prediction_models_08_09_2020_22_59'].csv",
    #                    old_results_name='all_results_20_09_new_models_before_concat_to_new_lstm_old_version.csv',
    #                    new_results_file_name_to_save=new_results_file_name)
    #
    # return

    # correlation_analysis(
    #     os.path.join(base_directory, 'data', 'verbal', 'cv_framework',
    #                  'all_data_single_round_label_crf_raisha_non_nn_turn_model_prev_round_label_all_history_features_'
    #                  'all_history_text_manual_binary_features_predict_first_round_verbal_data.csv'))
    #
    # return

    main_file_name = (os.path.join(base_directory, 'logs', new_results_file_name))
    all_measures = ['Bin_Accuracy', 'Bin_Fbeta_score_1/3 < total future payoff < 2/3',
                    'Bin_Fbeta_score_total future payoff < 1/3',
                    'Bin_Fbeta_score_total future payoff > 2/3', 'Bin_precision_1/3 < total future payoff < 2/3',
                    'Bin_precision_total future payoff < 1/3', 'Bin_precision_total future payoff > 2/3',
                    'Bin_recall_1/3 < total future payoff < 2/3', 'Bin_recall_total future payoff < 1/3',
                    'Bin_recall_total future payoff > 2/3', 'MAE', 'MSE', 'Per_round_Accuracy',
                    'Per_round_Fbeta_score_DM chose hotel', 'Per_round_Fbeta_score_DM chose stay home',
                    'Per_round_precision_DM chose hotel', 'Per_round_precision_DM chose stay home',
                    'Per_round_recall_DM chose hotel', 'Per_round_recall_DM chose stay home', 'RMSE']
    raishas = [f'raisha_{raisha}' for raisha in range(10)]
    raishas.append('All_raishas')
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

    best_results_to_plot = list()
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
            all_best_results, measure_best_results =\
                select_best_model_per_type(main_file_name, raishas=raisha, rounds='All_rounds',
                                           table_writer=table_writer, measure_to_select_best=measure_to_analyze[0],
                                           measure_to_select_best_name=measure_to_analyze[1],
                                           all_measures=other_measures, all_results=results_to_use,
                                           # all_not_include_model_num=all_not_include_model_num,
                                           per_model_num=False, hyper_parameters=hyper_parameters)
            if measure_to_analyze[0] in ['RMSE', 'Bin_Accuracy']:
                measure_best_results.name = f'Average_{measure_to_analyze[1]}_for_{raisha}'
                best_results_to_plot.append(measure_best_results)

    table_writer.save()

    best_results_to_plot = pd.concat(best_results_to_plot, sort=False, axis=1)
    best_results_to_plot.to_csv(os.path.join(base_directory, 'logs', 'analyze_results', best_model_to_plot_file_name))

    for dir in ['compare_prediction_models_24_09_2020_19_21',
                'compare_prediction_models_21_09_2020_11_52',
                'compare_prediction_models_29_09_2020_10_10']:
        find_all_best_models_of_directory(dir, best_model_file_name)
    return


if __name__ == '__main__':
    main(new_results_file_name='all_results_30_09_new_models.csv',
         old_results_name='all_results_24_09_new_models.csv', hyper_parameters=True,
         best_model_file_name='Best models analysis best models tunning new test data.xlsx',
         best_model_to_plot_file_name='Best models analysis best models tunning new test data for plot.csv'
    )
