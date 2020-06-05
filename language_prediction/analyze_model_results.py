import pandas as pd
import os
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import joblib
from language_prediction.utils import *
import itertools

base_directory = os.path.abspath(os.curdir)


def combine_models_results(folder_list: list):
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


def combine_models_all_results(folder_list: list):
    """Combine models results to one excel"""
    all_files_results = dict()
    all_models_results = pd.DataFrame()
    skip = False
    if not skip:
        for folder in folder_list:
            folder_path = os.path.join(base_directory, 'logs', folder)
            for inner_folder in listdir(folder_path):
                if inner_folder == '.DS_Store':
                    continue
                files_path = join(folder_path, inner_folder)  # , 'excel_models_results')
                print(f'load file {files_path}')
                df = pd.read_excel(os.path.join(files_path,
                                                f'Results_{inner_folder}_all_models.xlsx'),
                                   sheet_name='All models results', skiprows=[0], index_col=0)
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
                               all_not_include_model_num: list=None):
    if all_results is None:
        if 'csv' in file_name:
            all_results = pd.read_csv(file_name)
        else:
            all_results = pd.read_excel(file_name, sheet_name=sheet_name)
    all_rounds_all_raisha = all_results.loc[(all_results.Raisha == raishas) & (all_results.Round == rounds)]
    if all_not_include_model_num is not None:
        all_rounds_all_raisha = all_rounds_all_raisha.loc[
            ~all_rounds_all_raisha.model_num.isin(all_not_include_model_num)]
    all_rounds_all_raisha = all_rounds_all_raisha.fillna(0)
    all_model_types = all_rounds_all_raisha.model_type.unique()
    all_best_results = defaultdict(dict)
    for fold in range(6):
        best_results = defaultdict(dict)
        for model_type in all_model_types:
            data = pd.DataFrame(all_rounds_all_raisha.loc[(all_rounds_all_raisha.model_type == model_type) &
                                                          (all_rounds_all_raisha.fold == f'fold_{fold}')])
            if data.empty:
                print(f'data empty for model_type {model_type} and fold {fold}')
                continue
            if measure_to_select_best == 'RMSE':
                argmin_index = data[measure_to_select_best].idxmin()
            else:
                argmin_index = data[measure_to_select_best].idxmax()
            data = data.loc[argmin_index]
            best_results[model_type] = {f'Best {measure_to_select_best} for fold {fold}': data[measure_to_select_best],
                                        f'Best Model number for fold {fold}': data.model_num,
                                        f'Best Model folder for fold {fold}': data.folder,
                                        f'Best Model name for fold {fold}': data.model_name}
            if all_measures is not None:  # add all other measures of the best model
                for measure in all_measures:
                    best_results[model_type][f'Best {measure} for fold {fold}'] = data[measure]

        all_best_results = update_default_dict(all_best_results, best_results)

    all_best_results = pd.DataFrame.from_dict(all_best_results).T
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
    write_to_excel(table_writer, f'{measure_to_select_best_name}_{raishas}',
                   headers=[f'best_models_based_on_{measure_to_select_best_name}_for_{raishas}'], data=all_best_results)
    return all_best_results


def concat_new_results(new_results_name: str, old_results_name: str, new_results_file_name_to_save: str):
    if 'xslx' in old_results_name:
        all_results = pd.read_excel(os.path.join(base_directory, 'logs', f'{old_results_name}'), sheet_name='Sheet1',
                                    index_col=[0])
    elif 'csv' in old_results_name:
        all_results = pd.read_csv(os.path.join(base_directory, 'logs', f'{old_results_name}'), index_col=[0])
    else:
        print('old_results_name is not csv or excel file')
        return
    # all_results = all_results.loc[~all_results.model_type.isin(['SVMTurn'])]
    new_results = pd.read_csv(os.path.join(base_directory, 'logs', new_results_name))
    # svm_results = svm_results.loc[svm_results.model_type.isin(['SVMTotal', 'SVMTurn', 'CRF'])]
    new_results.rename(columns={'Unnamed: 0': 'folder'}, inplace=True)

    if 'hyper_parameters_dict' in new_results.columns:
        new_results.rename(columns={'hyper_parameters_dict': 'hyper_parameters_str'}, inplace=True)
    all_results = all_results[new_results.columns]

    final = pd.concat([all_results, new_results], axis=0)
    all_columns = final.columns.tolist()
    all_columns.remove('folder')
    final = final.drop_duplicates()
    joblib.dump(final, os.path.join(base_directory, 'logs', f'{new_results_file_name_to_save}.pkl'))
    final.to_csv(os.path.join(base_directory, 'logs', f'{new_results_file_name_to_save}.csv'))
    final.to_excel(os.path.join(base_directory, 'logs', f'{new_results_file_name_to_save}.xlsx'))


def correlation_analysis(data_path: str):
    """Analyze the correlation between rounds"""
    data = pd.read_csv(data_path)
    data = data.loc[data.raisha == 0]
    rounds_list = list(range(1, 11))
    rounds_combination = itertools.combinations(rounds_list, 2)
    results_dict = defaultdict(list)
    for my_round in rounds_list:
        results_dict[f'correlation between round {my_round}'] = [None]*9
    for round_1, round_2 in rounds_combination:
        round_1_data = data.loc[data.round_number == round_1].labels
        round_2_data = data.loc[data.round_number == round_2].labels
        correlation = np.corrcoef(round_1_data, round_2_data)[0, 1]
        results_dict[f'correlation between round {round_1}'][round_2-2] = correlation

    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    results_df.columns = [f'and round {i}' for i in range(2, 11)]
    results_df.to_csv(os.path.join(base_directory, 'logs', 'analyze_results', 'rounds_correlation_analysis.csv'))

    return


def find_all_best_models_of_directory(directory: str):
    xls = pd.ExcelFile(os.path.join(base_directory, 'logs', 'analyze_results', 'Best models analysis.xlsx'))
    number_best_fold = defaultdict(list)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet, skiprows=[0])
        for fold in range(6):
            data = df[f'Best Model folder for fold {fold}']
            index_data = data.str.findall(directory)
            for my_index in index_data.index:
                if type(index_data[my_index]) == list and len(index_data[my_index]) > 0:
                    number_best_fold[fold].append(f"Fold: {fold}, Model number: "
                                                  f"{df.iloc[my_index][f'Best Model number for fold {fold}']}")

    print(f'Best models in {directory}')
    for fold in range(6):
        best_models_list = sorted(set(number_best_fold[fold]))
        print(f'Number of best models for fold {fold} is {len(best_models_list)}')
        for item in best_models_list:
            print(item)

    return


def main(new_results_file_name, results_to_use: pd.DataFrame=None):
    # for dir in ['compare_prediction_models_04_05_2020_19_15', 'compare_prediction_models_07_05_2020_11_49',
    #             'compare_prediction_models_17_05_2020_22_12', 'compare_prediction_models_29_04_2020_23_59']:
    #     find_all_best_models_of_directory(dir)
    # return

    # results_path = combine_models_results(['compare_prediction_models_29_04_2020_23_59',
    #                                        'compare_prediction_models_30_04_2020_07_34_new_crf',
    #                                        'compare_prediction_models_30_04_2020_14_29_new_baseline'])
    # # results_path = combine_models_all_results(['compare_prediction_models_25_05_2020_12_53'])
    # results_path = f"all_server_results_['compare_prediction_models_04_05_2020_19_15'].csv"
    # concat_new_results(results_path, old_results_name='all_results_01_06_new_eval.csv',
    #                    new_results_file_name_to_save=new_results_file_name)
    #
    # return

    # correlation_analysis(
    #     os.path.join(base_directory, 'data', 'verbal', 'cv_framework',
    #                  'all_data_single_round_label_crf_raisha_non_nn_turn_model_prev_round_label_all_history_features_'
    #                  'all_history_text_manual_binary_features_predict_first_round_verbal_data.csv'))

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
    table_writer = pd.ExcelWriter(os.path.join(base_directory, 'logs', 'analyze_results', 'Best models analysis.xlsx'),
                                  engine='xlsxwriter')
    not_include_model_num = [16, 24, 32, 43, 47, 58, 69, 80, 94, 105, 119, 130, 145, 156, 170, 181, 187, 391, 402, 413,
                             424, 435, 446, 730, 744]
    all_not_include_model_num = copy.deepcopy(not_include_model_num)
    for num in not_include_model_num:
        all_not_include_model_num.append(str(num))
        for dropout in [None, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
            all_not_include_model_num.append(f'{str(num)}_{dropout}')
    for measure_to_analyze in [['Bin_Fbeta_score_total future payoff < 1/3', 'Bin_Fbeta_1_3'],
                               ['Bin_Fbeta_score_total future payoff > 2/3', 'Bin_Fbeta_2_3'],
                               ['Bin_Fbeta_score_1/3 < total future payoff < 2/3', 'Bin_Fbeta_1_3_2_3'],
                               ['RMSE', 'RMSE'], ['Bin_Accuracy', 'Bin_Accuracy']]:

        for raisha in raishas:
            print(f'Analyze results based on {measure_to_analyze} for {raisha}')
            other_measures = copy.deepcopy(all_measures)
            other_measures.remove(measure_to_analyze[0])
            select_best_model_per_type(main_file_name, raishas=raisha, rounds='All_rounds', table_writer=table_writer,
                                       measure_to_select_best=measure_to_analyze[0],
                                       measure_to_select_best_name=measure_to_analyze[1], all_measures=other_measures,
                                       all_results=results_to_use, all_not_include_model_num=all_not_include_model_num)

    table_writer.save()


if __name__ == '__main__':
    main(new_results_file_name='all_results_05_06_new_eval_select_correct_final_pred.csv')
