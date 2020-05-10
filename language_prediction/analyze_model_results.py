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
                files_path = join(folder_path, inner_folder, 'excel_models_results')
                folder_files = [f for f in listdir(files_path) if isfile(join(files_path, f))]
                for file in folder_files:
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
                df = pd.read_excel(os.path.join(files_path, f'Results_{inner_folder}_all_models.xlsx'),
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
                               sheet_name: str= 'Sheet1', all_measures: list=None,):
    all_results = pd.read_excel(file_name, sheet_name=sheet_name)
    all_rounds_all_raisha = all_results.loc[(all_results.Raisha == raishas) & (all_results.Round == rounds)]
    all_rounds_all_raisha = all_rounds_all_raisha.loc[~all_rounds_all_raisha.model_num.isin([181, 187])]
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
            argmin_index = data[measure_to_select_best].idxmin()
            data = data.loc[argmin_index]
            best_results[model_type] = {f'Best {measure_to_select_best} for fold {fold}': data[measure_to_select_best],
                                        f'Best Model number for fold {fold}': data.model_num,
                                        f'Best Model name for fold {fold}': data.model_name}
            if all_measures is not None:  # add all other measures of the best model
                for measure in all_measures:
                    best_results[model_type][f'Best {measure} for fold {fold}'] = data[measure]

        all_best_results = update_default_dict(all_best_results, best_results)

    all_best_results = pd.DataFrame.from_dict(all_best_results).T
    all_best_results.index.name = 'model_type'

    measure_to_select_all_folds = [f'Best {measure_to_select_best} for fold {fold}' for fold in range(6)]
    all_best_results[f'average_{measure_to_select_best}'] = all_best_results[measure_to_select_all_folds].mean(axis=1)
    all_best_results[f'STD_{measure_to_select_best}'] = all_best_results[measure_to_select_all_folds].std(axis=1)

    if all_measures is not None:  # add all other measures of the best model
        for measure in all_measures:
            measure_all_folds = [f'Best {measure} for fold {fold}' for fold in range(6)]
            all_best_results[f'average_{measure}'] = all_best_results[measure_all_folds].mean(axis=1)
            all_best_results[f'STD_{measure}'] = all_best_results[measure_all_folds].std(axis=1)

    write_to_excel(table_writer, f'{measure_to_select_best_name}_{raishas}',
                   headers=[f'best_models_based_on_{measure_to_select_best_name}_for_{raishas}'], data=all_best_results)
    return all_best_results


def concat_new_results(new_results_name: str, old_results_name: str, new_results_file_name_to_save: str):
    all_results = pd.read_excel(os.path.join(base_directory, 'logs', old_results_name), sheet_name='Sheet1',
                                index_col=[0])
    # all_results = all_results.loc[~all_results.model_type.isin(['SVMTurn'])]
    svm_results = pd.read_csv(os.path.join(base_directory, 'logs', new_results_name))
    # svm_results = svm_results.loc[svm_results.model_type.isin(['SVMTotal', 'SVMTurn', 'CRF'])]
    svm_results.rename(columns={'Unnamed: 0': 'folder'}, inplace=True)

    all_results = all_results[svm_results.columns]

    check = pd.concat([all_results, svm_results], axis=0)
    check.to_excel(os.path.join(base_directory, 'logs', f'{new_results_file_name_to_save}.xlsx'))


def correlation_analysis(data_path: str):
    """Analyze the correlation between rounds"""
    data = pd.read_csv(data_path)
    data = data.loc[data.raisha == 0]
    rounds_list = list(range(1, 11))
    rounds_combination = itertools.combinations(rounds_list, 2)
    for round_1, round_2 in rounds_combination:
        round_1_data = data.loc[data.round_number == round_1].values
        round_2_data = data.loc[data.round_number == round_2].values



def main():
    # results_path = combine_models_results(['compare_prediction_models_03_05_2020_11_53'])
    # results_path = combine_models_all_results(['compare_prediction_models_04_05_2020_19_15'])

    new_results_file_name = 'all_results_07_05.xlsx'

    # concat_new_results(results_path, old_results_name='all_results_07_05.xlsx',
    #                    new_results_file_name_to_save=new_results_file_name)

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
    raishas = [f'raisha_{raisha}' for raisha in range(9)]
    raishas.append('All_raishas')
    table_writer = pd.ExcelWriter(os.path.join(base_directory, 'logs', 'analyze_results', 'Best models analysis.xlsx'),
                                  engine='xlsxwriter')
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
                                       measure_to_select_best_name=measure_to_analyze[1], all_measures=other_measures)

    table_writer.save()


if __name__ == '__main__':
    main()
