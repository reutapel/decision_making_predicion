import pandas as pd
import os
import utils
from datetime import datetime
import logging
import json
import execute_cv_models
import joblib
import numpy as np
import torch


base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
data_directory = os.path.join(base_directory, 'data', condition, 'cv_framework')
run_dir = utils.set_folder(datetime.now().strftime(f'predict_best_models_%d_%m_%Y_%H_%M'), 'logs')


def predict_best_models(best_model_file_name: str):
    all_models_results = pd.DataFrame()
    best_models = pd.read_excel(os.path.join(base_directory, 'logs', best_model_file_name), sheet_name='table_to_load')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    participants_fold = pd.read_csv(os.path.join(data_directory, 'pairs_folds.csv'))
    participants_fold.index = participants_fold.pair_id
    excel_models_results = utils.set_folder(folder_name='excel_best_models_results', father_folder_path=run_dir)
    table_writer = pd.ExcelWriter(os.path.join(excel_models_results, f'Results_test_data_best_models.xlsx'),
                                  engine='xlsxwriter')
    log_file_name = os.path.join(run_dir, f'LogFile.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )
    for fold in range(6):
        pair_ids_in_fold = participants_fold[f'fold_{fold}']
        fold_split_dict = dict()
        for data_set in ['train', 'test', 'validation']:
            fold_split_dict[data_set] = pair_ids_in_fold.loc[pair_ids_in_fold == data_set].index.tolist()
        for index, row in best_models.iterrows():
            model_name = row['model_name']
            model_num = row['model_num']
            if model_num not in [30, 31, 42, 43, 46, 47]:
                continue

            model_type = row['model_type']
            function_to_run = row['function_to_run']
            data_file_name = row['data_file_name']
            hyper_parameters_str = row[f'hyper_parameters_fold_{fold}']
            model_folder = row[f'model_folder_fold_{fold}']
            if not os.path.exists(os.path.join(base_directory, 'logs', model_folder, f'fold_{fold}')):
                if not os.path.exists(os.path.join(base_directory, 'logs', f'{model_folder}_hyper', f'fold_{fold}')):
                    # the folder we need not exists
                    print(f'fold {fold} in folder {model_folder} is not exists')
                    continue
                else:
                    model_folder = f'{model_folder}_hyper'
            model_version_num = row[f'model_version_num_fold_{fold}']
            model_file_name = f'{model_version_num}_{model_type}_{model_name}_fold_{fold}.pkl'
            if function_to_run == 'ExecuteEvalLSTM':
                inner_model_folder = f'{model_version_num}_{model_type}_{model_name}_100_epochs_fold_num_{fold}'
            else:
                inner_model_folder = ''
            trained_model_dir = os.path.join(base_directory, 'logs', model_folder, f'fold_{fold}', inner_model_folder)
            # if torch.cuda.is_available() or function_to_run != 'ExecuteEvalLSTM':
            trained_model = joblib.load(os.path.join(trained_model_dir, model_file_name))
            # else:
            #     trained_model = torch.load(os.path.join(trained_model_dir, model_file_name),
            #                                map_location=torch.device('cpu'))

            # get hyper parameters as dict
            if type(hyper_parameters_str) == str:
                hyper_parameters_dict = json.loads(hyper_parameters_str)
            else:
                hyper_parameters_dict = None

            metadata_dict = {'model_num': model_num, 'model_type': model_type, 'model_name': model_name,
                             'data_file_name': data_file_name, 'hyper_parameters_str': hyper_parameters_dict,
                             'fold': fold}

            metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index').T
            model_class = getattr(execute_cv_models, function_to_run)(
                model_num, fold, run_dir, model_type, model_name, data_file_name, fold_split_dict, table_writer,
                data_directory, hyper_parameters_dict, excel_models_results, trained_model_dir=trained_model_dir,
                trained_model=trained_model, model_file_name=model_file_name)
            model_class.load_data_create_model()
            model_class.predict(predict_type='test')
            results_dict = model_class.eval_model(predict_type='test')
            results_df = pd.DataFrame.from_dict(results_dict).T
            results_df['raisha_round'] = results_df.index
            results_df[['Raisha', 'Round']] = results_df.raisha_round.str.split(expand=True)
            results_df = results_df.drop('raisha_round', axis=1)
            results_df.index = np.zeros(shape=(results_df.shape[0],))
            results_df = metadata_df.join(results_df)
            all_models_results = pd.concat([all_models_results, results_df], sort='False')
            utils.write_to_excel(model_class.model_table_writer, 'Model results', ['Model results'], results_df)
            model_class.model_table_writer.save()

    utils.write_to_excel(table_writer, 'All models results', ['All models results'], all_models_results)
    table_writer.save()

    logging.info(f'Finish predict best models')
    print(f'Finish predict best models')


if __name__ == '__main__':
    predict_best_models('best_model_file_name.xlsx')
