import pandas as pd
import os
import utils
from datetime import datetime
import logging
import json
import execute_cv_models
import joblib
import numpy as np


base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
data_directory = os.path.join(base_directory, 'data', condition, 'cv_framework')
run_dir = utils.set_folder(datetime.now().strftime(f'predict_best_models_%d_%m_%Y_%H_%M'), 'logs')


def predict_best_models(best_model_file_name: str):
    all_models_results = pd.DataFrame()
    best_models = pd.read_excel(os.path.join(base_directory, 'logs', best_model_file_name))
    models_to_compare = pd.read_excel(os.path.join(base_directory, 'models_to_compare.xlsx'),
                                      sheet_name='table_to_load', skiprows=[0])
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
        fold_best_models = best_models.loc[best_models.fold == fold]
        for index, row in fold_best_models.iterrows():
            model_name = row['model_name']
            model_num = row['model_num']
            model_info = models_to_compare.loc[models_to_compare.model_name == model_name]
            model_type = model_info['model_type']
            function_to_run = model_info['function_to_run']
            data_file_name = model_info['data_file_name']
            hyper_parameters_str = model_info['hyper_parameters']
            model_folder = row['model_folder']
            model_file_name = row['model_file_name']
            inner_model_folder = row['inner_model_folder']  # foe EvalLSTM models
            trained_model_dir = os.path.join(base_directory, 'logs', model_folder, inner_model_folder)
            trained_model = joblib.load(os.path.join(trained_model_dir, model_file_name))
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
                trained_model=trained_model)
            model_class.load_data_create_model()
            model_class.predict()
            results_dict = model_class.eval_model()
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
    predict_best_models('best_model_file_name.xslx')
