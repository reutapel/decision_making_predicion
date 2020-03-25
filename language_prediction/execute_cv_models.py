import pandas as pd
import tempural_analysis.utils as utils
import os
import argparse
import logging
from language_prediction.crf import LinearChainCRF
from sklearn import metrics
import math
import language_prediction.SVM_models as SVM_models
from language_prediction.utils import *
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from language_prediction.dataset_readers import TextExpDataSetReader, LSTMDatasetReader
import torch.optim as optim
import language_prediction.models as models
from allennlp.training.metrics import *
from datetime import datetime
from allennlp.data.vocabulary import Vocabulary
from tempural_analysis import *
import logging
import joblib
import torch
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from .utils import calculate_measures_for_continues_labels


per_round_predictions_name = 'per_round_predictions'
per_round_labels_name = 'per_round_labels'


class ExecuteEvalModel:
    """
    This is a father class for all models we want to compare.
    Load data, split to train-validation based on the fold_split_dict.
    Train model on train data.
    Predict on validation data.
    Evaluate model on validation data.
    Save model, log, return the evaluation.
    """
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str):

        self.model_num = model_num
        self.fold = fold
        self.fold_dir = fold_dir
        self.model_type = model_type
        self.model_name = model_name
        self.data_file_name = data_file_name
        self.fold_split_dict = fold_split_dict
        self.table_writer = table_writer
        self.model = None
        self.prediction = pd.DataFrame()
        self.data_directory = data_directory
        self.train_pair_ids = self.fold_split_dict['train']
        self.val_pair_ids = self.fold_split_dict['validation']
        self.results_dict = dict()
        print(f'Create Model: model num: {model_num}, model_type: {model_type}, model_name: {model_name}.'
              f'Data file name: {data_file_name}')
        logging.info(f'Create Model: model num: {model_num}, model_type: {model_type}, model_name: {model_name}.'
                     f'Data file name: {data_file_name}')

    def load_data_create_model(self):
        """This function should load the data, split to train-validation and create the model"""
        raise NotImplementedError

    def fit_predict(self):
        """This function should fit the model on the train data, predict on the validation data and dave the results"""
        raise NotImplementedError

    def eval_model(self):
        """This function should use the prediction of the model and eval these results"""
        raise NotImplementedError

    def save_model_prediction(self, data_to_save: pd.DataFrame, save_model=True, sheet_prefix_name: str='All'):
        """
        Save the model predictions and the model itself
        :param data_to_save: the data to save
        :param save_model: whether to save the model
        :param sheet_prefix_name: the sheet prefix name to save
        :return:
        """
        # save the model
        if save_model:
            joblib.dump(self.model, os.path.join(
                self.fold_dir, f'{self.model_num}_{self.model_type}_{self.model_name}_fold_{self.fold}.pkl'))
        utils.write_to_excel(self.table_writer, f'{self.model_name}_{sheet_prefix_name}_predictions',
                             headers=[f'{sheet_prefix_name} predictions for model {self.model_num}: {self.model_name} '
                                      f'of type {self.model_type} in fold {self.fold}'], data=data_to_save)

    def total_payoff_calculate_measures(self, final_total_payoff_prediction_column: str, total_payoff_label_column: str,
                                        raisha_column_name: str = 'raisha'):
        """
        Calculate the measures for seq models per raisha
        :param total_payoff_label_column: the name of the label column
        :param final_total_payoff_prediction_column: the name of the prediction label
        :param raisha_column_name:
        :return:
        """
        # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
        _, results_dict = \
            calculate_measures_for_continues_labels(
                self.prediction, final_total_payoff_prediction_column='predictions',
                total_payoff_label_column='labels',
                label_options=['total future payoff < 1/3', '1/3 < total future payoff < 2/3',
                               'total future payoff > 2/3'])
        if raisha_column_name in self.prediction.columns:  # do the raisha analysis
            raisha_options = self.prediction[raisha_column_name].unique()
            results_dict = dict()
            for raisha in raisha_options:
                raisha_data = self.prediction.loc[self.prediction[raisha_column_name] == raisha]
                _, results_dict_raisha = \
                    calculate_measures_for_continues_labels(
                        raisha_data, final_total_payoff_prediction_column=final_total_payoff_prediction_column,
                        total_payoff_label_column=total_payoff_label_column,
                        label_options=['total future payoff < 1/3', '1/3 < total future payoff < 2/3',
                                       'total future payoff > 2/3'], raisha=f'_raisha_{str(raisha)}')
                results_dict.update(results_dict_raisha)

        return results_dict

    def flat_seq_predictions_multiple_columns(self, label_column_name_per_round: str,
                                              prediction_column_name_per_round: str) -> pd.DataFrame:
        """
        Use the prediction DF to get one column of all rounds predictions and labels, in order to calculate
        the per round measures
        :param label_column_name_per_round: the name od the label column per round (for example: y_0, y_1, ..., y_9)
        :param prediction_column_name_per_round: the name od the prediction column per round
        (for example: y_prime_0, y_prime_1, ..., y_prime_9)
        :return: pd.Dataframe with 2 columns: labels and predictions with the labels and predictions per round for
        the saifa data
        """

        flat_data = dict()
        index_to_use = None
        for flat_col_name, name_for_dict in [[label_column_name_per_round, per_round_labels_name],
                                             [prediction_column_name_per_round, per_round_predictions_name]]:
            cols = [f'{flat_col_name}_{i}' for i in range(10)]
            data = self.prediction[cols]
            data.index = self.prediction.raisha
            stack_data = data.stack()
            # get only the relevant rounds --> the saifa rounds
            stack_data = stack_data.loc[stack_data != '-']
            temp_index = [f'raisha_{r}_round_{ro}' for r, ro in stack_data.index.values]
            stack_data = stack_data.reset_index(drop=True)
            # if index_to_use is None:
            #     temp_index = [f'raisha_{r}_round_{ro}' for r, ro in stack_data.index.values]
            #     stack_data = stack_data.reset_index(drop=True)
            #     index_to_use = stack_data.index
            # else:
            #     stack_data.index = index_to_use
            flat_data[name_for_dict] = stack_data

        try:
            if (flat_data[per_round_labels_name].index == flat_data[per_round_predictions_name].index).sum() == \
                    flat_data[per_round_predictions_name].shape[0]:  # if there are index that  are not the same
                flat_index = pd.DataFrame(temp_index, index=index_to_use)[0].str.split('_', expand=True)
                flat_index.columns = [1, 'raisha', 2, 3, 'round']
                flat_index = flat_index[['raisha', 'round']]
                flat_data_df = pd.DataFrame.from_dict(flat_data)
                flat_data_df = flat_data_df.merge(flat_index, left_index=True, right_index=True)
                # save the flat data
                self.save_model_prediction(data_to_save=flat_data_df, save_model=False, sheet_prefix_name='Per round')

                return flat_data_df

        except Exception:
            logging.exception(f'Failed in flat_seq_predictions_multiple_columns: index after flat are not the same')
            return pd.DataFrame()

    def flat_seq_predictions_list_column(self, label_column_name_per_round: str,
                                         prediction_column_name_per_round: str) -> pd.DataFrame:
        """
        Use the prediction DF to get one column of all rounds predictions and labels, in order to calculate
        the per round measures
        :param label_column_name_per_round: the name od the label column per round (for example: y_0, y_1, ..., y_9)
        :param prediction_column_name_per_round: the name od the prediction column per round
        (for example: y_prime_0, y_prime_1, ..., y_prime_9)
        :return: pd.Dataframe with 2 columns: labels and predictions with the labels and predictions per round for
        the saifa data
        """

        flat_data_dict = dict()
        for list_column, new_column in [[label_column_name_per_round, per_round_labels_name],
                                        [prediction_column_name_per_round, per_round_predictions_name]]:
            # create a pd with [new_column, 'raisha', 'sample_id'] columns
            flat_data = copy.deepcopy(self.prediction)
            # reset index to get numeric index for the future merge
            flat_data['sample_id'] = flat_data.index
            flat_data.reset_index(inplace=True, drop=True)
            flat_data = flat_data[[list_column, 'raisha', 'sample_id']]
            lens_of_lists = flat_data[list_column].apply(len)
            origin_rows = range(flat_data.shape[0])
            destination_rows = np.repeat(origin_rows, lens_of_lists)
            non_list_cols = [idx for idx, col in enumerate(flat_data.columns) if col != list_column]
            expanded_df = flat_data.iloc[destination_rows, non_list_cols].copy()
            expanded_df[new_column] = [i for items in flat_data[list_column] for i in items]
            # remove non 0/1 rows and reset index
            expanded_df.reset_index(inplace=True, drop=True)
            expanded_df = expanded_df.loc[expanded_df[new_column].isin(['0', '1'])]
            flat_data_dict[new_column] = expanded_df[[new_column]]
            flat_data_dict['metadata'] = expanded_df[['raisha', 'sample_id']]

        # concat the new labels and new predictions per round
        flat_data = flat_data_dict[per_round_labels_name].merge(flat_data_dict[per_round_predictions_name],
                                                                left_index=True, right_index=True).\
            merge(flat_data_dict['metadata'], left_index=True, right_index=True)
        flat_data.reset_index(inplace=True, drop=True)
        # save flat data
        self.save_model_prediction(data_to_save=flat_data, save_model=False, sheet_prefix_name='Per round')

        return flat_data


class ExecuteEvalCRF(ExecuteEvalModel):
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str, hyper_parameters_dict: dict):
        super(ExecuteEvalCRF, self).__init__(model_num, fold, fold_dir, model_type, model_name, data_file_name,
                                             fold_split_dict, table_writer, data_directory)
        self.squared_sigma = hyper_parameters_dict['squared_sigma']
        self.predict_future = hyper_parameters_dict['self.predict_future']
        self.use_forward_backward_fix_history = hyper_parameters_dict['use_forward_backward_fix_history']
        self.use_viterbi_fix_history = hyper_parameters_dict['use_viterbi_fix_history']
        self.predict_only_last = hyper_parameters_dict['predict_only_last']
        features_file_name = hyper_parameters_dict['features_file']
        self.model = None
        parser = argparse.ArgumentParser()
        parser.add_argument('train_data_file', help="data file for training input")
        parser.add_argument('features_file', help="the features file name input.")
        parser.add_argument('model_file', help="the model file name. (output)")
        self.args = parser.parse_args([
                os.path.join(self.data_directory, self.data_file_name),
                os.path.join(self.data_directory, features_file_name),
                os.path.join(self.fold_dir,
                             f'{self.model_num}_{self.model_type}_{self.model_name}_fold_{self.fold}.pkl'),
            ])
        self.correct_count = None
        self.total_count = None
        self.total_seq_count = None

    def load_data_create_model(self):
        print(f'Data file name: {self.args.train_data_file}')
        logging.info(f'Data file name: {self.args.train_data_file}')
        self.model = LinearChainCRF(squared_sigma=self.squared_sigma, predict_future=self.predict_future)

    def fit_predict(self):
        self.model.train(self.args.train_data_file, self.args.model_file, self.args.features_file,
                         vector_rep_input=True, pair_ids=self.train_pair_ids,
                         use_forward_backward_fix_history=self.use_forward_backward_fix_history)

        self.model.load(self.args.model_file, self.args.features_file, vector_rep_input=True)
        self.correct_count, self.total_count, self.prediction, self.total_seq_count = \
            self.model.test(self.args.train_data_file, predict_only_last=self.predict_only_last,
                            pair_ids=self.val_pair_ids, use_viterbi_fix_history=self.use_viterbi_fix_history)
        # create the bin analysis columns in self.prediction
        if 'total_payoff_label' in self.prediction.columns and 'total_payoff_prediction' in self.prediction.columns:
            bin_prediction, bin_test_y = create_bin_columns(self.prediction.total_payoff_prediction,
                                                            self.prediction.total_payoff_label)
            self.prediction = self.prediction.join(bin_test_y).join(bin_prediction)
        # in the CRF train the model is already saved
        self.save_model_prediction(data_to_save=self.prediction, save_model=False)

    def eval_model(self):
        try:
            if 'total_payoff_label' in self.prediction.columns and 'total_payoff_prediction' in self.prediction.columns:
                # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
                results_dict = self.total_payoff_calculate_measures(
                        final_total_payoff_prediction_column='total_payoff_prediction',
                        total_payoff_label_column='total_payoff_label')
                # measures per round
                flat_seq_predictions = self.flat_seq_predictions_multiple_columns(label_column_name_per_round='y',
                                                                                  prediction_column_name_per_round='y_prime')
                results_dict_per_round = calculate_per_round_measures(flat_seq_predictions,
                                                                      predictions_column=per_round_predictions_name,
                                                                      label_column=per_round_labels_name,
                                                                      label_options=['DM chose stay home',
                                                                                     'DM chose hotel'])
                results_dict.update(results_dict_per_round)
                if 'raisha' in flat_seq_predictions:
                    results_dict_per_round_per_raisha =\
                        calculate_per_round_per_raisha_measures(flat_seq_predictions,
                                                                predictions_column=per_round_predictions_name,
                                                                label_column=per_round_labels_name,
                                                                label_options=['DM chose stay home', 'DM chose hotel'])
                    results_dict.update(results_dict_per_round_per_raisha)

                return results_dict
        except Exception:
            logging.exception(f'total_payoff_label or total_payoff_prediction not in CRF prediction DF')
            return


class ExecuteEvalSVM(ExecuteEvalModel):
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str, hyper_parameters_dict: dict):
        super(ExecuteEvalSVM, self).__init__(model_num, fold, fold_dir, model_type, model_name, data_file_name,
                                             fold_split_dict, table_writer, data_directory)
        self.label_name = hyper_parameters_dict['label_name']
        self.train_x, self.train_y, self.validation_x, self.validation_y = None, None, None, None

    def load_data_create_model(self):
        if 'pkl' in self.data_file_name:
            data = joblib.load(os.path.join(self.data_directory, self.data_file_name))
        else:
            data = pd.read_csv(os.path.join(self.data_directory, self.data_file_name))

        # get the feature columns
        x_columns = data.columns.tolist()
        x_columns.remove(self.label_name)
        x_columns.remove('participant_code')
        # get train data
        train_data = data.loc[data.participant_code.isin(self.train_pair_ids)]
        self.train_y = train_data[self.label_name]
        self.train_x = train_data[x_columns]
        # get validation data
        validation_data = data.loc[data.participant_code.isin(self.val_pair_ids)]
        self.validation_y = validation_data[self.label_name]
        self.validation_x = validation_data[x_columns]

        # create model
        self.model = getattr(SVM_models, self.model_type)

    def fit_predict(self):
        self.model.fit(self.train_x, self.train_y)
        self.prediction = self.model.predict(self.validation_x)
        self.save_model_prediction(data_to_save=self.prediction)

    def eval_model(self):
        if self.model_type == 'SVMTotal':
            try:
                if 'labels' in self.prediction.columns and 'predictions' in self.prediction.columns:
                    # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
                    results_dict = self.total_payoff_calculate_measures(
                        final_total_payoff_prediction_column='predictions', total_payoff_label_column='labels')

                    return results_dict

            except Exception:
                logging.exception(f'labels or predictions not in SVMTotal prediction DF')
                return

        elif self.model_type == 'SVMTurn':
            # TODO: create the total payoff label and calculate calculate_measures_for_continues_labels
            # TODO: calculate measures per round
            return

        else:
            logging.exception(f'Model type is no SVMTurn and no SVMTotal')
            return


class ExecuteEvalLSTM(ExecuteEvalModel):
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str, hyper_parameters_dict: dict):
        super(ExecuteEvalLSTM, self).__init__(model_num, fold, fold_dir, model_type, model_name, data_file_name,
                                              fold_split_dict, table_writer, data_directory)
        self.lstm_hidden_dim = hyper_parameters_dict['lstm_hidden_dim']
        self.num_epochs = hyper_parameters_dict['num_epochs']
        self.all_validation_accuracy = list()
        self.all_train_accuracy = list()
        self.all_seq_predictions = pd.DataFrame()
        self.all_reg_predictions = pd.DataFrame()
        self.batch_size = 10
        self.trainer = None
        self.linear_hidden_dim = None
        self.avg_loss = 1.0  # if we don't use 2 losses - the weight of each of them should be 1
        self.turn_loss = 1.0
        try:
            if model_type == 'LSTM_turn':
                self.predict_seq = True
                self.predict_avg_total_payoff = False
            elif model_type == 'LSTM_avg':
                self.predict_seq = False
                self.predict_avg_total_payoff = True
            elif model_type == 'LSTM_avg_turn':
                self.predict_seq = True
                self.predict_avg_total_payoff = True
                self.avg_loss = hyper_parameters_dict['avg_loss']
                self.turn_loss = hyper_parameters_dict['turn_loss']
            elif model_type == 'LSTM_turn_linear':
                self.predict_seq = True
                self.predict_avg_total_payoff = False
                self.linear_hidden_dim = hyper_parameters_dict['linear_hidden_dim']
        except Exception:
            logging.exception(f'None of the optional types were given --> can not continue')
            return

    def load_data_create_model(self):
        run_log_directory = utils.set_folder(
            datetime.now().strftime(f'{self.model_name}_{self.num_epochs}_epochs_{self.fold}_folds_'
                                    f'{self.lstm_hidden_dim}_hidden_dim_%d_%m_%Y_%H_%M_%S'), self.fold_dir)
        all_data_file_path = os.path.join(self.data_directory, self.data_file_name)
        # load train data
        train_reader = LSTMDatasetReader(pair_ids=self.train_pair_ids)
        test_reader = LSTMDatasetReader(pair_ids=self.val_pair_ids)

        train_instances = train_reader.read(all_data_file_path)
        validation_instances = test_reader.read(all_data_file_path)
        vocab = Vocabulary.from_instances(train_instances + validation_instances)

        hotel_label_0 = True if vocab._index_to_token['labels'][0] == 'hotel' else False

        metrics_dict_seq = {
            'Accuracy': CategoricalAccuracy(),  # BooleanAccuracy(),
            # 'auc': Auc(),
            'F1measure_hotel_label': F1Measure(positive_label=vocab._token_to_index['labels']['hotel']),
            'F1measure_home_label': F1Measure(positive_label=vocab._token_to_index['labels']['stay_home']),
        }

        metrics_dict_reg = {
            'mean_absolute_error': MeanAbsoluteError(),
        }

        # TODO: change this if necessary
        # batch_size should be: 10 or 9 depends on the input
        # and not shuffle so all the data of the same pair will be in the same batch
        iterator = BasicIterator(batch_size=self.batch_size)  # , instances_per_epoch=10)
        iterator.index_with(vocab)
        lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(train_reader.num_features, self.lstm_hidden_dim, batch_first=True,
                                                   num_layers=1, dropout=0.0))
        model = models.LSTMAttention2LossesFixTextFeaturesDecisionResultModel(
            encoder=lstm, metrics_dict_seq=metrics_dict_seq, metrics_dict_reg=metrics_dict_reg, vocab=vocab,
            predict_seq=self.predict_seq, predict_avg_total_payoff=self.predict_avg_total_payoff,
            hotel_label_0=hotel_label_0, linear_dim=self.linear_hidden_dim, seq_weight_loss=self.turn_loss,
            reg_weight_loss=self.avg_loss)
        print(model)
        if torch.cuda.is_available():
            cuda_device = 0
            model = model.cuda(cuda_device)
        else:
            cuda_device = -1
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        validation_metric = '+Accuracy' if self.predict_seq else '-loss'

        self.trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_instances,
            validation_dataset=validation_instances,
            num_epochs=self.num_epochs,
            shuffle=False,
            serialization_dir=run_log_directory,
            patience=10,
            histogram_interval=10,
            cuda_device=cuda_device,
            validation_metric=validation_metric,
        )

    def fit_predict(self):

        model_dict = self.trainer.train()

        # print(f'{self.model_name}: evaluation measures for fold {self.fold} are:')
        # for key, value in model_dict.items():
        #     print(f'{key}: {value}')

        if self.predict_seq:
            self.all_seq_predictions = pd.DataFrame.from_dict(self.model.seq_predictions, orient='index')
            self.all_seq_predictions['final_prediction'] = \
                self.all_seq_predictions[f'predictions_{model_dict["training_epochs"]+1}']
            self.all_seq_predictions['final_total_payoff_prediction'] = \
                self.all_seq_predictions[f'total_payoff_prediction_{model_dict["training_epochs"]+1}']
            self.save_model_prediction(data_to_save=self.all_seq_predictions, sheet_prefix_name='seq_predictions')

        if self.predict_avg_total_payoff:
            self.all_reg_predictions = self.model.reg_predictions
            self.all_reg_predictions['final_total_payoff_prediction'] = \
                self.all_reg_predictions[f'prediction_{model_dict["training_epochs"]+1}']
            self.save_model_prediction(data_to_save=self.all_reg_predictions, sheet_prefix_name='reg_predictions',
                                       save_model=False)
            self.prediction = self.all_reg_predictions

    def eval_model(self):
        try:
            if 'label' in self.prediction.columns and 'final_total_payoff_prediction' in self.prediction.columns:
                # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
                results_dict = self.total_payoff_calculate_measures(
                        final_total_payoff_prediction_column='final_prediction',
                        total_payoff_label_column='labels')
                # measures per round
                if self.predict_seq and not self.predict_avg_total_payoff:
                    self.prediction = self.all_seq_predictions
                    flat_seq_predictions = self.flat_seq_predictions_list_column(
                        label_column_name_per_round='labels',
                        prediction_column_name_per_round='final_prediction')
                    results_dict_per_round = calculate_per_round_measures(flat_seq_predictions,
                                                                          predictions_column=per_round_predictions_name,
                                                                          label_column=per_round_labels_name,
                                                                          label_options=['DM chose stay home',
                                                                                         'DM chose hotel'])
                    results_dict.update(results_dict_per_round)
                    if 'raisha' in flat_seq_predictions:
                        results_dict_per_round_per_raisha =\
                            calculate_per_round_per_raisha_measures(flat_seq_predictions,
                                                                    predictions_column=per_round_predictions_name,
                                                                    label_column=per_round_labels_name,
                                                                    label_options=['DM chose stay home',
                                                                                   'DM chose hotel'])
                        results_dict.update(results_dict_per_round_per_raisha)

                return results_dict
        except Exception:
            logging.exception(f'total_payoff_label or total_payoff_prediction not in CRF prediction DF')
            return
