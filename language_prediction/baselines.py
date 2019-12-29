import joblib
import pandas as pd


class Baseline:
    """
    This is a parent class for baselines
    """
    def __init__(self, train_file_path: str, validation_file_path: str, binary_classification: bool=False,
                 label_col_name: str='label', use_first_round: bool=True):
        # load data
        if 'csv' in train_file_path:
            self.train_data = pd.read_csv(train_file_path)
            self.validation_data = pd.read_csv(validation_file_path)
        else:
            self.train_data = joblib.load(train_file_path)
            self.validation_data = joblib.load(validation_file_path)
        if not use_first_round:
            self.train_data = self.train_data.loc[self.train_data.k_size > 1]
            self.validation_data = self.validation_data.loc[self.validation_data.k_size > 1]
        self.binary_classification = binary_classification
        self.label_col_name = label_col_name

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class AverageOverTrainBaseLine(Baseline):
    """
    This baseline calculate the average label over the train data and predict it as this output for all the test data
    """
    def __init__(self, train_file_path: str, validation_file_path: str, binary_classification: bool=False,
                 label_col_name: str='label', use_first_round: bool=True):
        super(AverageOverTrainBaseLine, self).__init__(train_file_path, validation_file_path, binary_classification,
                                                       label_col_name, use_first_round)
        self.train_data_average_label = None

    def fit(self):
        """get train_data_average_label"""
        self.train_data_average_label = self.train_data[self.label_col_name].mean()
        if self.binary_classification:
            if self.train_data_average_label > 0:
                self.train_data_average_label = 1
            else:
                self.train_data_average_label = -1
        print(f'train_data_average_label is: {self.train_data_average_label}')

    def predict(self):
        """
        Predict the labels for the validation data
        :return:
        """
        # set the prediction to the validation data
        self.validation_data = self.validation_data.assign(prediction=self.train_data_average_label)
        self.train_data = self.train_data.assign(prediction=self.train_data_average_label)


class AverageRishaTrainBaseline(Baseline):
    """
    This baseline calculate the average label over the train data per Risha
    and predict this output to all the test data per Risha
    """
    def __init__(self, train_file_path: str, validation_file_path: str, binary_classification: bool=False,
                 label_col_name: str='label', use_first_round: bool=True):
        super(AverageRishaTrainBaseline, self).__init__(train_file_path, validation_file_path, binary_classification,
                                                        label_col_name, use_first_round)
        self.average_label_df = None

    def fit(self):
        """
        Get the train_data_average_label per Risha
        :return:
        """
        self.average_label_df = self.train_data.groupby(by='k_size')[self.label_col_name].mean().rename('prediction')
        if self.binary_classification:
            self.average_label_df[self.average_label_df > 0] = 1
            self.average_label_df[self.average_label_df < 0] = -1
        print(f'average_label_df is: {self.average_label_df}')

    def predict(self):
        """
        Predict the labels for the validation data per k_size
        :return:
        """
        # set the prediction to the validation data
        self.validation_data = self.validation_data.merge(self.average_label_df, right_index=True, left_on='k_size')
        self.train_data = self.train_data.merge(self.average_label_df, right_index=True, left_on='k_size')
