from sklearn.svm import SVR, SVC
import numpy as np
import pandas as pd
from language_prediction.utils import *
import logging
from sklearn.dummy import DummyClassifier, DummyRegressor


class SVMTotal:
    def __init__(self, features, model_name):
        if 'svm' in model_name:
            self.model = SVR(gamma='scale')
        elif 'stratified' in model_name:
            self.model = DummyClassifier(strategy='stratified')
        elif 'most_frequent' in model_name:
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            logging.error('Model name not in: svm, stratified, most_frequent')
            print('Model name not in: svm, stratified, most_frequent')
            raise Exception('Model name not in: svm, stratified, most_frequent')
        self.features = features

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        train_x = train_x[self.features]
        self.model = self.model.fit(train_x, train_y)

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        validation_x = validation_x[self.features]
        predictions = self.model.predict(validation_x)
        validation_y.name = 'labels'
        if predictions.dtype == float:  # regression- create bins to measure the F-score
            bin_prediction, bin_test_y = create_bin_columns(predictions, validation_y)
        else:
            bin_prediction, bin_test_y = pd.Series(name='bin_prediction'), pd.Series(name='bin_label')

        predictions = pd.DataFrame(predictions, columns=['predictions'], index=validation_y.index).join(validation_y).\
            join(bin_test_y).join(bin_prediction)

        return predictions


class SVMTurn:
    def __init__(self, features, model_name):
        if 'svm' in model_name:
            self.model = SVC(gamma='scale')
        elif 'average' in model_name:
            self.model = DummyRegressor(strategy='mean')
        elif 'median' in model_name:
            DummyRegressor(strategy='median')
        else:
            logging.error('Model name not in: svm, stratified, most_frequent')
            print('Model name not in: svm, stratified, most_frequent')
            raise Exception('Model name not in: svm, stratified, most_frequent')
        self.features = features

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        if 'features' in train_x.columns:
            train_x = pd.DataFrame(train_x['features'].values.tolist(), columns=self.features)
            self.model = self.model.fit(train_x, train_y)
        else:
            logging.exception('No features column when running SVMTurn model --> can not run it')

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        all_predictions = pd.Series()
        predictions_pair_id = pd.DataFrame()
        validation_x.index = validation_x.sample_id
        if 'features' in validation_x.columns and 'round_number' in validation_x.columns and\
                'raisha' in validation_x.columns and 'prev_round_label' in self.features:
            raisha_options = validation_x.raisha.unique()
            for raisha in raisha_options:
                for round_num in range(raisha+1, 11):
                    validation_round = validation_x.loc[(validation_x.round_number == round_num) &
                                                        (validation_x.raisha == raisha)].copy(deep=True)
                    validation_round_features = pd.DataFrame(validation_round['features'].values.tolist(),
                                                             columns=self.features, index=validation_round.sample_id)
                    if round_num > raisha + 1:  # the first round in the saifa --> no prediction for prev round
                        # merge with the previous round precdiction
                        validation_round = validation_round.merge(predictions_pair_id, on=['pair_id', 'raisha']).\
                            set_index(validation_round.index)
                        # remove prev_round_label and put the prediction instead
                        validation_round_features = validation_round_features.drop('prev_round_label', axis=1)
                        prev_round_prediction = validation_round[['prev_round_label']].copy(deep=True)
                        validation_round_features = validation_round_features.merge(prev_round_prediction,
                                                                                    left_index=True, right_index=True)
                        
                    predictions = self.model.predict(validation_round_features)
                    predictions = pd.Series(predictions, name='prev_round_label', index=validation_round.sample_id)
                    all_predictions = pd.concat([all_predictions, predictions])
                    predictions_pair_id = validation_round[['pair_id', 'raisha']].merge(predictions, left_index=True,
                                                                                        right_index=True)

            predictions = validation_x[['raisha', 'pair_id']].join(pd.Series(all_predictions, name='predictions')).\
                join(pd.Series(validation_y, name='labels'))
            return predictions

        else:
            logging.exception('No features or round_number or raisha column when running SVMTurn model '
                              '--> can not run it')
            return
