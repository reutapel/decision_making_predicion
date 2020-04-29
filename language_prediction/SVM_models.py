from sklearn.svm import SVR, SVC
import numpy as np
import pandas as pd
import utils
import logging
from sklearn.dummy import DummyClassifier, DummyRegressor


class SVMTotal:
    def __init__(self, features, model_name):
        if 'svm' in str.lower(model_name):
            self.model = SVR(gamma='scale')
        elif 'average' in str.lower(model_name):
            self.model = DummyRegressor(strategy='mean')
        elif 'median' in str.lower(model_name):
            self.model = DummyRegressor(strategy='median')
        else:
            logging.error('Model name not in: svm, average, median')
            print('Model name not in: svm, average, median')
            raise Exception('Model name not in: svm, average, median')
        self.features = features

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        train_x = train_x[self.features]
        self.model = self.model.fit(train_x, train_y)

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        validation_x = validation_x[self.features]
        predictions = self.model.predict(validation_x)
        validation_y.name = 'labels'
        if predictions.dtype == float:  # regression- create bins to measure the F-score
            bin_prediction, bin_test_y = utils.create_bin_columns(predictions, validation_y)
        else:
            bin_prediction, bin_test_y = pd.Series(name='bin_prediction'), pd.Series(name='bin_label')

        predictions = pd.DataFrame(predictions, columns=['predictions'], index=validation_y.index).join(validation_y).\
            join(bin_test_y).join(bin_prediction)

        return predictions


class SVMTurn:
    def __init__(self, features, model_name):
        if 'svm' in str.lower(model_name):
            self.model = SVC(gamma='scale')
        elif 'stratified' in str.lower(model_name):
            self.model = DummyClassifier(strategy='stratified')
        elif 'most_frequent' in str.lower(model_name):
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            logging.error('Model name not in: svm, stratified, most_frequent')
            print('Model name not in: svm, stratified, most_frequent')
            raise Exception('Model name not in: svm, stratified, most_frequent')
        self.features = features
        self.model_name = model_name

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        if 'features' in train_x.columns:
            train_x = pd.DataFrame(train_x['features'].values.tolist(), columns=self.features)
            self.model = self.model.fit(train_x, train_y)
        else:
            logging.exception('No features column when running SVMTurn model --> can not run it')

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        if 'svm' in str.lower(self.model_name):
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

                predictions = validation_x[['raisha', 'pair_id', 'round_number']].\
                    join(pd.Series(all_predictions, name='predictions')). join(pd.Series(validation_y, name='labels'))
                return predictions

            else:
                logging.exception('No features or round_number or raisha column when running SVMTurn model '
                                  '--> can not run it')
                return
        elif 'stratified' in str.lower(self.model_name) or 'most_frequent' in str.lower(self.model_name):
            data = validation_x[validation_x.columns[1]].copy(deep=True)
            predictions = self.model.predict(data)
            validation_x.index = validation_x.sample_id
            predictions = validation_x[['raisha', 'pair_id', 'round_number']].\
                join(pd.Series(predictions, name='predictions', index=validation_x.index)).\
                join(pd.Series(validation_y, name='labels'))
            return predictions
        else:
            logging.error('Model name not in: svm, stratified, most_frequent, average, median')
            print('Model name not in: svm, stratified, most_frequent, average, median')
            raise Exception('Model name not in: svm, stratified, most_frequent')


class RaishaMajorityBaseline:
    def __init__(self, features, model_name):
        self.model_name = model_name
        self.most_frequent = None
        self.mean = None

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        train_x = train_x.merge(train_y, right_index=True, left_index=True)
        rounds = train_x.round_number.unique()
        self.most_frequent = dict()
        for round_num in rounds:
            round_data = train_x.loc[train_x.round_number == round_num].copy(deep=True)
            self.most_frequent[round_num] = round_data.labels.mode().values[0]
        self.most_frequent = pd.DataFrame.from_dict(self.most_frequent, orient='index')
        self.most_frequent.columns = ['predictions']
        train_x = train_x.loc[train_x.raisha == 0]
        proportion = pd.DataFrame(train_x.groupby(by=['pair_id']).labels.mean())
        self.mean = round(proportion.mean().values[0], 2)

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        validation_x = validation_x.merge(validation_y, right_index=True, left_index=True)
        if 'proportion' in self.model_name:
            validation_y = validation_x.copy(deep=True)
            validation_y.labels = np.where(validation_y.labels == 1, 1, 0)
            validation_y = pd.DataFrame(validation_y.groupby(by=['pair_id', 'raisha']).labels.mean())
            validation_y['pair_id'] = validation_y.index.get_level_values(0)
            validation_y['raisha'] = validation_y.index.get_level_values(1)
            validation_y = validation_y.reset_index(drop=True)
        data_to_calculate = validation_x.loc[validation_x.raisha == 0]
        prediction = pd.DataFrame()
        unique_raisha = validation_x.raisha.unique()
        for raisha in unique_raisha:
            temp = validation_x.loc[validation_x.raisha == raisha][
                ['raisha', 'pair_id', 'round_number', 'sample_id']].copy(deep=True)
            if raisha == 0:
                if 'per_round' in self.model_name:
                    temp = temp.merge(self.most_frequent, left_on='round_number', right_index=True)
                else:
                    temp['predictions'] = self.mean
                prediction = pd.concat([prediction, temp])

            else:
                # get the raisha data
                raisha_data = data_to_calculate.loc[data_to_calculate.round_number <= raisha].copy(deep=True)
                raisha_data.labels = np.where(raisha_data.labels == 1, 1, 0)
                raisha_data = pd.DataFrame(raisha_data.groupby(by=['pair_id', 'raisha']).labels.mean())
                if 'per_round' in self.model_name:
                    raisha_data['predictions'] = np.where(raisha_data >= 0.5, 1, -1)
                else:
                    raisha_data.columns = ['predictions']
                raisha_data['raisha'] = raisha
                raisha_data['pair_id'] = raisha_data.index.get_level_values(0)
                raisha_data = raisha_data.reset_index(drop=True)
                temp = temp.merge(raisha_data[['predictions', 'raisha', 'pair_id']], on=['raisha', 'pair_id'])
                prediction = pd.concat([prediction, temp])

        prediction.index = prediction.sample_id
        if 'proportion' in self.model_name:
            prediction = prediction.loc[prediction.round_number == 10]
            prediction = prediction.merge(validation_y, on=['pair_id', 'raisha'])
            prediction.sample_id = prediction.pair_id + '_' + prediction.raisha.map(str)
            prediction.index = prediction.sample_id
            if prediction.predictions.dtype == float:  # regression- create bins to measure the F-score
                bin_prediction, bin_test_y = utils.create_bin_columns(prediction.predictions, prediction.labels)
            else:
                bin_prediction, bin_test_y = pd.Series(name='bin_prediction'), pd.Series(name='bin_label')

            prediction = prediction.join(bin_test_y).join(bin_prediction)

        else:
            prediction = prediction.merge(validation_y, right_index=True, left_index=True)
        return prediction
