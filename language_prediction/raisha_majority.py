import pandas as pd
import numpy as np


class PerRoundRaishaMajority:
    def __init__(self):
        self.most_frequent = None

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        self.most_frequent = train_y.mode().values[0]

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        validation_x = validation_x.merge(validation_y, right_index=True, left_on='sample_id')
        data_to_calculate = validation_x.loc[validation_x.raisha == 0]
        prediction = pd.DataFrame()
        unique_raisha = validation_x.raisha.unique()
        for raisha in unique_raisha:
            temp = validation_x.loc[validation_x.raisha == raisha][
                ['raisha', 'pair_id', 'round_number', 'sample_id']].copy(deep=True)
            if raisha == 0:
                temp['predictions'] = self.most_frequent
                prediction = pd.concat([prediction, temp])

            else:
                # get the raisha data
                raisha_data = data_to_calculate.loc[data_to_calculate.round_number <= raisha].copy(deep=True)
                raisha_data = pd.DataFrame(raisha_data.groupby(by=['pair_id', 'raisha']).labels.mean())
                raisha_data['predictions'] = np.where(raisha_data >= 0, 1, -1)
                raisha_data['raisha'] = raisha
                raisha_data['pair_id'] = raisha_data.index.get_level_values(0)
                raisha_data = raisha_data.reset_index(drop=True)
                temp = temp.merge(raisha_data[['predictions', 'raisha', 'pair_id']], on=['raisha', 'pair_id'])
                prediction = pd.concat([prediction, temp])

        prediction.index = prediction.sample_id
        prediction = prediction.merge(validation_y, right_index=True, left_index=True)
        return prediction


if __name__ == '__main__':
    data = pd.read_csv(
        '/Users/reutapel/Documents/Documents/Technion/Msc/thesis/experiment/decision_prediction/language_prediction/'
        'data/verbal/cv_framework/all_data_single_round_label_crf_raisha_non_nn_turn_model_prev_round_label_no_saifa'
        '_text_manual_binary_features_predict_first_round_verbal_data.csv')

    obj = PerRoundRaishaMajority()
    data.index = data.sample_id
    labels = data.labels
    data = data.drop('labels', axis=1)
    data = data.reset_index(drop=True)
    obj.fit(data, labels)
    obj.predict(data, labels)
