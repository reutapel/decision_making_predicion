__all__ = ['LearnMajorityLastTrials']
import pandas as pd
import numpy as np
import copy
import logging


class LearnMajorityLastTrials:
    def __init__(self, model):
        self.per_gender = False
        self.classifier = model

    def fit(self, x: pd.DataFrame, y: pd.Series):
        curr_x = copy.deepcopy(x)
        if 'player_receiver_choice_t_10' in curr_x.columns:
            columns = ['player_receiver_choice_t_' + str(i) for i in range(1, 11)]
            if 'player_receiver_choice_first_round' in curr_x.columns:
                columns.append('player_receiver_choice_first_round')
            data_for_majority = curr_x[columns]

        elif 'group_receiver_choice_t_10' in curr_x.columns:
            columns = ['group_receiver_choice_t_' + str(i) for i in range(1, 11)]
            if 'group_receiver_choice_first_round' in curr_x.columns:
                columns.append('group_receiver_choice_first_round')
            data_for_majority = curr_x[columns]

        else:
            raise Exception('Window size is not 10')

        self.classifier.fit(X=data_for_majority, y=y)

        return self

    def predict(self, x: pd.DataFrame):
        curr_x = copy.deepcopy(x)
        try:
            if 'player_receiver_choice_t_10' in curr_x.columns:
                columns = ['player_receiver_choice_t_' + str(i) for i in range(1, 11)]
                if 'player_receiver_choice_first_round' in curr_x.columns:
                    columns.append('player_receiver_choice_first_round')
                data_for_majority = curr_x[columns]

            elif 'group_receiver_choice_t_10' in curr_x.columns:
                columns = ['group_receiver_choice_t_' + str(i) for i in range(1, 11)]
                if 'group_receiver_choice_first_round' in curr_x.columns:
                    columns.append('group_receiver_choice_first_round')
                data_for_majority = curr_x[columns]

            else:
                raise Exception('Window size is not 10')

            predict = self.classifier.predict(data_for_majority)

            return predict

        except Exception as e:
                logging.exception('Window size is not 5 or 1')
                return e


# def main():
#     obj = MajorityLastTrials()
#     x = pd.DataFrame({'player_receiver_choice_t_1': [1, 0, 0, 1, 1], 'player_receiver_choice_t_2': [1, 0, 0, 1, 0],
#                       'player_receiver_choice_t_3': [1, 0, 0, 0, 1], 'player_receiver_choice_t_4': [0, 0, 0, 1, 1],
#                       'player_receiver_choice_t_5': [1, 0, 0, 0, 0]})
#     y = pd.Series([1, 1, 1, 0, 0])
#     y.name = 'label'
#     obj.fit(x, y)
#     prediction = obj.predict(x)
#     prediction = prediction
#
#
# if __name__ == '__main__':
#     main()
