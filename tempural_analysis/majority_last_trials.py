__all__ = ['MajorityLastTrials']
import pandas as pd
import numpy as np
import copy
import logging


class MajorityLastTrials:
    def __init__(self):
        self.per_gender = False

    def fit(self, x: pd.DataFrame, y: pd.Series):

        return self

    def predict(self, x: pd.DataFrame):
        curr_x = copy.deepcopy(x)
        try:
            if 'player_receiver_choice_t_5' in curr_x.columns:
                columns = ['player_receiver_choice_t_' + str(i) for i in range(1, 6)]
                if 'player_receiver_choice_first_round' in curr_x.columns:
                    columns.append('player_receiver_choice_first_round')
                data_for_majority = curr_x[columns]

            elif 'group_receiver_choice_t_5' in curr_x.columns:
                columns = ['group_receiver_choice_t_' + str(i) for i in range(1, 6)]
                if 'group_receiver_choice_first_round' in curr_x.columns:
                    columns.append('group_receiver_choice_first_round')
                data_for_majority = curr_x[columns]

            elif 'player_receiver_choice_t_1' in curr_x.columns:  # window size = 1, predict based on the last trial
                data_for_majority = curr_x[['player_receiver_choice_t_1']]

            elif 'group_receiver_choice_t_1' in curr_x.columns:  # window size = 1, predict based on the last trial
                data_for_majority = curr_x[['group_receiver_choice_t_1']]

            elif 'player_receiver_choice_t_10' in curr_x.columns:
                columns = ['player_receiver_choice_t_' + str(i) for i in range(1, 11)]
                if 'player_receiver_choice_first_round' in curr_x.columns:
                    columns.append('player_receiver_choice_first_round')
                data_for_majority = curr_x[columns]

            elif 'group_receiver_choice_t_10' in curr_x.columns:
                columns = ['group_receiver_choice_t_' + str(i) for i in range(1, 11)]
                if 'group_receiver_choice_first_round' in curr_x.columns:
                    columns.append('player_receiver_choice_first_round')
                data_for_majority = curr_x[columns]

            else:
                raise Exception('Window size is not 5 or 1 or 10')

            data_for_majority['sum_receiver_choice'] = data_for_majority.sum(axis=1)
            # if more than twice he chose certainty (sum_receiver_choice > 2) --> return 1, otherwise return 0
            data_for_majority['majority'] = np.where(data_for_majority.sum_receiver_choice > 2, 1, -1)

            return data_for_majority['majority']

        except Exception as e:
                logging.exception('Window size is not 5 or 1 or 10')
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
