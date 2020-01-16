__all__ = ['PredictLastDecision']
import pandas as pd
import numpy as np
import copy


class PredictLastDecision:
    """
    For each sample predict the decision from the previous round -->
     must got the feature group_sender_payoff_i for round i+1
    """
    def __init__(self):
        pass

    def fit(self, x: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, x: pd.DataFrame):
        x = x.copy()
        x['k_size'] = x.index.str.split('_').map(lambda x: x[2])
        x.k_size = x.k_size.astype(int)
        # to get the column of group_sender_payoff_i for round i_1
        x.k_size = x.k_size - 2
        prediction = pd.DataFrame(columns=['prediction'])
        for i in range(9):
            check = pd.DataFrame(x.loc[x.k_size == i][f'group_sender_payoff_{i}'])
            check.columns = ['prediction']
            prediction = prediction.append(check)
        x = x.merge(prediction, right_index=True, left_index=True)
        return x.predictions


# def main():
#     obj = MajorityPerProblem()
#     x = pd.DataFrame({'player_x_lottery_t_0': [4, 5, 7, 8, 4, 4], 'player_y_lottery_t_0': [6, 7, 9, 10, 6, 6],
#                       'player_p_lottery_t_0': [0.1, 0.2, 0.3, 0.4, 0.1, 0.1]})
#     y = pd.Series([1, 1, 1, 0, 0, 1])
#     y.name = 'label'
#     obj.fit(x, y)
#     prediction = obj.predict(x)
#     prediction = prediction
#
#
# if __name__ == '__main__':
#     main()
