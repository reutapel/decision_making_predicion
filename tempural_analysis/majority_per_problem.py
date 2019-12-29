__all__ = ['MajorityPerProblem']
import pandas as pd
import numpy as np
import copy


def add_prob_key(x: pd.DataFrame):
    if 'player_x_lottery_t_0' in x.columns:
        appendix = 'player'
    else:
        appendix = 'group'

    # create problem key
    x['prob_key'] = x[appendix + '_x_lottery_t_0'].map(str) + '_' + x[appendix + '_y_lottery_t_0'].map(str) + '_' +\
                    x[appendix + '_p_lottery_t_0'].map(str)

    return x


class MajorityPerProblem:
    def __init__(self):
        self.per_gender = False
        self.majority_class_per_problem = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        curr_x = copy.deepcopy(x)
        curr_x = copy.deepcopy(add_prob_key(curr_x))
        curr_x['index_col'] = curr_x.index
        y_prob_key = pd.DataFrame(y).merge(curr_x[['prob_key', 'index_col']], left_index=True, right_index=True)
        pivot = pd.pivot_table(y_prob_key, index=['prob_key'], columns=['label'], aggfunc=lambda col: len(col.unique()))
        pivot = pivot.fillna(0)
        pivot.columns = pivot.columns.levels[1].astype(str)
        pivot['majority'] = np.where(pivot['-1'] >= pivot['1'], -1, 1)
        self.majority_class_per_problem = pivot['majority'].map(int)

        return self

    def predict(self, x: pd.DataFrame):
        curr_x = copy.deepcopy(x)
        curr_x = add_prob_key(curr_x)

        predictions = pd.DataFrame(curr_x['prob_key']).merge(pd.DataFrame(self.majority_class_per_problem),
                                                             right_index=True, left_on='prob_key', how='left')
        return predictions['majority']


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
