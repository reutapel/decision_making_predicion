import pandas as pd


class EnsembleClassifier:
    def __init__(self, clf1, clf2, clf3):
        self.clf1 = clf1
        self.clf2 = clf2
        self.clf3 = clf3

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.clf1.fit(x, y)
        self.clf2.fit(x, y)
        self.clf3.fit(x, y)

        return self

    def predict(self, x: pd.DataFrame):
        prediction1 = self.clf1.predict(x)
        prediction2 = self.clf2.predict(x)
        prediction3 = self.clf3.predict(x)

        all_predictions = pd.concat([pd.Series(prediction1), pd.Series(prediction2), pd.Series(prediction3)], axis=1)
        predictions = all_predictions.mode(axis='columns', numeric_only=True)

        return predictions[0].values


# def main():
#     obj = EnsembleClassifier(XGBClassifier(max_depth=10), SVC(), LogisticRegression())
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
