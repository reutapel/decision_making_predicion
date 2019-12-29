__all__ = ['MajorityRule']
import pandas as pd


class MajorityRule:
    def __init__(self):
        self.per_gender = False
        self.majority_class = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        counts = y.groupby(y).count()
        self.majority_class = counts.idxmax()

        return self

    def predict(self, x: pd.DataFrame):
        predictions = pd.Series(data=[self.majority_class]*x.shape[0], index=x.index)
        return predictions


# def main():
#     obj = MajorityRule()
#     x = pd.DataFrame({'1': [4, 5, 7, 8], '2': [6, 7, 9, 10]})
#     y = pd.Series([1, 1, 1, 0])
#     obj.fit(x, y)
#     prediction = obj.predict(x)
#     prediction = prediction
#
#
# if __name__ == '__main__':
#     main()
