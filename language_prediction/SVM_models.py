from sklearn.svm import SVR, SVC
import numpy as np
import pandas as pd


class SVMTotal:
    def __init__(self):
        self.model = SVR()

    def fit(self, train_x, train_y):
        self.model = self.model.fit(train_x, train_y)

    def predict(self, validation_x, validation_y):
        predictions = self.model.predict(validation_x)
        validation_y.name = 'label'
        if predictions.dtype == float:  # regression- create bins to measure the F-score
            # for prediction
            keep_mask = predictions < 0.33
            bin_prediction = np.where(predictions < 0.67, 1, 2)
            bin_prediction[keep_mask] = 0
            bin_prediction = pd.Series(bin_prediction, name='bin_predictions', index=validation_y.index)
            # for test_y
            keep_mask = validation_y < 0.33
            bin_test_y = np.where(validation_y < 0.67, 1, 2)
            bin_test_y[keep_mask] = 0
            bin_test_y = pd.Series(bin_test_y, name='bin_label', index=validation_y.index)
        else:
            bin_prediction, bin_test_y = pd.Series(name='bin_prediction'), pd.Series(name='bin_label')
        predictions = validation_y.join(pd.Series(predictions, name='predictions', index=validation_y.index)).\
            join(bin_test_y).join(bin_prediction)

        return predictions


class SVMTurn:
    def __init__(self):
        self.model = SVC()

    def fit(self, train_x, train_y):
        self.model = self.model.fit(train_x, train_y)

    def predict(self, validation_x):
        return self.model.predict(validation_x)
