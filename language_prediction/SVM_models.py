from sklearn.svm import SVR, SVC
import numpy as np
import pandas as pd
from language_prediction.utils import *


class SVMTotal:
    def __init__(self):
        self.model = SVR()

    def fit(self, train_x, train_y):
        self.model = self.model.fit(train_x, train_y)

    def predict(self, validation_x, validation_y):
        predictions = self.model.predict(validation_x)
        validation_y.name = 'labels'
        if predictions.dtype == float:  # regression- create bins to measure the F-score
            bin_prediction, bin_test_y = create_bin_columns(predictions, validation_y)
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
