from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import joblib
import os


class KerasModel:
    """
    This is a simple keras model to predict the DM decision at each step using a given features set
    """
    def __init__(self, input_dim: int, batch_size: int=10):
        """
        :param input_dim: the input dimension
        """
        # define the keras model
        self.model = Sequential()
        self.model.add(Dense(300, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        # compile the keras model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.batch_size = batch_size

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.model.fit(x, y, epochs=300, batch_size=self.batch_size, shuffle=False)

        return self

    def predict(self, x: pd.DataFrame):
        predict = self.model.predict(x, batch_size=self.batch_size)
        predict = np.where(predict == 0, -1, 1).astype(int).astype(int)
        # _, accuracy = self.model.evaluate(x, y)
        return np.squeeze(predict)


def main():
    base_directory = os.path.abspath(os.curdir)
    condition = ''
    data_directory = os.path.join(base_directory, 'data', condition)
    data = joblib.load(os.path.join(data_directory, 'labeledTrainData_bert_embedding.pkl'))
    x = pd.DataFrame.from_records(data.text_0.values, index=data.text_0.index)
    y = data.single_label
    model = KerasModel(input_dim=x.shape[1])
    model.fit(x, y)
    prediction = model.predict(x)
    _, accuracy = model.model.evaluate(x, y)
    print(f'Accuracy for test data is: {accuracy}')


if __name__ == '__main__':
    main()

