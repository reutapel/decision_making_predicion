from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Softmax
import pandas as pd
import numpy as np
import joblib
import os


class LinearKerasModel:
    """
    This is a simple keras model to predict the DM decision at each step using a given features set
    """
    def __init__(self, input_dim: int, batch_size: int=10):
        """
        :param input_dim: the input dimension
        """
        # define the keras model
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=input_dim, activation='relu'))
        # self.model.add(Dense(50, activation='relu'))
        # self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        # compile the keras model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.batch_size = batch_size
        self.epochs = 150
        print(f'Model summary:\n {self.model.summary()}.\n'
              f'Run with {self.epochs} epochs and {self.batch_size} batch size')

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, shuffle=False)

        return self

    def predict(self, x: pd.DataFrame):
        predict = self.model.predict(x, batch_size=self.batch_size)
        predict = np.where(predict == 0, -1, 1).astype(int).astype(int)
        # _, accuracy = self.model.evaluate(x, y)
        return np.squeeze(predict)


class ConvolutionalKerasModel:
    """
    This is a simple keras model to predict the DM decision at each step using a given features set
    """
    def __init__(self, num_features: int, input_len: int, batch_size: int=10):
        """
        :param num_features: the number of features
        """
        # define the keras model
        self.model = Sequential()
        self.model.add(Conv2D(32, (1, num_features), activation='relu', input_shape=(1, input_len, 1),
                              strides=num_features, padding='valid'))
        # self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(32, (1, 3), activation='relu', strides=1))
        # self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='softmax'))
        # compile the keras model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.batch_size = batch_size
        self.epochs = 150
        print(f'Model summary:\n {self.model.summary()}.\n'
              f'Run with {self.epochs} epochs and {self.batch_size} batch size')

    def fit(self, x: pd.DataFrame, y: pd.Series):
        # TODO reshape to match network
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, shuffle=False)

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
    model = LinearKerasModel(input_dim=x.shape[1])
    model.fit(x, y)
    prediction = model.predict(x)
    _, accuracy = model.model.evaluate(x, y)
    print(f'Accuracy for test data is: {accuracy}')


if __name__ == '__main__':
    main()

