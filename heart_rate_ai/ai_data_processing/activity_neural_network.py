import os
import numpy as np
import pandas as pd
from definitions import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot


class ActivityClassifier:
    def __init__(self):
        self.combined_preprocessed_data_frame = None
        self.training_data_frame = None
        self.scaled_preprocessed_data = None
        self.training_data_percentage = 0.2
        self.model_neurons = 500
        self.model_epochs = 200
        self.model_batch_size = 128
        self.model_layers = 1
        self.model_dropout = 0.2
        self._categorical_data_scaler = None

    @staticmethod
    def initialize_processor():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # model will be trained on GPU 1

    def load_activity_data(self, date):
        # Load the scraped data set
        filename = ROOT_DIR + RAW_DATA_DIRECTORY + date + FILE_EXT
        activity_data_xls = pd.ExcelFile(filename)
        heart_data_frame = pd.read_excel(activity_data_xls, 'HEART')
        step_data_frame = pd.read_excel(activity_data_xls, 'STEPS')
        distance_data_frame = pd.read_excel(activity_data_xls, 'DISTANCE')
        calories_data_frame = pd.read_excel(activity_data_xls, 'CALORIES')

        # Reform the data into an organized dataset
        heart_data_frame = ActivityClassifier.data_reframing(heart_data_frame, date)
        step_data_frame = ActivityClassifier.data_reframing(step_data_frame, date)
        distance_data_frame = ActivityClassifier.data_reframing(distance_data_frame, date)
        calories_data_frame = ActivityClassifier.data_reframing(calories_data_frame, date)

        # Concatenate the data into one set
        frames = [heart_data_frame, step_data_frame, distance_data_frame, calories_data_frame]
        self.combined_preprocessed_data_frame = pd.concat(frames, axis=1)
        self.combined_preprocessed_data_frame.to_excel(ROOT_DIR + PROCESSED_DATA_DIRECTORY + date + FILE_EXT)

        # Load the corresponding training data set
        filename = ROOT_DIR + TRAINING_DATA_DIRECTORY + date + FILE_EXT
        activity_data_xls = pd.ExcelFile(filename)
        self.training_data_frame = pd.read_excel(activity_data_xls, index_col=0)

        # Convert the Activity label into an encoded value
        encoder = LabelEncoder()
        self.training_data_frame['ACTIVITY'] = encoder.fit_transform(self.training_data_frame['ACTIVITY'])

    @staticmethod
    def data_reframing(activity_data_frame, date):
        activity_data_frame['Time'] = date + ' ' + activity_data_frame['Time']
        activity_data_frame['Time'] = pd.to_datetime(activity_data_frame['Time'], format='%Y-%m-%d %H:%M:%S')
        activity_data_frame = activity_data_frame.set_index('Time').resample('1min').mean()
        return activity_data_frame

    def condition_data(self):
        conditioned_data = None
        frames = [self.combined_preprocessed_data_frame, self.training_data_frame]
        combined_frames = pd.concat(frames, axis=1)
        for column in combined_frames:
            values = combined_frames[column].values
            values = np.vstack(values)
            values = np.nan_to_num(values)
            values = values.astype('float32')
            if conditioned_data is not None:
                conditioned_data = np.append(conditioned_data, values, 1)
            else:
                conditioned_data = values
        self._categorical_data_scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_preprocessed_data = self._categorical_data_scaler.fit_transform(conditioned_data)

    def model_setup(self):
        # Define TRAIN and TEST data:
        # Randomly sample the TRAIN data set, all others go into the TEST data set
        train_data = None
        test_data = None
        current_row = 1
        number_of_rows = self.scaled_preprocessed_data.shape[0]
        random_rows = np.random.randint(number_of_rows, size=int(self.training_data_percentage*number_of_rows))
        for sample in self.scaled_preprocessed_data:
            # Check if current row is in the random sample for the training set
            if current_row in random_rows:
                row_data = self.scaled_preprocessed_data[current_row, :]
                if train_data is not None:
                    train_data = np.vstack((train_data, row_data))
                else:
                    train_data = row_data
            else:
                if test_data is not None:
                    test_data = np.vstack((test_data, sample))
                else:
                    test_data = sample
            current_row = current_row + 1

        # Split into inputs and outputs
        train_IN = train_data[:, :-1]
        train_OUT = train_data[:, -1]
        test_IN = test_data[:, :-1]
        test_OUT = test_data[:, -1]

        # Reshape into 3D Format [samples, timesteps, features]
        train_IN3D = train_IN.reshape((train_IN.shape[0], self.model_layers, train_IN.shape[1]))
        test_IN3D = test_IN.reshape((test_IN.shape[0], self.model_layers, test_IN.shape[1]))

        # Design the Network
        model = Sequential()
        model.add(LSTM(self.model_neurons, dropout=self.model_dropout, recurrent_dropout=self.model_dropout,
                       input_shape=(train_IN3D.shape[1], train_IN3D.shape[2])))
        model.add(Dense(self.model_layers, activation='sigmoid'))
        model.compile(loss='mae', optimizer='Adadelta')

        # Fit the Network
        history = model.fit(train_IN3D, train_OUT, epochs=self.model_epochs, batch_size=self.model_batch_size,
                            validation_data=(test_IN3D, test_OUT), verbose=2, shuffle=True)

        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

        # Evaluate Performance:
        # Predicted Output:
        test_pred = model.predict(test_IN3D)
        inv_pred = np.concatenate((test_IN, test_pred), axis=1)
        inv_pred = self._categorical_data_scaler.inverse_transform(inv_pred)
        inv_pred = inv_pred[:, 4]

        # Actual Output:
        inv_output = np.column_stack((test_IN, test_OUT))
        inv_output = self._categorical_data_scaler.inverse_transform(inv_output)
        inv_output = inv_output[:, 4]

        # Root-Mean-Square Error
        rmse = sqrt(mean_squared_error(inv_output, inv_pred))
        print('Test RMSE: %.3f' % rmse)

        # Actual vs Predicted Results
        pyplot.plot(inv_pred, label='Predicted')
        pyplot.plot(inv_output, label='Actual')
        pyplot.legend()
        pyplot.show()


def main():
    date = '2018-11-12'
    ActivityClassifier.initialize_processor()
    classifier = ActivityClassifier()
    classifier.load_activity_data(date)
    classifier.condition_data()
    classifier.model_setup()

    print('done')


if __name__ == '__main__':
    main()

