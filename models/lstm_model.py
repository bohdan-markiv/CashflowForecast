import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processing.data_processing import data_prep


class custom_LSTM:
    """
    Custom LSTM model for time series forecasting.

    Attributes:
        data (pd.DataFrame): Time series data for the model.
        name (str): Name of the company.
        type (str): Classification type.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        look_back (int): Number of previous time steps to use as input features.
        save (bool): Flag to save the plot or display it.
    """

    def __init__(self, data, name, type, epochs, batch_size, look_back, save):
        self.data = data
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.look_back = look_back
        self.type = type
        self.save = save

    def create_dataset(self, dataset):
        """
        Convert time series data into a dataset matrix.

        Args:
            dataset (np.ndarray): Scaled time series data.

        Returns:
            tuple: Arrays for input features (X) and target values (Y).
        """
        dataX, dataY = [], []
        for i in range(len(dataset)-self.look_back):
            a = dataset[i:(i+self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back, 0])
        return np.array(dataX), np.array(dataY)

    def initialize_model(self):
        """
        Initialize, train, and evaluate the LSTM model.
        """
        # Prepare the data
        self.df = self.data.values.astype('float32')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df = self.scaler.fit_transform(self.df)

        train_size = int(len(self.df) * 0.7)
        train, test = self.df[0:train_size,
                              :], self.df[train_size:len(self.df), :]

        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        # Reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=96, return_sequences=False,
                  activation='relu', input_shape=(1, self.look_back)))
        model.add(Dense(96, activation='relu'))
        model.add(Dropout(0))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

        # Train the model
        model.fit(trainX, trainY, epochs=self.epochs,
                  batch_size=self.batch_size, verbose=0)

        # Predict on train and test sets
        self.trainPredict = model.predict(trainX)
        self.testPredict = model.predict(testX)

        # Invert predictions to original scale
        self.trainPredict = self.scaler.inverse_transform(self.trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        self.testPredict = self.scaler.inverse_transform(self.testPredict)
        self.testY = self.scaler.inverse_transform([testY])

        # Calculate evaluation metrics
        self.mse = mean_squared_error(self.testY[0], self.testPredict[:, 0])
        self.rmse = np.sqrt(self.mse)
        self.mae = mean_absolute_error(self.testY[0], self.testPredict[:, 0])
        self.r2 = r2_score(self.testY[0], self.testPredict[:, 0])

    def create_plot(self):
        """
        Create and display/save a plot comparing actual and predicted values.
        """
        trainPredictPlot = np.empty_like(self.df)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[self.look_back:len(
            self.trainPredict)+self.look_back, :] = self.trainPredict

        testPredictPlot = np.empty_like(self.df)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(self.trainPredict) + (self.look_back*2)                        :len(self.df), :] = self.testPredict

        # Plot actual vs predicted values
        plt.figure(figsize=(12, 6))
        plt.plot(self.scaler.inverse_transform(
            self.df), label='Actual', marker='o')
        plt.plot(trainPredictPlot, label='Predicted', marker='x')
        plt.plot(testPredictPlot, linestyle='--',
                 label='Test Predictions', marker='x', color="red")
        plt.title('Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Net Cashflow')
        plt.legend()

        if self.save:
            MYDIR = f"graphs/lstm/{self.name}"
            os.makedirs(MYDIR, exist_ok=True)
            plt.savefig(f"{MYDIR}/{self.type}.png")
            plt.close()
        else:
            plt.show()


def create_lstms(data, epochs, batch_size, look_back):
    """
    Generate and evaluate LSTM models for multiple time series.

    Args:
        data (dict): Nested dictionary of company data with time series for different types.
        epochs (int): Number of training epochs for the LSTM model.
        batch_size (int): Batch size for training.
        look_back (int): Number of previous time steps to use as input features.
    """
    # Initialize output tables
    output_table_rmse = pd.DataFrame()
    output_table_mse = pd.DataFrame()
    output_table_mae = pd.DataFrame()
    output_table_r2 = pd.DataFrame()

    i = 1
    for company in data.keys():
        for type, df in data[company].items():
            if not isinstance(df, bool):
                print(f"Current LSTM progress - {i}")
                i += 1
                try:
                    # Process weekly data
                    if all(col in df.columns for col in ['week', 'year', 'amount']):
                        df = df["amount"]
                    elif 'effective_date' in df.columns and 'amount' in df.columns:
                        df.index = pd.to_datetime(df['effective_date'])
                        df = df["amount"]
                    else:
                        raise KeyError("Missing required columns.")

                    # Initialize and train LSTM model
                    my_lstm = custom_LSTM(
                        df, company, type, epochs, batch_size, look_back, save=True)
                    my_lstm.initialize_model()
                    my_lstm.create_plot()

                    # Save metrics
                    for table, metric in zip([output_table_rmse, output_table_mse, output_table_mae, output_table_r2],
                                             [my_lstm.rmse, my_lstm.mse, my_lstm.mae, my_lstm.r2]):
                        if company in table.index:
                            table.at[company, type] = metric
                        else:
                            table.loc[company, type] = metric

                except Exception as e:
                    print(f"LSTM failed for {company} - {type}: {e}")

    # Save results to Excel
    output_table_rmse.to_excel("output_tables/lstm/rmse.xlsx")
    output_table_mse.to_excel("output_tables/lstm/mse.xlsx")
    output_table_mae.to_excel("output_tables/lstm/mae.xlsx")
    output_table_r2.to_excel("output_tables/lstm/r2.xlsx")


if __name__ == "__main__":
    """
    Main execution block to create and evaluate LSTM models.
    """
    data = data_prep()
    create_lstms(data, epochs=50, batch_size=32, look_back=10)
