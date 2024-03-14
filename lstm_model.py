import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_processing import data_prep


class custom_LSTM():
    def __init__(self, data, name, type, epochs, batch_size, look_back, save):
        self.data = data
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.look_back = look_back
        self.type = type
        self.save = save

    def create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset)-self.look_back-1):
            a = dataset[i:(i+self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back, 0])
        return np.array(dataX), np.array(dataY)

    def initialize_model(self):
        self.df = self.data.values.astype('float32')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df = self.scaler.fit_transform(self.df)
        train_size = int(len(self.df) * 0.7)
        # test_size = len(df) - train_size
        train, test = self.df[0:train_size,
                              :], self.df[train_size:len(self.df), :]

        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        model = Sequential()
        model.add(LSTM(4, input_shape=(1, self.look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=self.epochs,
                  batch_size=self.batch_size, verbose=0)

        self.trainPredict = model.predict(trainX)
        self.testPredict = model.predict(testX)
        # invert predictions
        self.trainPredict = self.scaler.inverse_transform(self.trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        self.testPredict = self.scaler.inverse_transform(self.testPredict)
        testY = self.scaler.inverse_transform([testY])
# calculate root mean squared error
        self.mse = mean_squared_error(testY[0], self.testPredict[:, 0])
        self.rmse = np.sqrt(self.mse)
        self.mae = mean_absolute_error(testY[0], self.testPredict[:, 0])

    def create_plot(self):

        trainPredictPlot = np.empty_like(self.df)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[self.look_back:len(
            self.trainPredict)+self.look_back, :] = self.trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(self.df)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(self.trainPredict) +
                        (self.look_back*2)+1:len(self.df)-1, :] = self.testPredict
        # plot baseline and predictions
        plt.plot(self.scaler.inverse_transform(self.df))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        if self.save:
            MYDIR = (f"graphs/lstm/{self.name}")
            CHECK_FOLDER = os.path.isdir(MYDIR)

            # If folder doesn't exist, then create it.
            if not CHECK_FOLDER:
                os.makedirs(MYDIR)
            plt.savefig(f"graphs/lstm/{self.name}/{self.type}")
            plt.close()
        else:
            plt.show()


def create_lstms(data, epochs, batch_size, look_back):
    output_table_rmse = pd.DataFrame({
        'COGS': [],
        'Operational Revenue': [],
        'Other Revenue': [],
        'Other Operating Costs': [],
        'Organizational Costs': []
    })
    output_table_mse = pd.DataFrame({
        'COGS': [],
        'Operational Revenue': [],
        'Other Revenue': [],
        'Other Operating Costs': [],
        'Organizational Costs': []
    })
    output_table_mae = pd.DataFrame({
        'COGS': [],
        'Operational Revenue': [],
        'Other Revenue': [],
        'Other Operating Costs': [],
        'Organizational Costs': []
    })

    i = 1
    for company in data.keys():
        for type, df in data[company].items():
            if not isinstance(df, bool):
                print(i)
                print(f"Current LSTM progress - {i}")
                try:
                    # Check if 'week', 'year', and 'amount' are in the columns
                    if all(col in df.columns for col in ['week', 'year', 'amount']):
                        df = df[["week", "year", "amount"]]
                        df = df.sort_values(['year', 'week'], ascending=[
                            True, True]).reset_index(drop=True)
                        df = df.drop(columns=["week", "year"])
                    else:
                        raise KeyError(
                            "One of 'week', 'year', 'amount' is not in DataFrame")
                except KeyError as e:  # Catching if columns are missing
                    print(f"Missing columns for weekly data processing: {e}")
                    try:
                        # Attempt processing assuming the structure is for 'effective_date'
                        if "effective_date" in df.columns and "amount" in df.columns:
                            df = df[["effective_date", "amount"]]
                            # Ensure index is datetime for time series
                            df.index = pd.to_datetime(df["effective_date"])
                            df = df.drop(columns=["effective_date"])
                        else:
                            raise KeyError(
                                "One of 'effective_date', 'amount' is not in DataFrame")
                    except KeyError as error:
                        print(
                            f"Missing columns for daily data processing: {error}")
                my_lstm = custom_LSTM(df, company, type, epochs, batch_size,
                                      look_back, save=True)

                my_lstm.initialize_model()
                my_lstm.create_plot()
                if company in output_table_rmse.index:
                    output_table_rmse.at[company, type] = my_lstm.rmse
                else:
                    output_table_rmse.loc[company,
                                          type] = my_lstm.rmse
                if company in output_table_mse.index:
                    output_table_mse.at[company, type] = my_lstm.mse
                else:
                    output_table_mse.loc[company, type] = my_lstm.mse
                if company in output_table_mae.index:
                    output_table_mae.at[company, type] = my_lstm.mae
                else:
                    output_table_mae.loc[company, type] = my_lstm.mae
    output_table_rmse.to_excel("output_tables/lstm/rmse.xlsx")
    output_table_mse.to_excel("output_tables/lstm/mse.xlsx")
    output_table_mae.to_excel("output_tables/lstm/mae.xlsx")


# create_lstms(df, 1200, 7, 4)
"""
weekly = True
df = data_prep(weekly=weekly)
# One model instance
df = df[54468226]["Operational Revenue"]
try:
    # Check if 'week', 'year', and 'amount' are in the columns
    if all(col in df.columns for col in ['week', 'year', 'amount']):
        df = df[["week", "year", "amount"]]
        df = df.sort_values(['year', 'week'], ascending=[
            True, True]).reset_index(drop=True)
        df = df.drop(columns=["week", "year"])
    else:
        raise KeyError("One of 'week', 'year', 'amount' is not in DataFrame")
except KeyError as e:  # Catching if columns are missing
    print(f"Missing columns for weekly data processing: {e}")
    try:
        # Attempt processing assuming the structure is for 'effective_date'
        if "effective_date" in df.columns and "amount" in df.columns:
            df = df[["effective_date", "amount"]]
            # Ensure index is datetime for time series
            df.index = pd.to_datetime(df["effective_date"])
            df = df.drop(columns=["effective_date"])
        else:
            raise KeyError(
                "One of 'effective_date', 'amount' is not in DataFrame")
    except KeyError as error:
        print(f"Missing columns for daily data processing: {error}")
df.plot()
plt.show()


df = df.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
train_size = int(len(df) * 0.7)
test_size = len(df) - train_size
train, test = df[0:train_size, :], df[train_size:len(df), :]
print(len(train), len(test))


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1200, batch_size=7, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print(f"train score - {trainScore}")
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print(f"test score - {testScore}")


trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
"""
