import boto3
import pandas as pd
import numpy as np
import sagemaker
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from math import sqrt
"""
role = sagemaker.get_execution_role()
bucket='bohdan-example-data-sagemaker'
data_key = 'monthly-beer-production-in-austr.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)
"""
df = pd.read_csv(data_location, header=0, index_col=0)

df.plot()
plt.show()
df
# transform a time series dataset into a supervised learning dataset


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
    # fit model on history and make a prediction
        yhat = random_forest_forecast(history, testX)
    # store forecast in list of predictions
        predictions.append(yhat)
    # add actual observation to history for the next loop
        history.append(test[i])
    # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, 1], predictions


def random_forest_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=1500, max_features="sqrt")
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


# training
values = df.values
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=6)
# evaluate
mae, y, yhat = walk_forward_validation(data, 15)
print('MAE: %.3f' % mae)
# plot expected vs predicted
plt.plot(y, label='Expected')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.show()

# full data
train = series_to_supervised(values, n_in=6)
# split into input and output columns
trainX, trainy = train[:, :-1], train[:, -1]
# fit model
model = RandomForestRegressor(n_estimators=1000)
model.fit(trainX, trainy)
# construct an input for a new prediction
row = values[-6:].flatten()

actual = df.values[-len(trainy):]  # Actual values for the training period
predictions = []  # To store predictions

# Loop through the training set to make predictions one step ahead each time
for i in range(len(trainy)):
    # Prepare the input data
    row = trainX[i].reshape((1, len(trainX[i])))
    # Make a prediction
    yhat = model.predict(row)
    # Store the prediction
    predictions.append(yhat[0])

# Convert predictions list to an array for easier handling
predictions = asarray(predictions)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual', marker='o')
plt.plot(predictions, label='Predicted', marker='x')
plt.title('Actual vs Predicted Births')
plt.xlabel('Time')
plt.ylabel('Births')
plt.legend()
plt.show()
rmse = sqrt(mean_squared_error(actual, predictions))
print('RMSE: %.3f' % rmse)
