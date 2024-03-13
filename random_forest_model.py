import pandas as pd
import os
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from math import sqrt
from data_processing import data_prep
"""
role = sagemaker.get_execution_role()
bucket='bohdan-example-data-sagemaker'
data_key = 'monthly-beer-production-in-austr.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)
"""


class custom_RandomForest:
    def __init__(self, df, name, type, n_lags, n_estimators=1500, save=False):
        self.df = df
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.name = name
        self.type = type
        self.save = save

    def _series_to_supervised(self, data, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(self.n_lags, 0, -1):
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

    def _train_test_split(data, n_test):
        return data[:-n_test, :], data[-n_test:, :]

    def _walk_forward_validation(self, data, n_test):
        predictions = list()
        # split dataset
        train, test = self._train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # split test row into input and output columns
            testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
            yhat = self._random_forest_forecast(history, testX)
        # store forecast in list of predictions
            predictions.append(yhat)
        # add actual observation to history for the next loop
            history.append(test[i])
        # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
        # estimate prediction error
        error = mean_absolute_error(test[:, -1], predictions)
        return error, test[:, 1], predictions

    def _random_forest_forecast(train, testX):
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

    def initialize_model(self):
        # training
        """
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
        """
        self.values = self.df.values

        size = int(len(self.values) * 0.7)
        train_data, test_data = self.values[0:
                                            size], self.values[size-6:len(self.values)]
        test = self._series_to_supervised(test_data)
        # full data
        train = self._series_to_supervised(train_data)
        # split into input and output columns
        testX, testy = test[:, :-1], test[:, -1]
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        model = RandomForestRegressor(n_estimators=2000)
        model.fit(trainX, trainy)
        # construct an input for a new prediction
        row = self.values[-6:].flatten()
        actual_raw = self._series_to_supervised(self.values)[:, -1]
        # Actual values for the training period
        self.actual = self.df.values[-len(actual_raw):]
        self.predictions_train = []  # To store predictions

        # Loop through the training set to make predictions one step ahead each time
        for i in range(len(trainy)):
            # Prepare the input data
            row = trainX[i].reshape((1, len(trainX[i])))
            # Make a prediction
            yhat = model.predict(row)
            # Store the prediction
            self.predictions_train.append(yhat[0])

        self.predictions_test = []  # To store predictions

        # Loop through the training set to make predictions one step ahead each time
        for i in range(len(testy)):
            # Prepare the input data
            row = testX[i].reshape((1, len(testX[i])))
            # Make a prediction
            yhat = model.predict(row)
            # Store the prediction
            self.predictions_test.append(yhat[0])

        # Convert predictions list to an array for easier handling
        self.predictions_train = asarray(self.predictions_train)
        self.predictions_test = asarray(self.predictions_test)
        self.mse = mean_squared_error(testy, self.predictions_test)
        self.rmse = sqrt(self.mse)
        self.mae = mean_absolute_error(testy, self.predictions_test)

    def create_plot(self):
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(self.actual, label='Actual', marker='o')
        plt.plot(self.predictions_train, label='Predicted', marker='x')
        plt.plot(range(len(self.predictions_train), len(self.predictions_train) + len(self.predictions_test)),
                 self.predictions_test, color='red', linestyle='--', label='Test Predictions', marker='x')
        plt.title('Actual vs Predicted Births')
        plt.xlabel('Time')
        plt.ylabel('Births')
        plt.legend()
        if self.save:
            MYDIR = (f"graphs/random_forest/{self.name}")
            CHECK_FOLDER = os.path.isdir(MYDIR)

            # If folder doesn't exist, then create it.
            if not CHECK_FOLDER:
                os.makedirs(MYDIR)
            plt.savefig(f"graphs/random_forest/{self.name}/{self.type}")
            plt.close()
        else:
            plt.show()


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


def create_random_forests(data, n_estimators, n_lags):
    i = 0
    for company in data.keys():

        for type, df in data[company].items():
            if not isinstance(df, bool):
                print(i)
                i += 1
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
                my_random_forest = custom_RandomForest(df, company, type, n_lags=n_lags,
                                                       n_estimators=n_estimators, save=True)

                my_random_forest.initialize_model()
                my_random_forest.create_plot()
                if company in output_table_rmse.index:
                    output_table_rmse.at[company, type] = my_random_forest.rmse
                else:
                    output_table_rmse.loc[company,
                                          type] = my_random_forest.rmse
                if company in output_table_mse.index:
                    output_table_mse.at[company, type] = my_random_forest.mse
                else:
                    output_table_mse.loc[company, type] = my_random_forest.mse
                if company in output_table_mae.index:
                    output_table_mae.at[company, type] = my_random_forest.mae
                else:
                    output_table_mae.loc[company, type] = my_random_forest.mae
    output_table_rmse.to_excel("output_tables/random_forest/rmse.xlsx")
    output_table_mse.to_excel("output_tables/random_forest/mse.xlsx")
    output_table_mae.to_excel("output_tables/random_forest/mae.xlsx")


weekly = True
data = data_prep(weekly=weekly)

"""
role = sagemaker.get_execution_role()
bucket='bohdan-example-data-sagemaker'
data_key = 'monthly-beer-production-in-austr.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)
"""
create_random_forests(data, 2000, 6)


weekly = True
df = data_prep(weekly=weekly)
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
