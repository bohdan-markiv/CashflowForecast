import pandas as pd
import warnings
import os
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_processing import data_prep
warnings.filterwarnings(
    "ignore", message="Maximum Likelihood optimization failed to converge.*")


class custom_ARIMA:
    def __init__(self, p, d, q, data, name, type):
        self.p = p
        self.d = d
        self.q = q
        self.data = data
        self.test = ""
        self.name = name
        self.type = type

    def check_residuals(self):

        model = ARIMA(self.data, order=(self.p, self.d, self.q))
        model_fit = model.fit()
        model_fit.summary()
        residuals = pd.DataFrame(model_fit.resid)

        residuals.plot()
        pyplot.show()

        residuals.plot(kind='kde')
        pyplot.show()

    def initialize_model(self):
        self.X = self.data.values
        size = int(len(self.X) * 0.7)
        self.train, self.test = self.X[0:size], self.X[size:len(self.X)]
        history_train = [x for x in self.train]
        history_test = [x for x in self.train]
        self.train_predictions = []
        self.test_predictions = []
        try:
            for t in range(len(self.train)):
                model = ARIMA(history_train, order=(self.p, self.d, self.q))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                self.train_predictions.append(yhat)
                obs = self.train[t]
                history_train.append(obs)  # only train data used here

            # Add history of the entire training data for predicting the test
            for t in range(len(self.test)):
                model = ARIMA(history_test, order=(self.p, self.d, self.q))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                self.test_predictions.append(yhat)
                obs = self.test[t]
                history_test.append(obs)
        except Exception as e:
            print(
                f"for this company operation was not successfull - {self.name}")
            raise Exception

        self.mse = mean_squared_error(
            self.test, self.test_predictions)
        self.rmse = sqrt(self.mse)
        self.mae = mean_absolute_error(self.test, self.test_predictions)

    def get_rmse(self):
        if isinstance(self.test, str):
            print("Do the initialisation first.")
        else:
            print('Test RMSE: %.3f' % self.rmse)

    def create_the_full_graph(self, save=False):

        if isinstance(self.test, str):
            print("Do the initialisation first.")

        else:
            pyplot.figure(figsize=[15, 5])
            pyplot.plot(self.X, label='Actual')
            pyplot.plot(self.train_predictions, color='orange',
                        linestyle='--', label='Train Predictions')
            pyplot.plot(range(len(self.train), len(self.train) + len(self.test_predictions)),
                        self.test_predictions, color='red', linestyle='--', label='Test Predictions')
            pyplot.legend()
            if save:
                MYDIR = (f"graphs/arima/{self.name}")
                CHECK_FOLDER = os.path.isdir(MYDIR)

                # If folder doesn't exist, then create it.
                if not CHECK_FOLDER:
                    os.makedirs(MYDIR)
                pyplot.savefig(f"graphs/arima/{self.name}/{self.type}")
            else:
                pyplot.show()
            pyplot.close()


def create_arimas(data, p, d, q):
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
                print(f"Current Arima progress - {i}")
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
                my_arima = custom_ARIMA(p, d, q, df, company, type)
                try:
                    my_arima.initialize_model()
                    my_arima.create_the_full_graph(save=True)
                    if company in output_table_rmse.index:
                        output_table_rmse.at[company, type] = my_arima.rmse
                    else:
                        output_table_rmse.loc[company, type] = my_arima.rmse
                    if company in output_table_mse.index:
                        output_table_mse.at[company, type] = my_arima.mse
                    else:
                        output_table_mse.loc[company, type] = my_arima.mse
                    if company in output_table_mae.index:
                        output_table_mae.at[company, type] = my_arima.mae
                    else:
                        output_table_mae.loc[company, type] = my_arima.mae
                except Exception:
                    continue

    output_table_rmse.to_excel("output_tables/arima/rmse.xlsx")
    output_table_mse.to_excel("output_tables/arima/mse.xlsx")
    output_table_mae.to_excel("output_tables/arima/mae.xlsx")


"""
role = sagemaker.get_execution_role()
bucket='bohdan-example-data-sagemaker'
data_key = 'monthly-beer-production-in-austr.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)
"""
# create_arimas(data, 3, 2, 4)
"""
weekly = True
data = data_prep(weekly=weekly)
data = data[54468226]["Operational Revenue"]
try:
    # Check if 'week', 'year', and 'amount' are in the columns
    if all(col in data.columns for col in ['week', 'year', 'amount']):
        data = data[["week", "year", "amount"]]
        data = data.sort_values(['year', 'week'], ascending=[
                                True, True]).reset_index(drop=True)
        data = data.drop(columns=["week", "year"])
    else:
        raise KeyError("One of 'week', 'year', 'amount' is not in DataFrame")
except KeyError as e:  # Catching if columns are missing
    print(f"Missing columns for weekly data processing: {e}")
    try:
        # Attempt processing assuming the structure is for 'effective_date'
        if "effective_date" in data.columns and "amount" in data.columns:
            data = data[["effective_date", "amount"]]
            # Ensure index is datetime for time series
            data.index = pd.to_datetime(data["effective_date"])
            data = data.drop(columns=["effective_date"])
        else:
            raise KeyError(
                "One of 'effective_date', 'amount' is not in DataFrame")
    except KeyError as error:
        print(f"Missing columns for daily data processing: {error}")


data.plot()
pyplot.show()

plot_pacf(data, lags=20)  # Change lags as needed
pyplot.title('Partial Autocorrelation Function')
pyplot.show()

# Draw the ACF
plot_acf(data, lags=20)  # Change lags as needed
pyplot.title('Autocorrelation Function')
pyplot.show()

# Stationarity check
result = adfuller(data)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


my_arima = custom_ARIMA(3, 2, 4, data, 54468226, "Operational Revenue")

my_arima.check_residuals()
my_arima.initialize_model()
my_arima.get_rmse()
my_arima.create_the_full_graph(save=True)
"""
