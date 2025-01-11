import pandas as pd
import os
from numpy import asarray
from pandas import DataFrame, concat
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from math import sqrt
from data_processing.data_processing import data_prep


class custom_RandomForest:
    """
    Custom Random Forest model for time series forecasting.

    Attributes:
        df (pd.DataFrame): Time series data for the model.
        name (str): Name of the company.
        type (str): Classification type.
        n_lags (int): Number of previous time steps to use as input features.
        n_estimators (int): Number of trees in the Random Forest model.
        save (bool): Flag to save the plot or display it.
    """

    def __init__(self, df, name, type, n_lags, n_estimators=1500, save=False):
        self.df = df
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.name = name
        self.type = type
        self.save = save

    def _series_to_supervised(self, data, n_out=1, dropnan=True):
        """
        Transform time series data into a supervised learning dataset.

        Args:
            data (array-like): Time series data.
            n_out (int): Number of output steps.
            dropnan (bool): Whether to drop rows with NaN values.

        Returns:
            np.ndarray: Transformed dataset.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols = []
        for i in range(self.n_lags, 0, -1):
            cols.append(df.shift(i))
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        agg = concat(cols, axis=1)
        if dropnan:
            agg.dropna(inplace=True)
        return agg.values

    def _train_test_split(self, data, n_test):
        """
        Split data into training and testing datasets.

        Args:
            data (np.ndarray): Time series dataset.
            n_test (int): Number of test samples.

        Returns:
            tuple: Training and testing datasets.
        """
        return data[:-n_test, :], data[-n_test:, :]

    def _random_forest_forecast(self, train, testX):
        """
        Fit Random Forest model and make a one-step forecast.

        Args:
            train (np.ndarray): Training dataset.
            testX (np.ndarray): Test input features.

        Returns:
            float: Predicted value.
        """
        train = asarray(train)
        trainX, trainy = train[:, :-1], train[:, -1]
        model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_features="sqrt")
        model.fit(trainX, trainy)
        yhat = model.predict([testX])
        return yhat[0]

    def initialize_model(self):
        """
        Initialize, train, and evaluate the Random Forest model.
        """
        self.values = self.df.values

        size = int(len(self.values) * 0.7)
        train_data, test_data = self.values[:
                                            size], self.values[size-self.n_lags:]
        train = self._series_to_supervised(train_data)
        test = self._series_to_supervised(test_data)

        trainX, trainy = train[:, :-1], train[:, -1]
        testX, testy = test[:, :-1], test[:, -1]

        model = RandomForestRegressor(n_estimators=self.n_estimators)
        model.fit(trainX, trainy)

        self.predictions_train = model.predict(trainX)
        self.predictions_test = model.predict(testX)

        self.mse = mean_squared_error(testy, self.predictions_test)
        self.rmse = sqrt(self.mse)
        self.mae = mean_absolute_error(testy, self.predictions_test)
        self.r2 = r2_score(testy, self.predictions_test)

    def create_plot(self):
        """
        Create and display/save a plot comparing actual and predicted values.
        """
        actual_values = self.df.values

        plt.figure(figsize=(12, 6))
        plt.plot(actual_values, label='Actual', marker='o')
        plt.plot(range(len(self.predictions_train)),
                 self.predictions_train, label='Train Predictions', marker='x')
        plt.plot(range(len(self.predictions_train), len(self.predictions_train) + len(self.predictions_test)),
                 self.predictions_test, color='red', linestyle='--', label='Test Predictions', marker='x')
        plt.title('Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Net Cashflow')
        plt.legend()

        if self.save:
            output_dir = f"graphs/random_forest/{self.name}"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/{self.type}.png")
            plt.close()
        else:
            plt.show()


def create_random_forests(data, n_estimators, n_lags):
    """
    Generate and evaluate Random Forest models for multiple time series.

    Args:
        data (dict): Nested dictionary of company data with time series for different types.
        n_estimators (int): Number of trees in the Random Forest model.
        n_lags (int): Number of previous time steps to use as input features.
    """
    output_table_rmse = pd.DataFrame()
    output_table_mse = pd.DataFrame()
    output_table_mae = pd.DataFrame()
    output_table_r2 = pd.DataFrame()

    i = 1
    for company in data.keys():
        for type, df in data[company].items():
            if not isinstance(df, bool):
                print(f"Current Random Forest progress - {i}")
                i += 1
                try:
                    if all(col in df.columns for col in ['week', 'year', 'amount']):
                        df = df[['amount']]
                    elif 'effective_date' in df.columns and 'amount' in df.columns:
                        df.index = pd.to_datetime(df['effective_date'])
                        df = df[['amount']]
                    else:
                        raise KeyError("Missing required columns.")

                    my_random_forest = custom_RandomForest(df, company, type, n_lags=n_lags,
                                                           n_estimators=n_estimators, save=True)

                    my_random_forest.initialize_model()
                    my_random_forest.create_plot()

                    for table, metric in zip([output_table_rmse, output_table_mse, output_table_mae, output_table_r2],
                                             [my_random_forest.rmse, my_random_forest.mse, my_random_forest.mae, my_random_forest.r2]):
                        if company in table.index:
                            table.at[company, type] = metric
                        else:
                            table.loc[company, type] = metric

                except Exception as e:
                    print(f"Random Forest failed for {company} - {type}: {e}")

    output_table_rmse.to_excel("output_tables/random_forest/rmse.xlsx")
    output_table_mse.to_excel("output_tables/random_forest/mse.xlsx")
    output_table_mae.to_excel("output_tables/random_forest/mae.xlsx")
    output_table_r2.to_excel("output_tables/random_forest/r2.xlsx")


if __name__ == "__main__":
    """
    Main execution block to create and evaluate Random Forest models.
    """
    data = data_prep()
    create_random_forests(data, n_estimators=1500, n_lags=6)
