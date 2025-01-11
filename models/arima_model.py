import pandas as pd
import warnings
import os
import pmdarima as pm
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processing.data_processing import data_prep

# Suppress convergence warnings
warnings.filterwarnings(
    "ignore", message="Maximum Likelihood optimization failed to converge.*")


class custom_ARIMA:
    """
    A class to create and evaluate ARIMA models for time series data.

    Attributes:
        data (pd.Series): Time series data.
        name (str): Name of the entity (e.g., company).
        type (str): Type of data (e.g., revenue type).
    """

    def __init__(self, data, name, type):
        self.data = data
        self.test = ""  # Placeholder for test data
        self.name = name
        self.type = type

    def initialize_model(self):
        """
        Splits data into training and test sets, fits an ARIMA model, and calculates evaluation metrics.
        """
        self.X = self.data
        self.size = int(len(self.X) * 0.7)  # 70% train, 30% test split
        self.train, self.test = self.X[0:self.size], self.X[self.size:len(
            self.X)]

        try:
            # Automatically determine ARIMA parameters using auto_arima
            order = pm.auto_arima(self.train,
                                  start_p=0, start_q=0,
                                  max_p=12, max_q=12,
                                  test='adf',
                                  d=None, D=None,
                                  trace=False,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True).order

            # Fit ARIMA model
            model = ARIMA(self.train, order=order)
            self.model_fit = model.fit()

            # Predict on test set
            self.test_predictions = self.model_fit.predict(
                start=self.size, end=len(self.X)-1, dynamic=True)

        except Exception as e:
            print(f"ARIMA model failed for: {self.name} - {self.type}")
            raise e

        # Calculate metrics
        self.mse = mean_squared_error(self.test, self.test_predictions)
        self.rmse = sqrt(self.mse)
        self.mae = mean_absolute_error(self.test, self.test_predictions)
        self.r2 = r2_score(self.test, self.test_predictions)

    def get_rmse(self):
        """
        Print the RMSE of the model if it has been initialized.
        """
        if isinstance(self.test, str):
            print("Model not initialized. Call initialize_model() first.")
        else:
            print(f'Test RMSE: {self.rmse:.3f}')

    def create_the_full_graph(self, save=False):
        """
        Generate and display or save a graph of the actual vs predicted values.

        Args:
            save (bool): If True, save the graph to a file.
        """
        if isinstance(self.test, str):
            print("Model not initialized. Call initialize_model() first.")
        else:
            fig, ax = pyplot.subplots()
            ax = self.X.plot(ax=ax)
            plot_predict(self.model_fit, start=len(self.train),
                         end=len(self.data)-1, ax=ax, dynamic=False)

            # Customize legend
            ax.legend()
            if save:
                # Save graph to folder
                folder_path = f"graphs/arima/{self.name}"
                os.makedirs(folder_path, exist_ok=True)
                pyplot.savefig(f"{folder_path}/{self.type}.png")
            else:
                pyplot.show()
            pyplot.close()


def create_arimas(data):
    """
    Generate ARIMA models for multiple time series and save evaluation metrics.

    Args:
        data (dict): Nested dictionary of company data with time series for different types.
    """
    # Initialize output tables
    output_table_rmse = pd.DataFrame()
    output_table_mse = pd.DataFrame()
    output_table_mae = pd.DataFrame()
    output_table_r2 = pd.DataFrame()

    progress_counter = 1
    for company in data.keys():
        for type, df in data[company].items():
            if not isinstance(df, bool):
                print(f"Current ARIMA progress - {progress_counter}")
                progress_counter += 1

                try:
                    # Process weekly data
                    if all(col in df.columns for col in ['week', 'year', 'amount']):
                        df = df.sort_values(
                            ['year', 'week']).reset_index(drop=True)
                        df = df.drop(columns=['week', 'year'])
                    # Process daily data
                    elif 'effective_date' in df.columns and 'amount' in df.columns:
                        df.index = pd.to_datetime(df['effective_date'])
                        df = df.drop(columns=['effective_date'])
                    else:
                        raise KeyError("Missing required columns.")

                    # Create ARIMA model
                    my_arima = custom_ARIMA(df, company, type)
                    my_arima.initialize_model()
                    my_arima.create_the_full_graph(save=True)

                    # Save metrics
                    for table, metric in zip([output_table_rmse, output_table_mse, output_table_mae, output_table_r2],
                                             [my_arima.rmse, my_arima.mse, my_arima.mae, my_arima.r2]):
                        table.at[company, type] = metric

                except Exception as e:
                    print(f"ARIMA failed for {company} - {type}: {e}")
                    continue

    # Save results to Excel
    output_table_rmse.to_excel("output_tables/arima/rmse.xlsx")
    output_table_mse.to_excel("output_tables/arima/mse.xlsx")
    output_table_mae.to_excel("output_tables/arima/mae.xlsx")
    output_table_r2.to_excel("output_tables/arima/r2.xlsx")
