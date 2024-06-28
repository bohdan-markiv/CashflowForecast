from data_processing import data_prep
from arima_model import custom_ARIMA, create_arimas
from random_forest_model import custom_RandomForest, create_random_forests
from lstm_model import custom_LSTM, create_lstms
import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 54468226 -COGS
weekly = False
data = data_prep(weekly=weekly)
origin = data[5616614]["Operational Revenue"]
df = origin.copy()
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

my_arima = custom_ARIMA(data=df, name=54468226, type="COGS")
my_arima.initialize_model()
len(my_arima.test_predictions)
my_rf = custom_RandomForest(df, 54468226, "COGS", n_lags=10)
my_rf.initialize_model()
my_rf.predictions_test
len(my_rf.predictions_test)
my_lstm = custom_LSTM(df, 54468226, "COGS", epochs=200,
                      batch_size=8, look_back=10, save=False)
my_lstm.initialize_model()


my_rf.r2
r2_score(my_arima.test, my_arima.test_predictions)
r2_score(my_lstm.testY[0], my_lstm.testPredict[:, 0])
print(
    f"ARIMA {-1 * r2_score(my_arima.test, my_arima.test_predictions)}, LSTM {-1 * r2_score(my_lstm.testY[0], my_lstm.testPredict[:, 0])}")

data_block = origin.reset_index().drop(
    columns=["index", "Unnamed: 0"])
size = int(0.7 * len(data_block))
data_block_test = data_block[size:]
data_block_test["month"] = data_block_test["effective_date"].apply(
    lambda x: x.month)
data_block_test["arima_pred"] = my_arima.test_predictions
data_block_test["rf_pred"] = my_rf.predictions_test
data_block_test["lstm_pred"] = [None]*10 + \
    [float(x) for x in my_lstm.testPredict]
agg_data = data_block_test.groupby(
    "month")[["amount", "arima_pred", "rf_pred", "lstm_pred"]].sum()

for col in agg_data:
    if col != "amount":
        agg_data[f"diff_{col}"] = agg_data[col]-agg_data["amount"]

agg_data = agg_data.loc[5:]

agg_data = agg_data.reset_index()
conditions = [
    agg_data['month'].isin([1, 2, 3]),
    agg_data['month'].isin([4, 5, 6]),
    agg_data['month'].isin([7, 8, 9]),
    agg_data['month'].isin([10, 11, 12])
]
choices = [1, 2, 3, 4]

# Use numpy.select to assign quarters
agg_data['quarter'] = np.select(conditions, choices, default=None)
agg_data.columns
agg_agg_data = agg_data.groupby("quarter")[['amount', 'arima_pred', 'rf_pred', 'lstm_pred',
                                           'diff_arima_pred', 'diff_rf_pred', 'diff_lstm_pred', 'quarter']].sum()
# Super aggregated difference (quarter)
pyplot.figure(figsize=[15, 5])
pyplot.plot(agg_agg_data["diff_arima_pred"], label='ARIMA diff')
pyplot.plot(agg_agg_data["diff_rf_pred"], label='RF diff')
pyplot.plot(agg_agg_data["diff_lstm_pred"], label='LSTM diff')
pyplot.axhline(0, color='red', linewidth=1)
pyplot.legend()
pyplot.show()

# Super aggregated predictions (quarter)
pyplot.figure(figsize=[15, 5])
pyplot.plot(agg_agg_data["amount"], label='Actual')
pyplot.plot(agg_agg_data["arima_pred"], label='ARIMA pred')
pyplot.plot(agg_agg_data["rf_pred"], label='RF pred')
pyplot.plot(agg_agg_data["lstm_pred"], label='LSTM pred')
pyplot.legend()
pyplot.show()

# Aggregated predictions (month)
pyplot.figure(figsize=[15, 5])
pyplot.plot(agg_data["amount"], label='Actual')
pyplot.plot(agg_data["arima_pred"], label='ARIMA pred')
pyplot.plot(agg_data["rf_pred"], label='RF pred')
pyplot.plot(agg_data["lstm_pred"], label='LSTM pred')
pyplot.legend()
pyplot.show()

# Aggregated diff (month)
pyplot.figure(figsize=[15, 5])
pyplot.plot(agg_data["diff_arima_pred"], label='ARIMA pred')
pyplot.plot(agg_data["diff_rf_pred"], label='RF pred')
pyplot.plot(agg_data["diff_lstm_pred"], label='LSTM pred')
pyplot.axhline(0, color='red', linewidth=1)
pyplot.legend()
pyplot.show()
