from data_processing import data_prep
from arima_model import custom_ARIMA, create_arimas
from random_forest_model import custom_RandomForest, create_random_forests
from lstm_model import custom_LSTM, create_lstms


def sequential_intitializer(data):
    create_arimas(data, 2, 2, 1)
    create_random_forests(data, 3000, 10)
    create_lstms(data, 1200, 7, 4)


if __name__ == '__main__':
    weekly = True
    df = data_prep(weekly=weekly)
    sequential_intitializer(df)
