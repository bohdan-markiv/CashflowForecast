from data_processing import data_prep
from arima_model import custom_ARIMA, create_arimas
from random_forest_model import custom_RandomForest, create_random_forests
from lstm_model import custom_LSTM, create_lstms


def sequential_intitializer(data):
    create_arimas(data)
    create_random_forests(data, 1500, 10)
    create_lstms(df, epochs=500,
                 batch_size=16, look_back=10)


if __name__ == '__main__':
    weekly = False
    df = data_prep(weekly=weekly)
    sequential_intitializer(df)
