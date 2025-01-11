from data_processing.data_processing import data_prep
from models.arima_model import custom_ARIMA, create_arimas
from models.random_forest_model import custom_RandomForest, create_random_forests
from models.lstm_model import custom_LSTM, create_lstms


def sequential_initializer(data):
    """
    Sequentially initializes and runs ARIMA, Random Forest, and LSTM models 
    on the provided dataset.

    Args:
        data (dict): Preprocessed dataset structured by company and classification.

    """
    create_arimas(data)
    create_random_forests(data, n_estimators=1500, n_lags=10)
    create_lstms(data, epochs=500, batch_size=16, look_back=10)


if __name__ == '__main__':
    """
    Main execution block for preprocessing data and running model pipelines.
    """
    weekly = False  # Set to True for weekly aggregation, False otherwise
    df = data_prep(weekly=weekly)  # Preprocess data
    sequential_initializer(df)  # Run model pipelines
