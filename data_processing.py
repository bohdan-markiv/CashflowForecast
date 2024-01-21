import pandas as pd


def prepare_data():
    data = pd.read_excel("example_data/transaction_data_sample.xlsx")
    return data
