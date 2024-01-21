import pandas as pd


def prepare_data():

    data = pd.read_excel("example_data/transaction_data_sample.xlsx")
    output = {}
    for type in data["Transaction_Type"]:
        output[type] = data[data["Transaction_Type"] == type]

    return output
