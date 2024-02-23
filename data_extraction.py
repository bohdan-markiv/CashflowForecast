import pandas as pd
import numpy as np


data = pd.read_csv("C:/Users/b.markiv/Downloads/full_data_set.csv")
revenues = pd.read_csv("revenues.csv")

data = data[data["company_id"].isin([27934117, 262866, 24872290, 18165120, 22205603, 11036957, 2199709,
                                     8922050, 18305616, 2807402, 9646170, 11745239, 23673416,
                                     16075016, 10374661, 18117434, 30879313, 26390251, 5645364,
                                     27233208, 9296446, 397293, 16862710, 10209459,
                                     15878443, 1559829, 1628862])]
revenues = revenues[revenues["Division"].isin([2163245, 999296, 2659781, 556104, 236141, 1555285, 2403092, 925373,
                                               1426111, 2850605, 1122522, 1385214, 3138419, 3128238, 3383896,
                                               2111655, 2781225, 375035, 365180, 3358248,
                                               3646589, 2667341, 2391738, 1186317, 3469108, 2377423, 250980])]
revenues.to_csv("filtered_revenue.csv")


def anonymyse_ids(df, name):
    df[name] = (df[name] * 2) + 1810
    return df

# data = anonymyse_ids(data, "company_id")


data["ledger_id"] = data["ledger_id"].astype(str)
data['effective_date'] = pd.to_datetime(data['effective_date'])
data = data.drop(columns=["type"])

mask = (
    data['ledger_id'].str.startswith('44') |
    data['ledger_id'].str.startswith('40') |
    data['ledger_id'].str.startswith('42') |
    data['ledger_id'].str.startswith('45') |
    data['ledger_id'].str.startswith('7') |
    data['ledger_id'].str.startswith('800') |
    data['ledger_id'].str.startswith('801') |
    data['ledger_id'].str.startswith('802') |
    data['ledger_id'].str.startswith('803') |
    data['ledger_id'].str.startswith('804') |
    data['ledger_id'].str.startswith('805') |
    data['ledger_id'].str.startswith('82')
)
data = data[mask]
data = data[data["effective_date"] >= '2022-01-01']

data["AR/AP"] = np.where(data["ledger_id"].str.startswith("8"),
                         "Account Receivable", "Account Payable")
data["Classification"] = np.where((data["ledger_id"].str.startswith("800")) | (data["ledger_id"].str.startswith("801")) | (
    data["ledger_id"].str.startswith("802")) | (data["ledger_id"].str.startswith("803")) | (data["ledger_id"].str.startswith("804")), "Operational Revenue", "")
data["Classification"] = np.where((data["ledger_id"].str.startswith("805")) | (
    data["ledger_id"].str.startswith("82")), "Other Revenue", data["Classification"])
data["Classification"] = np.where((data["ledger_id"].str.startswith("40")) | (
    data["ledger_id"].str.startswith("44")), "Organizational Costs", data["Classification"])
data["Classification"] = np.where((data["ledger_id"].str.startswith("42")) | (
    data["ledger_id"].str.startswith("45")), "Other Operating Costs", data["Classification"])
data["Classification"] = np.where(
    (data["ledger_id"].str.startswith("7")), "COGS", data["Classification"])
# data.to_csv("raw_but_final_df.csv")

data["amount"] = abs(data["amount"])

grouped_data1 = data.groupby(["company_id", "Classification"])[
    "amount"].count().reset_index()
grouped_data2 = data.groupby(["company_id", "Classification"])[
    "amount"].sum().reset_index()
overview = grouped_data1.merge(
    grouped_data2, on=["company_id", "Classification"])
overview.to_csv("grouped_data.csv")


data.to_csv("classified_data.csv")
