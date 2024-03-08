import pandas as pd
import numpy as np
from datetime import datetime, timedelta

data = pd.read_csv("C:/Users/b.markiv/Downloads/full_data_set.csv")
revenues = pd.read_csv("revenues.csv")

data = data[data["company_id"].isin([27934117, 262866, 24872290, 22205603, 11036957, 2199709,
                                     8922050, 18305616, 2807402, 9646170, 11745239, 5645364,
                                     27233208, 9296446, 397293, 16862710, 10209459,
                                     8685703, 30940084, 17200872, 25061843, 31927532, 1172922,
                                     248552, 25265199, 30549617])]

revenues = revenues[revenues["Division"].isin([2163245, 999296, 2659781, 236141, 1555285, 2403092, 925373,
                                               1426111, 2850605, 1122522, 1385214,
                                               2111655, 2781225, 375035, 365180, 3358248,
                                               1186317, 3388580, 1890439, 551084, 2143, 1235235,
                                               1006367, 2465565, 1410815, 2448229])]
revenues.to_csv("filtered_revenue.csv")

def anonymyse_ids(df, name):
    df[name] = (df[name] * 2) + 1810
    return df

def convert_date(date):
    # Check if the value is an integer, which would imply it's an Excel serial date
    try:
        if isinstance(date, int) or (isinstance(date, str) and date.isdigit()):
            excel_start_date = datetime(1899, 12, 30)
            delta = timedelta(days=int(date))
            return (excel_start_date + delta).date()
        else:
            # Parse the date string to datetime
            return pd.to_datetime(date).date()
    except Exception as e:
        return pd.NaT 

# data = anonymyse_ids(data, "company_id")




data["ledger_id"] = data["ledger_id"].astype(str)
data['effective_date'] = data['effective_date'].apply(convert_date)
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
data = data[data["effective_date"] >= datetime.strptime('2022-01-01', '%Y-%m-%d').date()]

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

overview.to_csv("overview_data.csv")

grouped_on_ar_ap = data.groupby(["company_id", "AR/AP"])[
    "amount"].agg(["mean", "count", "std"]).reset_index()

wide_df = grouped_on_ar_ap.pivot_table(index='company_id', 
                         columns='AR/AP', 
                         values=['mean', 'count', "std"]).reset_index()

# Flatten the MultiIndex in columns
wide_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in wide_df.columns.values]

wide_df.rename(columns={
    'company_id_': 'company_id', 
    'mean_Account Payable': 'AP_mean', 
    'mean_Account Receivable': 'AR_mean', 
    'count_Account Payable': 'AP_count', 
    'count_Account Receivable': 'AR_count',
    'std_Account Payable': 'AP_std', 
    'std_Account Receivable': 'AR_std',
}, inplace=True)

wide_df = anonymyse_ids(wide_df, "company_id")
grouped_data = data.groupby(["company_id", "Classification", "effective_date"])[
    "amount"].sum().reset_index()
grouped_data.to_csv("grouped_data.csv")

data.to_csv("classified_data.csv")
