import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("grouped_data.csv")

data = data[data["effective_date"] <= '2023-12-31']

grouped = data.groupby(["company_id", "Classification"])[
    "amount"].agg(["mean", "count", "std"]).reset_index()

# Cretae weekly data
data['effective_date'] = pd.to_datetime(data['effective_date'])
data["week"] = data["effective_date"].dt.isocalendar().week
data["year"] = data["effective_date"].dt.isocalendar().year

weekly_data = data.groupby(["company_id", "Classification", "week", "year"])[
    "amount"].sum().reset_index()

example = weekly_data[(weekly_data["company_id"] == 248552) & (
    weekly_data["Classification"] == "Operational Revenue")]

example = example[["week", "year", "amount"]]
example.index = example["week"].astype(str) + example["year"].astype(str)
example = example.drop(columns=["week", "year"])

plt.plot(example)
plt.show()
