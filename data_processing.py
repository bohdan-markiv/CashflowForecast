import pandas as pd
from matplotlib import pyplot as plt


def anonymyse_ids(df, name):
    df[name] = (df[name] * 2) + 1810
    return df


def data_prep(weekly=False):

    if weekly:
        threshold = 40
    else:
        threshold = 50

    output_list = {}

    data = anonymyse_ids(pd.read_csv("data/grouped_data.csv"), "company_id")

    data = data[data["effective_date"] <= '2023-12-31']

    # Cretae weekly data
    data['effective_date'] = pd.to_datetime(data['effective_date'])
    data["week"] = data["effective_date"].dt.isocalendar().week
    data["year"] = data["effective_date"].dt.isocalendar().year

    if weekly:
        weekly_data = data.groupby(["company_id", "Classification", "week", "year"])[
            "amount"].sum().reset_index()
    else:
        weekly_data = data

    for company in set(weekly_data["company_id"]):
        output_list[company] = {}
        for type in {'COGS', 'Other Operating Costs', 'Other Revenue', 'Organizational Costs', 'Operational Revenue'}:

            cut = weekly_data[(weekly_data["company_id"] == company) &
                              (weekly_data["Classification"] == type)]

            if len(cut) > threshold:
                output_list[company][type] = cut
            else:
                output_list[company][type] = False

    return output_list
