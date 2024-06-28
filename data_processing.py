import pandas as pd
from matplotlib import pyplot as plt


def anonymyse_ids(df, name):
    df[name] = (df[name] * 2) + 1810
    return df


def data_prep(weekly=False):

    if weekly:
        threshold = 40
    else:
        threshold = 70

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

                all_dates = pd.DataFrame(pd.date_range(
                    start='2022-01-01', end='2023-12-31'), columns=['effective_date'])

                # Merge the original DataFrame with all dates
                merged_df = pd.merge(
                    all_dates, cut, on='effective_date', how='left')

                # Fill missing values
                merged_df['Unnamed: 0'] = merged_df['Unnamed: 0'].fillna(
                    0).astype(int)
                merged_df['company_id'] = merged_df['company_id'].fillna(
                    company).astype(int)
                merged_df['Classification'] = merged_df['Classification'].fillna(
                    type)
                merged_df['amount'] = merged_df['amount'].fillna(0)
                merged_df['week'] = merged_df['effective_date'].dt.isocalendar().week
                merged_df['year'] = merged_df['effective_date'].dt.year

                output_list[company][type] = merged_df
            else:
                output_list[company][type] = False

    return output_list


if __name__ == "__main__":
    response = data_prep()
