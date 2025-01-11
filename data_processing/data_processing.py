import pandas as pd
from matplotlib import pyplot as plt


def anonymyse_ids(df, name):
    """
    Anonymize IDs by applying a transformation.

    Args:
        df (pd.DataFrame): DataFrame containing the column to anonymize.
        name (str): Name of the column to anonymize.

    Returns:
        pd.DataFrame: DataFrame with anonymized IDs.
    """
    df[name] = (df[name] * 2) + 1810
    return df


def data_prep(weekly=False):
    """
    Prepare and process financial data for analysis.

    Args:
        weekly (bool): Flag to determine whether to aggregate data weekly.

    Returns:
        dict: Processed data grouped by company and classification.
    """
    # Set threshold for minimum required data points
    threshold = 40 if weekly else 70

    output_list = {}

    # Load and anonymize data
    data = anonymyse_ids(pd.read_csv("data/grouped_data.csv"), "company_id")

    # Filter data by effective date
    data = data[data["effective_date"] <= '2023-12-31']

    # Create weekly and yearly data columns
    data['effective_date'] = pd.to_datetime(data['effective_date'])
    data["week"] = data["effective_date"].dt.isocalendar().week
    data["year"] = data["effective_date"].dt.isocalendar().year

    # Aggregate data weekly if specified
    if weekly:
        weekly_data = data.groupby(["company_id", "Classification", "week", "year"])[
            "amount"].sum().reset_index()
    else:
        weekly_data = data

    # Process data for each company and classification type
    for company in set(weekly_data["company_id"]):
        output_list[company] = {}
        for type in {"COGS", "Other Operating Costs", "Other Revenue", "Organizational Costs", "Operational Revenue"}:

            cut = weekly_data[(weekly_data["company_id"] == company) &
                              (weekly_data["Classification"] == type)]

            if len(cut) > threshold:
                # Create a DataFrame with all dates in the range
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
    """
    Main execution block to process data and prepare it for analysis.
    """
    response = data_prep()
