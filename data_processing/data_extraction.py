import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def anonymize_ids(df, column_name):
    """
    Anonymize IDs by applying a transformation.

    Args:
        df (pd.DataFrame): DataFrame containing the column to anonymize.
        column_name (str): Name of the column to anonymize.

    Returns:
        pd.DataFrame: DataFrame with anonymized IDs.
    """
    df[column_name] = (df[column_name] * 2) + 1810
    return df


def convert_date(date):
    """
    Convert a date from Excel serial format or string to datetime.date.

    Args:
        date: Date in Excel serial format or string.

    Returns:
        datetime.date: Converted date or NaT if conversion fails.
    """
    try:
        if isinstance(date, int) or (isinstance(date, str) and date.isdigit()):
            excel_start_date = datetime(1899, 12, 30)
            delta = timedelta(days=int(date))
            return (excel_start_date + delta).date()
        else:
            return pd.to_datetime(date).date()
    except Exception:
        return pd.NaT


def process_data(data_path, revenues_path, output_dir):
    """
    Process and classify financial data for analysis.

    Args:
        data_path (str): Path to the full data set CSV.
        revenues_path (str): Path to the revenues CSV.
        output_dir (str): Directory to save processed files.
    """
    # Load data
    data = pd.read_csv(data_path)
    revenues = pd.read_csv(revenues_path)

    # Filter data by company and division IDs
    company_ids = [27934117, 262866, 24872290, 22205603, 11036957, 2199709,
                   8922050, 18305616, 2807402, 9646170, 11745239, 5645364,
                   27233208, 9296446, 397293, 16862710, 10209459,
                   8685703, 30940084, 17200872, 25061843, 31927532, 1172922,
                   248552, 25265199, 30549617]
    revenue_divisions = [2163245, 999296, 2659781, 236141, 1555285, 2403092, 925373,
                         1426111, 2850605, 1122522, 1385214, 2111655, 2781225,
                         375035, 365180, 3358248, 1186317, 3388580, 1890439,
                         551084, 2143, 1235235, 1006367, 2465565, 1410815, 2448229]

    data = data[data["company_id"].isin(company_ids)]
    revenues = revenues[revenues["Division"].isin(revenue_divisions)]
    revenues.to_csv(f"{output_dir}/filtered_revenue.csv", index=False)

    # Convert dates and filter data
    data["ledger_id"] = data["ledger_id"].astype(str)
    data['effective_date'] = data['effective_date'].apply(convert_date)
    data = data.drop(columns=["type"])

    # Filter ledger IDs and dates
    mask = data['ledger_id'].str.startswith(
        ('44', '40', '42', '45', '7', '800', '801', '802', '803', '804', '805', '82'))
    data = data[mask]
    data = data[data["effective_date"] >=
                datetime.strptime('2022-01-01', '%Y-%m-%d').date()]

    # Classify transactions
    data["AR/AP"] = np.where(data["ledger_id"].str.startswith("8"),
                             "Account Receivable", "Account Payable")
    data["Classification"] = np.where(data["ledger_id"].str.startswith(
        ("800", "801", "802", "803", "804")), "Operational Revenue", "")
    data["Classification"] = np.where(data["ledger_id"].str.startswith(
        ("805", "82")), "Other Revenue", data["Classification"])
    data["Classification"] = np.where(data["ledger_id"].str.startswith(
        ("40", "44")), "Organizational Costs", data["Classification"])
    data["Classification"] = np.where(data["ledger_id"].str.startswith(
        ("42", "45")), "Other Operating Costs", data["Classification"])
    data["Classification"] = np.where(
        data["ledger_id"].str.startswith("7"), "COGS", data["Classification"])

    # Ensure positive amounts
    data["amount"] = abs(data["amount"])

    # Save grouped data
    grouped_data1 = data.groupby(["company_id", "Classification"])[
        "amount"].count().reset_index()
    grouped_data2 = data.groupby(["company_id", "Classification"])[
        "amount"].sum().reset_index()
    overview = grouped_data1.merge(
        grouped_data2, on=["company_id", "Classification"])
    overview.to_csv(f"{output_dir}/overview_data.csv", index=False)

    # Analyze AR/AP data
    grouped_on_ar_ap = data.groupby(
        ["company_id", "AR/AP"])["amount"].agg(["mean", "count", "std"]).reset_index()
    wide_df = grouped_on_ar_ap.pivot_table(index='company_id',
                                           columns='AR/AP',
                                           values=['mean', 'count', "std"]).reset_index()

    # Flatten MultiIndex columns
    wide_df.columns = ['_'.join(col).strip() if col[1] else col[0]
                       for col in wide_df.columns.values]
    wide_df.rename(columns={
        'mean_Account Payable': 'AP_mean',
        'mean_Account Receivable': 'AR_mean',
        'count_Account Payable': 'AP_count',
        'count_Account Receivable': 'AR_count',
        'std_Account Payable': 'AP_std',
        'std_Account Receivable': 'AR_std',
    }, inplace=True)

    wide_df = anonymize_ids(wide_df, "company_id")
    wide_df.to_csv(f"{output_dir}/wide_data.csv", index=False)

    # Group by company, classification, and date
    grouped_data = data.groupby(["company_id", "Classification", "effective_date"])[
        "amount"].sum().reset_index()
    grouped_data.to_csv(f"{output_dir}/grouped_data.csv", index=False)

    # Save classified data
    data.to_csv(f"{output_dir}/classified_data.csv", index=False)


if __name__ == "__main__":
    data_path = ""
    revenues_path = "revenues.csv"
    output_dir = "processed_data"
    process_data(data_path, revenues_path, output_dir)
