import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_clean_data(file_path):
    """
    Load and clean data by dropping unnecessary columns and handling headers.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    data = pd.read_excel(file_path)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    return data


def calculate_mean(data):
    """
    Calculate the mean for each column in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing mean values.
    """
    mean_df = pd.DataFrame(columns=data.columns)
    for col in data.columns:
        mean_df.at[0, col] = data[col].mean()
    return mean_df


def prep_df(df):
    """
    Prepare the DataFrame for analysis by reshaping and cleaning it.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Long-form DataFrame with separate columns for Category and Model.
    """
    df.columns = [f"{col}_{df[col][0]}" if not col.startswith(
        'Unnamed') else col for col in df.columns]

    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Company_ID'}, inplace=True)
    df['Company_ID'] = df['Company_ID'].fillna(method='ffill')

    df_long = pd.melt(df, id_vars='Company_ID',
                      var_name='Category_Model', value_name='Value')
    df_long[['Category', 'Model']] = df_long['Category_Model'].str.split(
        '_', expand=True)
    df_long.drop(columns=['Category_Model'], inplace=True)
    df_long['Category'] = df_long['Category'].str.replace(
        ".1", "", regex=False).str.replace(".2", "", regex=False)
    return df_long


def plot_boxplots(filtered_df, title, x_col, y_col, hue_col=None):
    """
    Create a boxplot for the filtered DataFrame.

    Args:
        filtered_df (pd.DataFrame): Filtered DataFrame.
        title (str): Title of the plot.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        hue_col (str, optional): Column name for hue. Defaults to None.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x_col, y=y_col, hue=hue_col, data=filtered_df)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()


def process_and_plot(file_path, threshold=40000):
    """
    Process the data, filter it, and generate boxplots for different models.

    Args:
        file_path (str): Path to the Excel file containing the errors.
        threshold (int): Maximum value for filtering.

    Returns:
        None
    """
    rmse_all_errors = pd.read_excel(file_path, sheet_name="rmse")
    rmse_all_errors = prep_df(rmse_all_errors).dropna(subset=["Company_ID"])

    filtered_df = rmse_all_errors[rmse_all_errors["Value"] <= threshold]

    plot_boxplots(filtered_df, 'Box Plot of Values by Category for Different Models',
                  'Category', 'Value', 'Model')

    for model in ["arima", "random forest", "lstm"]:
        filtered_model_df = filtered_df[filtered_df["Model"] == model]
        plot_boxplots(
            filtered_model_df, f'Box Plot of Values for {model.upper()}', 'Category', 'Value')


if __name__ == "__main__":
    """
    Main execution block for processing and visualizing data.
    """
    mse_file_path = "output_tables/lstm/mse.xlsx"
    all_errors_path = "all_errors.xlsx"

    mse_data = load_and_clean_data(mse_file_path)
    mean_df = calculate_mean(mse_data)

    process_and_plot(all_errors_path)
