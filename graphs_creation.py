import pandas as pd

# Load the data
data = pd.read_excel("output_tables/lstm/mse.xlsx")

# Drop the "Unnamed: 0" column from data, if it exists
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

# Initialize an empty DataFrame with the same columns as data
mean_df = pd.DataFrame(columns=data.columns)

# Calculate the mean for each column and store it in mean_df
for col in data.columns:
    # Store the mean in the first row of mean_df
    mean_df.at[0, col] = data[col].mean()

rmse_all_errors = pd.read_excel("all_errors.xlsx", sheet_name="rmse")
mse_all_errors = pd.read_excel("all_errors.xlsx", sheet_name="mse")
mae_all_errors = pd.read_excel("all_errors.xlsx", sheet_name="mae")


def prep_df(df):

    df.columns = [f"{col}_{df[col][0]}" if not col.startswith(
        'Unnamed') else col for col in df.columns]

    # Drop the row that was used for headers if it hasn't been set as header
    # data = data.drop(index=0)  # Uncomment if the first data row is part of data and not header

    # Resetting index if 'Unnamed: 0' column is the company ID
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Company_ID'}, inplace=True)
    # Forward fill to handle any NaNs if it makes sense
    df['Company_ID'] = df['Company_ID'].fillna(method='ffill')

    # Melting the DataFrame to long format
    df_long = pd.melt(df, id_vars='Company_ID',
                      var_name='Category_Model', value_name='Value')

    # Splitting the 'Category_Model' into separate 'Category' and 'Model' columns
    df_long[['Category', 'Model']] = df_long['Category_Model'].str.split(
        '_', expand=True)

    # Drop the combined column if no longer needed
    df_long.drop(columns=['Category_Model'], inplace=True)

    df_long['Category'] = df_long['Category'].str.replace(
        ".1", "", regex=False)
    df_long['Category'] = df_long['Category'].str.replace(
        ".2", "", regex=False)

    return df_long


rmse_all_errors = prep_df(rmse_all_errors)
mse_all_errors = prep_df(mse_all_errors)
mae_all_errors = prep_df(mae_all_errors)
