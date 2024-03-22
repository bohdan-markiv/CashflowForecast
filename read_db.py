import sqlite3
import pandas as pd


con = sqlite3.connect("CF_db.db")

table_df = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type = 'table'", con)
print(table_df)  # This will print the list of tables as a DataFrame

# Fetching data from 'companies_overview' into a DataFrame
companies_overview_df = pd.read_sql_query(
    "SELECT * FROM companies_overview", con)
print(companies_overview_df)

# Be sure to close the connection
con.close()
