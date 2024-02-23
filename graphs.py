import pandas as pd
import matplotlib.pyplot as plt

revenue = pd.read_csv("filtered_revenue.csv")
revenue["rev_in_ks"] = revenue["Revenue 2023"]/1000

revenue["rev_in_ks"].describe()

fig = plt.figure(figsize=(10, 7))
plt.boxplot(revenue["rev_in_ks"])
plt.show()
