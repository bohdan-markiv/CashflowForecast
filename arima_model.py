import matplotlib.pyplot as plt
import pandas as pd
from data_processing import prepare_data

# data = prepare_data()
data = pd.read_csv("example_data/shampoo.csv")

plt.plot(data["PAY"]["Date"], data["PAY"]['Amount_EUR'],
         label='amount', marker='o')


plt.xlabel("amount")
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.show()
