import boto3
import pandas as pd
import numpy as np
import sagemaker
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_squared_error
from data_processing import prepare_data

# data = prepare_data()
"""
role = sagemaker.get_execution_role()
bucket='bohdan-example-data-sagemaker'
data_key = 'monthly-beer-production-in-austr.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)
"""
data = pd.read_csv(data_location, header=0, index_col=0)

data.plot()
pyplot.show()

model = ARIMA(data, order=(3, 1, 0))
model_fit = model.fit()
model_fit.summary()
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show
residuals.plot(kind='kde')
pyplot.show()
X = data.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)


rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
