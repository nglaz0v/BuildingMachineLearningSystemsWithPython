# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import pandas as pd
import numpy as np
# from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt


# boston = load_boston()
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston_target = raw_df.values[1::2, 2]

# Index number five in the number of rooms
fig, ax = plt.subplots()
ax.scatter(boston_data[:, 5], boston_target)
ax.set_xlabel("Average number of rooms (RM)")
ax.set_ylabel("House Price")

x = boston_data[:, 5]
# fit (used below) takes a two-dimensional array as input. We use np.atleast_2d
# to convert from one to two dimensional, then transpose to make sure that the
# format matches:
x = np.transpose(np.atleast_2d(x))

y = boston_target

lr = LinearRegression(fit_intercept=False)
lr.fit(x, y)

ax.plot([0, boston_data[:, 5].max() + 1],
        lr.predict([[0], [boston_data[:, 5].max() + 1]]), '-', lw=4)
fig.savefig('Figure1.png')

mse = mean_squared_error(y, lr.predict(x))
rmse = np.sqrt(mse)
print('RMSE (no intercept): {}'.format(rmse))

# Repeat, but fitting an intercept this time:
lr = LinearRegression(fit_intercept=True)

lr.fit(x, y)

fig, ax = plt.subplots()
ax.set_xlabel("Average number of rooms (RM)")
ax.set_ylabel("House Price")
ax.scatter(boston_data[:, 5], boston_target)
xmin = x.min()
xmax = x.max()
ax.plot([xmin, xmax], lr.predict([[xmin], [xmax]]), '-', lw=4)
fig.savefig('Figure2.png')

mse = mean_squared_error(y, lr.predict(x))
print("Mean squared error (of training data): {:.3}".format(mse))

rmse = np.sqrt(mse)
print("Root mean squared error (of training data): {:.3}".format(rmse))

cod = r2_score(y, lr.predict(x))
print('COD (on training data): {:.2}'.format(cod))
