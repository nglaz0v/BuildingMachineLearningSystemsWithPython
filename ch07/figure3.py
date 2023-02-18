# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from sklearn.linear_model import LinearRegression, Lasso
import pandas as pd
import numpy as np
# from sklearn.datasets import load_boston
from matplotlib import pyplot as plt


# boston = load_boston()
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston_target = raw_df.values[1::2, 2]
fig, ax = plt.subplots()
ax.scatter(boston_data[:, 5], boston_target)
ax.set_xlabel("Number of rooms (RM)")
ax.set_ylabel("House Price")


x = boston_data[:, 5]
xmin = x.min()
xmax = x.max()
x = np.transpose(np.atleast_2d(x))
y = boston_target

lr = LinearRegression()
lr.fit(x, y)
ax.plot([xmin, xmax], lr.predict([[xmin], [xmax]]), ':', lw=4, label='OLS model')

las = Lasso()
las.fit(x, y)
ax.plot([xmin, xmax], las.predict([[xmin], [xmax]]), '-', lw=4, label='Lasso model')
fig.savefig('Figure3.png')
