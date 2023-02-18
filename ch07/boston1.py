# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script shows an example of simple (ordinary) linear regression

# The first edition of the book NumPy functions only for this operation. See
# the file boston1numpy.py for that version.

import pandas as pd
import numpy as np
# from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


# boston = load_boston()
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston_target = raw_df.values[1::2, 2]
x = boston_data
y = boston_target

# Fitting a model is trivial: call the ``fit`` method in LinearRegression:
lr = LinearRegression()
lr.fit(x, y)

# The instance member `residues_` contains the sum of the squared residues
# rmse = np.sqrt(lr.residues_/len(x))
rmse = mean_squared_error(y, lr.predict(x), squared=False)
print('RMSE: {}'.format(rmse))

fig, ax = plt.subplots()
# Plot a diagonal (for reference):
ax.plot([0, 50], [0, 50], '-', color=(.9, .3, .3), lw=4)

# Plot the prediction versus real:
ax.scatter(lr.predict(x), boston_target)

ax.set_xlabel('predicted')
ax.set_ylabel('real')
fig.savefig('Figure_07_08.png')
