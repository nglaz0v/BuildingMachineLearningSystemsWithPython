# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script generates web traffic data for our hypothetical
# web startup "MLASS" in chapter 01

import os
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

from utils import DATA_DIR, CHART_DIR

np.random.seed(3)  # to reproduce the data later on

x = np.arange(1, 31*24)
y = np.array(200*(np.sin(2*np.pi*x/(7*24))), dtype=float)
y += gamma.rvs(15, loc=0, scale=100, size=len(x))
y += 2 * np.exp(x/100.0)
y = np.ma.array(y, mask=[y < 0])
print(sum(y), sum(y < 0))

plt.scatter(x, y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(5)],
           ['week %i' % (w+1) for w in range(5)])
plt.autoscale(tight=True)
plt.grid()
plt.savefig(os.path.join(CHART_DIR, "1400_01_01.png"))

np.savetxt(os.path.join(DATA_DIR, "web_traffic.tsv"),
           list(zip(x, y)), delimiter="\t", fmt="%s")
