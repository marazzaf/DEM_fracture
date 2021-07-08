#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys

h1 = 1 / 10
data1 = np.loadtxt('elastic_en/elastic_energy_10.txt')
diff_1 = data1[1:,1] - data1[:-1,1]
print(diff_1)
sys.exit()
h2 = 1 / 20
data2 = np.loadtxt('elastic_en/elastic_energy_20.txt')
diff_2 = data2[1:,1] - data2[:-1,1]

#plot afterwards.
plt.plot(diff_1 / h1)
plt.plot(diff_2 / h2)
plt.show()
