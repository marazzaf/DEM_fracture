#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys

Gc = 1e-2

h1 = 1 / 5
data1 = np.loadtxt('elastic_en/elastic_energy_5.txt')
diff_1 = data1[1:,1] - data1[:-1,1]
h2 = 1 / 20
data2 = np.loadtxt('elastic_en/elastic_energy_20.txt')
diff_2 = data2[1:,1] - data2[:-1,1]
h3 = 1 / 10
data3 = np.loadtxt('elastic_en/elastic_energy_10.txt')
diff_3 = data3[1:,1] - data3[:-1,1]

G1 = max(-diff_1 / h1) + abs(min(-diff_1 / h1))
G2 = max(-diff_2 / h2) + abs(min(-diff_2 / h2))
G3 = max(-diff_3 / h3) + abs(min(-diff_3 / h3))
print(G1, abs(G1-Gc)/Gc*100)
print(G3, abs(G3-Gc)/Gc*100)
print(G2, abs(G2-Gc)/Gc*100)

#plot afterwards.
plt.plot(-diff_1 / h1, label='5')
plt.plot(-diff_2 / h2, label='20')
plt.plot(-diff_3 / h3, label='10')
plt.legend()
plt.savefig('G.pdf')
plt.show()
