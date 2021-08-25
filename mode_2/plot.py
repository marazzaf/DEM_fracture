#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import sys

data1 = np.loadtxt('unstructured/coarse/ld_4_save.txt')
data2 = np.loadtxt('unstructured/med/ld_5.txt')
data3 = np.loadtxt('unstructured/fine/ld_5_save.txt')

#regression
lim = 5
slope, intercept, r, p, se = linregress(data1[:lim,0], data1[:lim,1])

##verif
#print(intercept)
#plt.plot(data1[:lim,0], data1[:lim,1], 'o-')
#plt.show()
#sys.exit()

x = np.arange(0, 0.05, 0.0001)

plt.plot(data1[:,0], data1[:,1], 'o-', label='coarse')
plt.plot(data2[:,0], data2[:,1], 'o-', label='med')
plt.plot(data3[:,0], data3[:,1], 'o-', label='fine')
plt.plot(x, slope*x, label='elasticity')
plt.xlim(0,0.05)
plt.ylim(0, 0.3)
plt.legend()
plt.show()
