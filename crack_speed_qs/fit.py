# coding: utf-8

import numpy as np
import scipy.stats as st

size_ref = 20

values = np.loadtxt('du_3/length_crack_%i.txt' % size_ref)

mu = 0.2
H = 1
Gc = 0.01
ref_vel = np.sqrt(mu*H/Gc) #ref velocity (computed by me !)
print(ref_vel)

x = values[:,0]
y = values[:,1]


slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
print(slope)
print('Relative error slope: %.3e%%' % (np.absolute(ref_vel - slope) / ref_vel * 100))
print(r_value)
