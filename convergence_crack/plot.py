# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

res = np.loadtxt('res.txt', skiprows=0, comments='#')

#plt.xlabel('Size of mesh - h')
plt.xlabel('Number of dof')
plt.ylabel('error')
plt.xscale('log')
plt.yscale('log')
#plt.xlim([4.e-2,3.e-1])
#plt.ylim([5.e-4,9.e-4])
#plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

nb = res[:,0]

plt.plot(nb, res[:,1], 'bo', label='$L^2$ error in $u_{CR}$')
plt.plot(nb, res[:,2], 'go', label='error in energy-norm')

print('erreur L^2 DG 1:')
print(2. * np.log(res[:-1,1] / res[1:,1]) / np.log(res[:-1,0] / res[1:,0]))
#print(2. * np.log(res[1:,2] / res[0,2]) / np.log(res[0,1] / res[1:,1]))

print('erreur energy:')
print(2. * np.log(res[:-1,2] / res[1:,2]) / np.log(res[:-1,0] / res[1:,0]))
#print(2. * np.log(res[1:,3] / res[0,3]) / np.log(res[0,1] / res[1:,1]))

a, b, c, d, e =  np.polyfit(np.log(res[:,0]), np.log(res[:,2]), 1, full=True)
print('Du:')
print(2. * np.abs(a[0]))
print(a[1])
print(1. - b[0])
plt.plot(nb,np.exp(a[1])*nb**a[0], label='Fit', color='black')
#plt.text(1.e5, 1.e-3, 'order 1', fontsize=16)
#plt.text(0.1, 2.e-4, b'$r^2=0.9968$', fontsize=16)
a, b, c, d, e =  np.polyfit(np.log(res[:,0]), np.log(res[:,1]), 1, full=True)
print('u:')
print(2. * np.abs(a[0]))
print(a[1])
print(1. - b[0])
plt.plot(nb,np.exp(a[1])*nb**a[0], color='black')
#plt.text(1.e5, 2.e-6, 'order 2', fontsize=16)
#plt.text(0.1, 2.e-4, b'$r^2=0.9968$', fontsize=16)

plt.legend(loc = 'lower right')
plt.savefig('conv_elasticite.pdf')
plt.show()
