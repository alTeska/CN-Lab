#class 1
import matplotlib.pyplot as plt
import numpy as np

C = 1e3
Tr = 20
K = 0.2 # * W/K

def dTcoffee(K, C, Tr, Tc):
    return K/C * (Tr - Tc)

def dTcSteadState(K, C, Tr, dTcoffee):
    return Tr - dTcoffee * C/K

def Tcoffee(t, K, C, Tr, Tc0):
    return Tr + (Tc0 - Tr) * np.exp(-K/C * t)

dt = 1
l = int(30000/dt)
t = np.zeros(l)
T = np.zeros(l)
T[0] = 80
t[0] = 0

for n in np.arange(0,l-1,1):
    dT = dTcoffee(K, C, Tr, T[n])
    T[n+1] = T[n] + dT * dt
    t[n+1] = n+1

print (t)
print (T)

fig = plt.plot(t, T)
#plt.ylim([0,2])
plt.grid(True)
plt.ioff()
plt.show()
