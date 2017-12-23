#class 1 -  differential equations - coffee example
import matplotlib.pyplot as plt
import numpy as np

C = 1e3
Tr = 20
K = 0.2 # * W/K

def dTcSteadState(K, C, Tr, dTcoffee):
    return Tr - dTcoffee * C/K

def analytic_dTcoffee(t, K, C, Tr, Tc0):
    return Tr + (Tc0 - Tr) * np.exp(-K/C * t)

def dTcoffee(K, C, Tr, Tc):
    return K/C * (Tr - Tc)

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

slope = (T[1] - T[0]) / dt
line = lambda t: slope*t + T[0]
tslope = t[:5000]

#plot
fig, ax = plt.subplots()
ax.plot(t, T, label="temp")
ax.plot(tslope, line(tslope), label="tau")
ax.set_xlabel('t')
ax.set_ylabel('coffee temp')
ax.set_title('Nummerical solution for coffee state');
ax.grid(True)
plt.show()
