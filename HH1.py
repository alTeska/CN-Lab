#class 1
import matplotlib.pyplot as plt
import numpy as np

def alfaM(u):
    x = 2.5 - 0.1 * (u + 65)
    return x / (np.exp(x) - 1)

def alfaN(u):
    return (0.1 - 0.01 * (u + 65)) / (np.exp(1 - 0.1 * (u + 65)))

def alfaH(u):
    return 0.07 * np.exp((u + 65) / 20)

def betaM(u):
    return 4 * np.exp(-(u + 65) / 18)

def betaN(u):
    return 0.125 * np.exp(-(u + 65) / 80)

def betaH(u):
    return 1 / (np.exp(3 - 0.1 * (u + 65)) + 1)

# F: < m,n,h>
dM = lambda m,u: alfaM(u) * (1 - m) - betaM(u) * m
dN = lambda n,u: alfaN(u) * (1 - n) - betaN(u) * n
dH = lambda h,u: alfaH(u) * (1 - h) - betaH(u) * h

#u = -65
dt = 1e-3
u = np.arange(-80, 20)

#compute steady state values - system specification
MSS = alfaM(u) / (alfaM(u) + betaM(u))
NSS = alfaN(u) / (alfaN(u) + betaN(u))
HSS = alfaH(u) / (alfaH(u) + betaH(u))

fig1 = plt.plot(u, MSS)
plt.grid(True)
plt.ioff()
plt.show()

#starting points - values of gating variables at the starting time
#for starting points the MSS when u = 0

l = int(10/dt)
t = np.zeros(l)
t[0] = 0

M = np.zeros(l)
M[0] = -80

N = np.zeros(l)
N[0] = -80

H = np.zeros(l)
H[0] = -80

u = -65 #membrane with fixed voltage

#evolution of state values over time
for n in np.arange(0,l-1,1):
    M[n+1] = M[n] + dM(M[n], u) * dt
    N[n+1] = N[n] + dN(N[n], u) * dt
    H[n+1] = H[n] + dH(H[n], u) * dt
    t[n+1] = n+1

#fig = plt.plot(t, M)
#plt.grid(True)
#plt.ioff()
#plt.show()
