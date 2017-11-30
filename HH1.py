##Hodgkin-Huxley Model - class 1
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Constants
C_m  =   0.5 #membrane capacitance, in uF/cm^2
g_Na = 120.0 #maximum conducances, in mS/cm^2
g_K  =  36.0
g_L  =   0.3
E_Na =  50.0 #Nernst reversal potentials, in mV
E_K  = -77.0
E_L = -54.387

##non-linear paramiters
alpha_m = lambda u: (2.5 - 0.1 * (u + 65)) / (np.exp(2.5 - 0.1 * (u + 65)) - 1)
alpha_n = lambda u: (0.1 - 0.01 * (u + 65)) / (np.exp(1 - 0.1 * (u + 65)) - 1)
alpha_h = lambda u: (0.07 * np.exp((-u - 65) / 20))
beta_m = lambda u: 4 * np.exp((-u - 65) / 18)
beta_n = lambda u: 0.125 * np.exp((-u - 65) / 80)
beta_h = lambda u: 1 / (np.exp(3 - 0.1 * (u + 65)) + 1)

#Membrane currents
INa = lambda u, m, h: g_Na * m**3 * h * (u - E_Na)
IK  = lambda u, n   : g_K  * n**4     * (u - E_K)
IL  = lambda u      : g_L             * (u - E_L)

# F: < m,n,h>
dMdt = lambda m,u: alpha_m(u) * (1 - m) - beta_m(u) * m
dNdt = lambda n,u: alpha_n(u) * (1 - n) - beta_n(u) * n
dHdt = lambda h,u: alpha_h(u) * (1 - h) - beta_h(u) * h
dUdt = lambda u, m, n, h, I=0: I - INa(u, m, h)- IK(u, n)- IL(u)/ C_m

#compute steady state values - system specification
MSS = lambda u: alpha_m(u) / (alpha_m(u) + beta_m(u))
NSS = lambda u: alpha_n(u) / (alpha_n(u) + beta_n(u))
HSS = lambda u: alpha_h(u) / (alpha_h(u) + beta_h(u))

#starting points - values of gating variables at the starting time
#for starting points the MSS when u = 0
#initiating
dt = 1e-3
l = int(10/dt)
t = np.zeros(l)
u = -65 #membrane with fixed voltage

M = np.zeros(l)
N = np.zeros(l)
H = np.zeros(l)
U = np.zeros(l)
t[0] = 0
M[0] = MSS(-80)
N[0] = NSS(-80)
H[0] = HSS(-80)
U[0] = dUdt(-80, M[0], N[0], H[0])

#evolution of state values over time
for n in np.arange(0,l-1,1):
    M[n+1] = M[n] + dMdt(M[n], u) * dt
    N[n+1] = N[n] + dNdt(N[n], u) * dt
    H[n+1] = H[n] + dHdt(H[n], u) * dt
    U[n+1] = U[n] + dUdt(U[n], M[n+1], N[n+1], H[n+1]) * dt
    t[n+1] = n+1

##Plots
##probability of steady states plot
dt = 1e-3
u = np.arange(-80, 20)
fig, ax = plt.subplots()
ax.plot(u, MSS(u), label="MSS")
ax.plot(u, NSS(u), label="NSS")
ax.plot(u, HSS(u), label="HSS")
ax.grid(True)
ax.legend(loc=2); # upper left corner
ax.set_xlabel('u')
ax.set_ylabel('probability')
ax.set_title('Steady states probabilties');

fig2, ax2 = plt.subplots()
ax2.plot(t, M, label="M")
ax2.plot(t, N, label="N")
ax2.plot(t, H, label="H")
#ax2.plot(t, U, label="U")
ax2.grid(True)
ax2.legend(loc=2); # upper left corner

plt.show()
