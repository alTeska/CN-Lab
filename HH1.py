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

##non-linear paramiters alfa and beta
def alpha_m(u): return (2.5 - 0.1 * (u + 65)) / (np.exp(2.5 - 0.1 * (u + 65)) - 1)
def alpha_n(u): return (0.1 - 0.01 * (u + 65)) / (np.exp(1 - 0.1 * (u + 65)) - 1)
def alpha_h(u): return (0.07 * np.exp((-u - 65) / 20))
def beta_m(u):  return 4 * np.exp((-u - 65) / 18)
def beta_n(u):  return 0.125 * np.exp((-u - 65) / 80)
def beta_h(u):  return 1 / (np.exp(3 - 0.1 * (u + 65)) + 1)

#Membrane currents
def INa(u, m, h): return g_Na * m**3 * h * (u - E_Na)  #sodium channel
def IK (u, n   ): return g_K  * n**4     * (u - E_K)   #potas
def IL (u      ): return g_L             * (u - E_L)

# step up 10 uA/cm^2 every 100ms for 400ms
def I_inj(t):
    return 10*(t>100) - 10*(t>200) + 35*(t>300)
    #return 10*t

# F: < m,n,h>
dmdu = lambda m,u: alpha_m(u) * (1 - m) - beta_m(u) * m
dndu = lambda n,u: alpha_n(u) * (1 - n) - beta_n(u) * n
dhdu = lambda h,u: alpha_h(u) * (1 - h) - beta_h(u) * h
dudu = lambda u, m, n, h, I=0: I - INa(u, m, h)- IK(u, n)- IL(u)/ C_m

#compute steady state values - system specification (starting points calc)
MSS = lambda u: alpha_m(u) / (alpha_m(u) + beta_m(u))
NSS = lambda u: alpha_n(u) / (alpha_n(u) + beta_n(u))
HSS = lambda u: alpha_h(u) / (alpha_h(u) + beta_h(u))

#starting points - values of gating variables (MSS etc. when u = 0)
dt = 1e-3
u0 = -65          #membrane with fixed voltage
time_vec = np.arange(0, 50, dt)

M = np.zeros_like(time_vec)
N = np.zeros_like(time_vec)
H = np.zeros_like(time_vec)
U = np.ones_like(time_vec) * u0
U[time_vec >= 10] = 10

dU = np.zeros_like(time_vec)

M[0] = MSS(U[0])
N[0] = NSS(U[0])
H[0] = HSS(U[0])
dU[0] = U[0]

#evolution of state values over time
for t in range(0, len(time_vec)-1):
    M[t+1] = M[t] + dmdu(M[t], U[t]) * dt
    N[t+1] = N[t] + dndu(N[t], U[t]) * dt
    H[t+1] = H[t] + dhdu(H[t], U[t]) * dt
    dU[t+1] = U[t] + dudu(U[t], M[t+1], N[t+1], H[t+1]) * dt

##Plots
#probability of steady states plot
dt = 1e-3
u = np.arange(-80, 20)

fig, ax = plt.subplots()
ax.plot(u, MSS(u), label="MSS")
ax.plot(u, NSS(u), label="NSS")
ax.plot(u, HSS(u), label="HSS")
ax.grid(True)
ax.legend(loc=2); # upper left corner
ax.set_xlabel('U[mV]')
ax.set_ylabel('probability')
ax.set_title('Steady states probabilties');

#steady states changes over time(dt)
fig2, ax2 = plt.subplots()
ax2.plot(time_vec, M, label="M")
ax2.plot(time_vec, N, label="N")
ax2.plot(time_vec, H, label="H")
ax2.grid(True)
ax2.legend(loc=2); # upper left corner
ax2.set_xlabel('t[ms]')
ax2.set_ylabel('steady state values')
ax2.set_title('Gating voltage');

#Gating voltage plot
fig3, ax3 = plt.subplots()
ax3.plot(time_vec, dU, label="U")
ax3.grid(True)
ax3.set_xlabel('t[ms]')
ax3.set_ylabel('dU[mV]')
ax3.set_title('Gating voltage');

plt.show()
