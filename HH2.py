##Hodgkin-Huxley Model - class 2
import matplotlib.pyplot as plt
import numpy as np
#import scipy as sp

# Constants
C_m  =   0.1 #membrane capacitance, in uF/cm^2
g_Na = 120.0 #maximum conducances, in mS/cm^2
g_K  =  36.0
g_L  =   0.3
E_Na =  55.0 #Nernst reversal potentials, in mV
E_K  = -75.0
E_L  = -69

##non-linear paramiters alfa and beta
def alpha_m(u): return (2.5 - 0.1 * (u + 65)) / (np.exp(2.5 - 0.1 * (u + 65)) - 1)
def alpha_n(u): return (0.1 - 0.01 * (u + 65)) / (np.exp(1 - 0.1 * (u + 65)) - 1)
def alpha_h(u): return (0.07 * np.exp((-u - 65) / 20))
def beta_m(u):  return 4 * np.exp((-u - 65) / 18)
def beta_n(u):  return 0.125 * np.exp((-u - 65) / 80)
def beta_h(u):  return 1 / (np.exp(3 - 0.1 * (u + 65)) + 1)

#Membrane currents
def I_Na(u, m, h): return g_Na * m**3 * h * (u - E_Na)  #sodium channel
def I_K (u, n   ): return g_K  * n**4     * (u - E_K)   #potas
def I_L (u      ): return g_L             * (u - E_L)

# F: < m,n,h>
def dm_du(m, u): return alpha_m(u) * (1 - m) - beta_m(u) * m
def dn_du(n, u): return alpha_n(u) * (1 - n) - beta_n(u) * n
def dh_du(h, u): return alpha_h(u) * (1 - h) - beta_h(u) * h
#def du_du(u, m, n, h, I=0): return I - I_Na(u, m, h)- I_K(u, n)- I_L(u)/ C_m
def du_du(u, m, n, h, I=0): return (I_Na(u, m, h) + I_K(u, n) + I_L(u) - I)/ -C_m


#compute steady state values - system specification (starting points)
def MSS(u): return alpha_m(u) / (alpha_m(u) + beta_m(u))
def NSS(u): return alpha_n(u) / (alpha_n(u) + beta_n(u))
def HSS(u): return alpha_h(u) / (alpha_h(u) + beta_h(u))

def HHModel(I, m0, n0, h0, u0):
    dt = 1e-3
    time = np.arange(0, 500, dt)

    M = np.zeros_like(time)
    N = np.zeros_like(time)
    H = np.zeros_like(time)
    U = np.zeros_like(time)
    M[0] = m0
    N[0] = n0
    H[0] = h0
    U[0] = u0

    if type(I) is np.ndarray:
        iinj = np.ones_like(time) * max(I)
        print(I.shape[0])
        #iinj[time <= max(I)*2.5] = I
        #iinj[time <= I[-1]] = I
        iinj[0:I.shape[0]] = I
    else:
        iinj = np.ones_like(time) * I

    #t_cur = 10
    #istep = np.arange(0, t_cur, dt)
    #istep = istep / (t_cur/10)
    #iinj = np.ones_like(time) * I
    #iinj[time < t_cur] = istep

    #evolution of state values over time
    for t in range(0, len(time)-1):
        M[t+1] = M[t] + dm_du(M[t], U[t]) * dt
        N[t+1] = N[t] + dn_du(N[t], U[t]) * dt
        H[t+1] = H[t] + dh_du(H[t], U[t]) * dt
        U[t+1] = U[t] + du_du(U[t], M[t+1], N[t+1], H[t+1], iinj[t]) * dt

    return U, time, M, N, H, iinj

#starting points - values of gating variables (MSS etc. when u = 0)
u0 = -70
I0 = 10
U, t, M, N, H, iinj = HHModel(I0, MSS(u0), NSS(u0), HSS(u0), u0)

def iinj_r(t_cur, Imax,  dt):
    istep = np.arange(0, t_cur, dt)
    istep = istep / (t_cur/Imax)
    return istep

I0 = iinj_r(250, 8.5, 1e-3)
U2, t, M2, N2, H2, iinj2 = HHModel(8.5, MSS(u0), NSS(u0), HSS(u0), u0)

##PLOTS
'''
fig, ax = plt.subplots()
ax.plot(t, iinj, label="M")
ax.grid(True)


fig1, ax1 = plt.subplots()
ax1.plot(t, M, label="M")
ax1.plot(t, N, label="N")
ax1.plot(t, H, label="H")
ax1.grid(True)
ax1.legend(loc=2); # upper left corner
ax1.set_xlabel('t[ms]')
ax1.set_ylabel('steady state values')
ax1.set_title('steady state values');

fig2, ax2 = plt.subplots()
ax2.plot(t, U, label="U")
ax2.grid(True)
ax2.legend(loc=2); # upper left corner
ax2.set_xlabel('t[ms]')
ax2.set_ylabel('U[mV]')
ax2.set_title('Gating voltage');
'''
figi, axi = plt.subplots()
axi.plot(t, iinj2, label="M")
axi.grid(True)

fig3, ax3 = plt.subplots()
ax3.plot(t, M2, label="M")
ax3.plot(t, N2, label="N")
ax3.plot(t, H2, label="H")
ax3.grid(True)
ax3.legend(loc=2); # upper left corner
ax3.set_xlabel('t[ms]')
ax3.set_ylabel('steady state values')
ax3.set_title('steady state values');

fig4, ax4 = plt.subplots()
ax4.plot(t, U2, label="U")
ax4.grid(True)
ax4.legend(loc=2); # upper left corner
ax4.set_xlabel('t[ms]')
ax4.set_ylabel('U[mV]')
ax4.set_title('Gating voltage');

plt.show()
