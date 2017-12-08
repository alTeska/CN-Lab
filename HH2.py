##Hodgkin-Huxley Model - class 2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

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
def du_du(u, m, n, h, I=0): return (I_Na(u, m, h) + I_K(u, n) + I_L(u) - I)/ -C_m

#compute steady state values - system specification (starting points)
def MSS(u): return alpha_m(u) / (alpha_m(u) + beta_m(u))
def NSS(u): return alpha_n(u) / (alpha_n(u) + beta_n(u))
def HSS(u): return alpha_h(u) / (alpha_h(u) + beta_h(u))

def iinj_rising(t_cur, Imax,  dt):
    istep = np.arange(0, t_cur, dt)
    istep = istep / (t_cur/Imax)
    return istep

def iinj_f(vec):
    I = lambda t: 15 + np.sin(2 * np.pi * 1e-3 * t)
    vfunc = np.vectorize(I)
    return vfunc(vec)

#HHmodel with
def HHModel(I, m0, n0, h0, u0, t):
    dt = 1e-3
    time = np.arange(0, t, dt)
    M = np.zeros_like(time)
    N = np.zeros_like(time)
    H = np.zeros_like(time)
    U = np.zeros_like(time)
    M[0], N[0], H[0], U[0] = m0, n0, h0, u0

    #if type(I) is np.ndarray:
    #    iinj = np.ones_like(time) * max(I)
    #    iinj[0 : I.shape[0]] = I
    #else:
    #    iinj = np.ones_like(time) * I

    iinj = iinj_f(time)

    #evolution of state values over time
    for t in range(0, len(time)-1):
        M[t+1] = M[t] + dm_du(M[t], U[t]) * dt
        N[t+1] = N[t] + dn_du(N[t], U[t]) * dt
        H[t+1] = H[t] + dh_du(H[t], U[t]) * dt
        U[t+1] = U[t] + du_du(U[t], M[t+1], N[t+1], H[t+1], iinj[t]) * dt

    return U, time, M, N, H, iinj

#starting points - values of gating variables (MSS etc. when u = 0)
t = 500
u0 = -70
I0 = iinj_rising(250, 10, 1e-3)
U, t, M, N, H, iinj = HHModel(I0, MSS(u0), NSS(u0), HSS(u0), u0, t)

##PLOT
fig, axes = plt.subplots(3, 1, figsize=(12, 4))
axes[0].plot(t, iinj, label="Iinj")
axes[0].grid(True)
axes[0].set_ylabel('I[uA]')
axes[0].set_title("Injected current")

axes[1].plot(t, M, label="M")
axes[1].plot(t, N, label="N")
axes[1].plot(t, H, label="H")
axes[1].grid(True)
axes[1].set_ylabel('Steady state val')
axes[1].set_title("Steady state values")

axes[2].plot(t, U, label="U")
axes[2].grid(True)
axes[2].set_xlabel('t[ms]')
axes[2].set_ylabel('U[mV]')
axes[2].set_title("Membrane potential")

plt.show()
