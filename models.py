'''
Collection of models to be used during Computational Neuroscience classes at MSNE TUM
'''
import matplotlib.pyplot as plt
import numpy as np

'''
class HHModel():
    def __init__(self, time_vec ,self.I):
        # Constants
        C_m  =   0.1 #membrane capacitance, in uF/cm^2
        g_Na = 120.0 #maximum conducances, in mS/cm^2
        g_K  =  36.0
        g_L  =   0.3
        E_Na =  55.0 #Nernst reversal potentials, in mV
        E_K  = -75.0
        E_L  = -69

        self.M = np.zeros_like(time)
        self.N = np.zeros_like(time)
        self.H = np.zeros_like(time)
        self.U = np.zeros_like(time)
'''
w = 0.8
v = 0.2
W = np.matrix([[w, v], [v, w]])

dt = 0.01
t_end = 10
time_vec = np.arange(0, t_end, dt)

class Attractor_Network:
    def __init__(self, W, E, time_vec):
        tau = 0.2
        self.N = len(time_vec)

        self.Y1 = np.zeros_like(time_vec)
        self.Y2 = np.zeros_like(time_vec)
        self.Y = np.zeros([2,self.N])
        self.I = np.zeros([2,self.N])
        self.I[0,0], [1,0] = 2, 0
        Y[:,0] = activation_vec(self.I[:,0], activation_fun)
        self.Y1[0], self.Y2[0] = self.Y[0,0], self.Y[1,0]

    def dI_dt(I, y, E):
         return ( np.dot(W, y) - I + E)/tau

    def iter(self):
        for t in range(0, self.N-1):
            I[:,t+1] = I[:,t] + dI_dt(I[:,t], Y[:,t], E[t]) * dt
            self.Y[:,t+1] = activation_vec(I[:,t+1], activation_fun)
            self.Y1[t+1] = self.Y[0,t+1]
            self.Y2[t+1] = self.Y[1,t+1]
