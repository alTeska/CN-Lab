##ATTRACTOR NETWORKS
import matplotlib.pyplot as plt
import numpy as np

def activation_fun(I):
    if I <= 0: y = 0
    else: y = I
    return y

def activation_vec(vec, y):
    vfunc = np.vectorize(y)
    return vfunc(vec)

tau = 0.2

#ex 1 - without external input E(t)
dt = 0.01
t_end = 10
time_vec = np.arange(0, t_end, dt)

#a)
w = 0.8
v = 0.2

W = np.matrix([[w, v], [v, w]])
N = len(time_vec)

Y1 = np.zeros_like(time_vec)
Y2 = np.zeros_like(time_vec)
E = np.zeros_like(time_vec)
Y = np.zeros([2,N])
I = np.zeros([2,N])
I[0,0] = 2
I[1,0] = 0
Y[:,0] = activation_vec(I[:,0], activation_fun)
Y1[0] = Y[0,0]
Y2[0] = Y[1,0]

def dI_dt(I, y, E): return ( np.dot(W, y) - I + E)/tau

for t in range(0, len(time_vec)-1):
    I[:,t+1] = I[:,t] + dI_dt(I[:,t], Y[:,t], E[t]) * dt
    Y[:,t+1] = activation_vec(I[:,t+1], activation_fun)
    Y1[t+1] = Y[0,t+1]
    Y2[t+1] = Y[1,t+1]

#plots
fig, axes = plt.subplots()
axes.grid(True)
axes.plot(time_vec, Y1, label="y1")
axes.plot(time_vec, Y2, label="y2")
axes.legend(loc=2)
axes.set_ylabel('activity')
axes.set_title("time")

#EX 2
E = np.zeros_like(time_vec)
E[500:510] = 1
E[900:910] =-1

fig, axes = plt.subplots()
axes.grid(True)
axes.plot(time_vec, E)

for t in range(0, len(time_vec)-1):
    I[:,t+1] = I[:,t] + dI_dt(I[:,t], Y[:,t], E[t]) * dt
    #Y[:,t+1] = activation_vec(I[:,t+1], activation_fun)
    Y[:,t+1] = activation_vec(I[:,t+1], activation_fun)
    Y1[t+1] = Y[0,t+1]
    Y2[t+1] = Y[1,t+1]

#plots
fig, axes = plt.subplots()
axes.grid(True)
axes.plot(time_vec, Y1, label="y1")
axes.plot(time_vec, Y2, label="y2")
axes.legend(loc=2)
axes.set_ylabel('activity')
axes.set_title("time")

#EX3
def activation_sigm(I):
    y = (1 + np.tanh(I))/2
    return y

fig, axes = plt.subplots()
axes.grid(True)
axes.plot(time_vec, E)

for t in range(0, len(time_vec)-1):
    I[:,t+1] = I[:,t] + dI_dt(I[:,t], Y[:,t], E[t]) * dt
    Y[:,t+1] = activation_vec(I[:,t+1], activation_sigm)
    Y1[t+1] = Y[0,t+1]
    Y2[t+1] = Y[1,t+1]

#plots
fig, axes = plt.subplots()
axes.grid(True)
axes.plot(time_vec, Y1, label="y1")
axes.plot(time_vec, Y2, label="y2")
axes.legend(loc=2)
axes.set_ylabel('activity')
axes.set_title("time")

plt.show()
