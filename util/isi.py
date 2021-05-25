### Utility for ISI calculations ###


import numpy as np


## Computes spike times from falling-edge zero crossings
def spike_times_zc(sig, dt):

    zc = []

    for i in range(len(sig) - 1):
        if sig[i] > 0 and sig[i+1] < 0:
            zc.append(i * dt)
    
    return np.array(zc)


## Computes spike times from threshold-reaching
def spike_times_th(sig, dt, th):

    zc = []

    for i in range(len(sig)):
        if sig[i] >= th:
            zc.append(i * dt)
    
    return np.array(zc)



## Computes ISIs from spike times
def isi(st):
    return np.diff(st)

