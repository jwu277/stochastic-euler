from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
from util.ml import *
import matplotlib.pyplot as plt

import time


def trial(phi, Nk, w0, dt, tmax):

    # C ~ uF
    # g ~ mS
    # V ~ mV
    # t ~ ms
    
    I_ampl = 90
    C = 20
    gL = 2.0
    gCa = 4.4
    gK = 8
    VL = -60
    VCa = 120
    VK = -84
    V1 = -1.2
    V2 = 18.0
    V3 = 2.0
    V4 = 30.0

    x0 = np.array([-40.0, 0.42])

    # t_avg = 0.1
    I = current.constant(tmax, dt, I_ampl)
    # I = current.poisson(tmax, t_avg, dt, I_ampl)

    t = time.time()

    neuron = MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='euler', Nk=Nk)
    return neuron.signal(tmax, x0)


def main():

    epochs = 3

    dt = 0.1
    tmax = 100000.0
    vdiff = 2.5 # orbit amplitude minimum
    sthresh = 20 # spiking threshold

    Nk = 500

    isi = []

    for i in range(epochs):
        print(f'Starting epoch {i+1} of {epochs}...')
        sig = trial(Nk, dt, tmax)
        isi += list(get_isi2(sig, vdiff, sthresh, dt))
    
    plt.hist(isi, bins=100, range=(0, 600), label=f'$N_k = {Nk}$', density=True)
    plt.xlabel('ISI (ms)')
    plt.ylabel('Estimated Probability Density')
    plt.yscale('log')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

