from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util.ml import *
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


THREADS = 12


def trial(Nk, dt, tmax):

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

    phi = 0.04

    x0 = np.array([-40.0, 0.42])

    # t_avg = 0.1
    I = current.constant(tmax, dt, I_ampl)
    # I = current.poisson(tmax, t_avg, dt, I_ampl)

    t = time.time()

    neuron = MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='euler', Nk=Nk)
    return neuron.signal(tmax, x0)


def _get_isi_wrapper(args):
    return get_isi2(trial(args[0], args[1], args[2]), args[3], args[4], args[1])


def main():

    epochs = 3200

    dt = 0.1
    tmax = 1000.0
    vdiff = 2.5 # orbit amplitude minimum
    sthresh = 20 # spiking threshold

    Nk = 2000000

    with Pool(THREADS) as p:
        isi = np.concatenate(p.map(_get_isi_wrapper, [(Nk, dt, tmax, vdiff, sthresh)] * epochs))
    
    plt.hist(isi, bins=200, range=(0, 600), label=f'$N_k = {Nk}$', density=True)
    plt.xlabel('ISI (ms)')
    plt.ylabel('Estimated Probability Density')
    plt.yscale('log')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

