from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
from util.ml import *
import matplotlib.pyplot as plt

import time


def trial(I_ampl, phi, Nk, dt, tmax):

    # C ~ uF
    # g ~ mS
    # V ~ mV
    # t ~ ms
    
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

    dt = 0.1
    tmax = 1000.0

    I_ampl = 90

    sig = trial(I_ampl, 0.04, 500, dt, tmax)
    np.save('picklejar/phi_noise_t99.npy', sig) # save to desired directory


if __name__ == "__main__":
    main()

